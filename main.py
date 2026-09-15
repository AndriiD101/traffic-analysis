"""
Real-time vehicle traffic analysis over an RTSP stream.

Pipeline per frame:
  1. YOLOv8n detects vehicles; ByteTrack (built into Ultralytics) assigns and
     persists a track ID per vehicle across frames.
  2. For each tracked vehicle, color and license plate are computed on a budget
     (a few frames per track, then fused/locked) instead of every frame -- the
     main edge-optimization lever.
       * Color: confidence-weighted vote over the first few sightings, using a
         glass/shadow-masked, multi-cluster, CIELAB-nearest-anchor classifier
         (see color.py).
       * Plate: the plate is *localized* first (classical morphology, or an
         optional trained YOLO plate model), deskewed, OCR'd on several
         preprocessing variants with an alphanumeric allowlist, then fused
         across frames into a consensus string (see plate.py).
  3. Boxes are drawn in the vehicle's detected color, annotated with the track
     ID, class, color and plate text; the plate rectangle is drawn when found.

Usage:
    python main.py --source rtsp://user:pass@192.168.1.10:554/stream1
    python main.py --source rtsp://... --plate-model lpd_yolo.pt   # optional
"""

import argparse
import time
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

from color import get_dominant_color, COLOR_DRAW_BGR
from plate import PlateReader, init_ocr_reader, fuse_plate_reads

# COCO class IDs relevant to "vehicles". 2=car, 3=motorcycle, 5=bus, 7=truck.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# --- Tunable pipeline constants -------------------------------------------
COLOR_SAMPLE_COUNT = 6         # sightings to vote over before locking color
OCR_RETRY_INTERVAL = 12        # frames between OCR attempts while unlocked
PLATE_LOCK_CONF = 0.80         # fused confidence at which OCR stops for a track
PLATE_MAX_READS = 12           # cap on stored reads per track (memory bound)
MIN_PLATE_CROP_AREA = 1200     # px^2; skip OCR on vehicle crops too small
TRACK_TTL_FRAMES = 300         # drop cached data for tracks unseen this long

EMPTY_FRAMES_BEFORE_SKIP = 30  # consecutive empty frames before we throttle
SKIP_INTERVAL_WHEN_EMPTY = 5   # only run detection every Nth frame when empty

DISPLAY_MAX_WIDTH = 800
FONT_SCALE = 0.85
FONT_THICKNESS = 2
BOX_THICKNESS = 2
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Real-time vehicle traffic analysis over RTSP.")
    p.add_argument("--source", required=True,
                   help="RTSP URL of the camera stream (e.g. rtsp://user:pass@ip:554/stream1). "
                        "A local video file path or webcam index also works for testing.")
    p.add_argument("--model", default="yolov8n.pt",
                   help="Ultralytics YOLO weights for vehicle detection. Default: yolov8n.pt.")
    p.add_argument("--plate-model", default=None,
                   help="Optional trained YOLO license-plate detector weights. If omitted, a "
                        "model-free classical localizer is used.")
    p.add_argument("--conf", type=float, default=0.4, help="Vehicle detection confidence threshold.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size fed to YOLO.")
    p.add_argument("--device", default="cuda", help="Inference device: 'cpu', 'cuda:0', etc.")
    p.add_argument("--classes", default="car",
                   help="Comma-separated vehicle classes: car,motorcycle,bus,truck. Default: car.")
    p.add_argument("--gpu-ocr", action="store_true", help="Run EasyOCR on GPU instead of CPU.")
    p.add_argument("--no-display", action="store_true",
                   help="Run headless (no cv2.imshow window) - useful on servers.")
    p.add_argument("--max-retries", type=int, default=10,
                   help="Max consecutive reconnect attempts before giving up on the stream.")
    p.add_argument("--display-width", type=int, default=DISPLAY_MAX_WIDTH,
                   help="Max width (px) of the display window. Default: 800.")
    return p.parse_args()


class RTSPStream:
    """cv2.VideoCapture wrapper that reconnects on read failure with backoff."""

    def __init__(self, source, max_retries=10, retry_delay=2.0, retry_delay_cap=30.0):
        self.source = source
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_delay_cap = retry_delay_cap
        self.cap = None
        if not self._open():
            raise RuntimeError(f"Could not open video source on startup: {source}")

    def _open(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # stay near the live edge
        except Exception:
            pass
        return self.cap.isOpened()

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            if not self._reconnect():
                return False, None
        ok, frame = self.cap.read()
        if not ok:
            if not self._reconnect():
                return False, None
            ok, frame = self.cap.read()
        return ok, frame

    def _reconnect(self):
        delay = self.retry_delay
        for attempt in range(1, self.max_retries + 1):
            print(f"[stream] connection lost. Reconnect attempt {attempt}/{self.max_retries} "
                  f"to {self.source} ...")
            if self._open():
                print("[stream] reconnected successfully.")
                return True
            time.sleep(delay)
            delay = min(delay * 1.5, self.retry_delay_cap)
        print("[stream] failed to reconnect after max retries. Giving up.")
        return False

    def release(self):
        if self.cap is not None:
            self.cap.release()


def new_cache_entry(frame_idx):
    return {
        # --- color voting ---
        "color_votes": defaultdict(float),   # name -> summed confidence
        "color_bgr": {},                     # name -> draw bgr
        "color_samples": 0,
        "color_name": None,
        "box_color": COLOR_DRAW_BGR["unknown"],
        "color_locked": False,
        # --- plate voting ---
        "plate_reads": [],                   # list[(text, score)]
        "plate_text": None,
        "plate_conf": 0.0,
        "plate_locked": False,
        "plate_box_abs": None,               # (x1,y1,x2,y2) in frame coords
        "frames_since_ocr": OCR_RETRY_INTERVAL,
        # --- bookkeeping ---
        "last_seen": frame_idx,
    }


def update_color(cache, crop):
    """Accumulate a confidence-weighted color vote; lock after enough samples."""
    if cache["color_locked"]:
        return
    name, bgr, conf = get_dominant_color(crop)
    if name == "unknown":
        return
    cache["color_votes"][name] += conf
    cache["color_bgr"][name] = bgr
    cache["color_samples"] += 1

    best = max(cache["color_votes"], key=cache["color_votes"].get)
    cache["color_name"] = best
    cache["box_color"] = cache["color_bgr"][best]
    if cache["color_samples"] >= COLOR_SAMPLE_COUNT:
        cache["color_locked"] = True


def update_plate(cache, crop, plate_reader, origin):
    """Run localization+OCR on a schedule and fuse reads across frames."""
    if cache["plate_locked"]:
        return
    if cache["frames_since_ocr"] < OCR_RETRY_INTERVAL:
        cache["frames_since_ocr"] += 1
        return
    cache["frames_since_ocr"] = 0

    text, score, box = plate_reader.read(crop)
    if not text:
        return

    cache["plate_reads"].append((text, score))
    if len(cache["plate_reads"]) > PLATE_MAX_READS:
        cache["plate_reads"].pop(0)

    if box is not None:
        ox, oy = origin
        cache["plate_box_abs"] = (ox + box[0], oy + box[1], ox + box[2], oy + box[3])

    fused_text, fused_conf = fuse_plate_reads(cache["plate_reads"])
    if fused_text:
        cache["plate_text"] = fused_text
        cache["plate_conf"] = fused_conf
        if fused_conf >= PLATE_LOCK_CONF:
            cache["plate_locked"] = True


def draw_annotation(frame, box, track_id, cache, class_name,
                    font_scale, font_thickness, box_thickness):
    x1, y1, x2, y2 = box
    color = cache["box_color"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

    # Plate rectangle (if we localized one), drawn in a contrasting yellow.
    pb = cache["plate_box_abs"]
    if pb is not None:
        cv2.rectangle(frame, (pb[0], pb[1]), (pb[2], pb[3]), (0, 255, 255),
                      max(1, box_thickness - 1))

    label_parts = [f"ID {track_id}", class_name]
    if cache["color_name"]:
        label_parts.append(cache["color_name"])
    if cache["plate_text"]:
        label_parts.append(cache["plate_text"])
    label = " | ".join(label_parts)

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                         font_scale, font_thickness)
    label_y1 = max(y1 - th - baseline - 8, 0)
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 8, y1), color, -1)
    brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)
    cv2.putText(frame, label, (x1 + 4, y1 - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                font_thickness, cv2.LINE_AA)


def main():
    args = parse_args()

    requested = {c.strip().lower() for c in args.classes.split(",") if c.strip()}
    class_ids = [cid for cid, name in VEHICLE_CLASSES.items() if name in requested]
    if not class_ids:
        print(f"[config] No valid classes matched '{args.classes}', defaulting to 'car'.")
        class_ids = [2]

    print(f"[init] Loading YOLO model '{args.model}' on device '{args.device}' ...")
    model = YOLO(args.model)

    print("[init] Loading EasyOCR reader (this can take a moment on first run) ...")
    ocr_reader = init_ocr_reader(languages=("en",), gpu=args.gpu_ocr)

    if args.plate_model:
        print(f"[init] Loading dedicated plate detector '{args.plate_model}' ...")
    else:
        print("[init] Using model-free classical plate localizer.")
    plate_reader = PlateReader(ocr_reader, plate_model=args.plate_model, device=args.device)

    print(f"[init] Opening stream: {args.source}")
    source = int(args.source) if args.source.isdigit() else args.source
    stream = RTSPStream(source, max_retries=args.max_retries)

    track_cache = defaultdict(lambda: None)
    frame_idx = 0
    consecutive_empty = 0
    fps_smoothed = 0.0
    prev_time = time.time()
    window_ready = False

    print("[run] Starting main loop. Press 'q' in the video window to quit.")
    try:
        while True:
            ok, frame = stream.read()
            if not ok:
                print("[run] Stream ended or unrecoverable. Exiting.")
                break
            frame_idx += 1

            # Frame filtering: throttle detection during long empty spells.
            run_detection = True
            if consecutive_empty >= EMPTY_FRAMES_BEFORE_SKIP:
                run_detection = (frame_idx % SKIP_INTERVAL_WHEN_EMPTY == 0)

            detections = []
            if run_detection:
                results = model.track(
                    frame, persist=True, classes=class_ids, conf=args.conf,
                    imgsz=args.imgsz, device=args.device,
                    tracker="bytetrack.yaml", verbose=False)
                boxes = results[0].boxes
                if boxes is None or boxes.id is None or len(boxes) == 0:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0
                    xyxy = boxes.xyxy.cpu().numpy().astype(int)
                    ids = boxes.id.cpu().numpy().astype(int)
                    clss = boxes.cls.cpu().numpy().astype(int)
                    for b, tid, cid in zip(xyxy, ids, clss):
                        detections.append((b, int(tid), int(cid)))

            h_frame, w_frame = frame.shape[:2]
            seen_ids = set()

            display_scale = min(1.0, args.display_width / w_frame) if w_frame > 0 else 1.0
            f_scale = FONT_SCALE / display_scale if display_scale > 0 else FONT_SCALE
            f_thick = max(1, round(FONT_THICKNESS / display_scale))
            b_thick = max(1, round(BOX_THICKNESS / display_scale))

            for box, track_id, cls_id in detections:
                x1, y1, x2, y2 = box
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w_frame - 1), min(y2, h_frame - 1)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                seen_ids.add(track_id)
                if track_cache[track_id] is None:
                    track_cache[track_id] = new_cache_entry(frame_idx)
                cache = track_cache[track_id]
                cache["last_seen"] = frame_idx

                update_color(cache, crop)

                area = (x2 - x1) * (y2 - y1)
                if area >= MIN_PLATE_CROP_AREA:
                    update_plate(cache, crop, plate_reader, origin=(x1, y1))

                class_name = VEHICLE_CLASSES.get(cls_id, "vehicle")
                draw_annotation(frame, (x1, y1, x2, y2), track_id, cache, class_name,
                                f_scale, f_thick, b_thick)

            # Evict stale tracks to keep memory bounded.
            stale = [tid for tid, c in track_cache.items()
                     if c is not None and frame_idx - c["last_seen"] > TRACK_TTL_FRAMES]
            for tid in stale:
                del track_cache[tid]

            # FPS overlay.
            now = time.time()
            inst = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            fps_smoothed = fps_smoothed * 0.9 + inst * 0.1 if fps_smoothed else inst
            status = f"FPS: {fps_smoothed:.1f} | Tracks: {len(seen_ids)}"
            if consecutive_empty >= EMPTY_FRAMES_BEFORE_SKIP:
                status += " | mode: sparse-skip"
            cv2.putText(frame, status,
                        (int(10 / display_scale), int(30 / display_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, f_scale, (0, 255, 0), f_thick, cv2.LINE_AA)

            if not args.no_display:
                if display_scale < 1.0:
                    disp = cv2.resize(
                        frame,
                        (int(round(w_frame * display_scale)), int(round(h_frame * display_scale))),
                        interpolation=cv2.INTER_AREA)
                else:
                    disp = frame
                if not window_ready:
                    cv2.namedWindow("Traffic Analysis", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Traffic Analysis", disp.shape[1], disp.shape[0])
                    window_ready = True
                cv2.imshow("Traffic Analysis", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[run] 'q' pressed. Exiting.")
                    break
    except KeyboardInterrupt:
        print("[run] Interrupted by user.")
    finally:
        stream.release()
        cv2.destroyAllWindows()
        print("[run] Shutdown complete.")


if __name__ == "__main__":
    main()
