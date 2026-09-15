# Real-Time Vehicle Traffic Analysis

Detects vehicles from an RTSP camera stream, tracks them with persistent IDs,
classifies each vehicle's color, and reads license plates via OCR — tuned to
run in real time on edge/CPU hardware.

This version focuses on two upgrades over the baseline: **license-plate
detection** and **vehicle color detection**. Both were rewritten to be
substantially more accurate while staying model-free by default (no extra
downloads, no extra per-frame networks).

## 1. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**CPU-only edge devices:** install the CPU build of PyTorch first to avoid
pulling a large CUDA package:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The first run auto-downloads `yolov8n.pt` (~6 MB) and EasyOCR's models
(~100 MB) — make sure the machine has internet access at least once, or
pre-seed those files.

## 2. Usage

```bash
python main.py --source rtsp://user:password@192.168.1.10:554/Streaming/Channels/101
```

Optional: use a dedicated trained license-plate detector for maximum accuracy
(any Ultralytics YOLO plate-detection weights):

```bash
python main.py --source rtsp://... --plate-model lpd_yolo.pt
```

Useful flags:

| Flag             | Default      | Description                                                          |
|------------------|--------------|---------------------------------------------------------------------|
| `--source`       | *(required)* | RTSP URL, video file path, or webcam index (`0`) for local testing. |
| `--model`        | `yolov8n.pt` | Ultralytics weights for vehicle detection.                          |
| `--plate-model`  | *(none)*     | Optional trained YOLO plate detector. Falls back to classical CV.   |
| `--conf`         | `0.4`        | Vehicle detection confidence threshold.                             |
| `--imgsz`        | `640`        | YOLO inference resolution (lower = faster, less accurate).          |
| `--device`       | `cpu`        | `cpu`, `cuda:0`, etc.                                                |
| `--classes`      | `car`        | Comma list from `car,motorcycle,bus,truck`.                         |
| `--gpu-ocr`      | off          | Run EasyOCR on GPU.                                                  |
| `--no-display`   | off          | Headless mode (no `cv2.imshow` window).                             |
| `--display-width`| `800`        | Max width of the display window.                                    |

Quick local test with a webcam instead of RTSP:

```bash
python main.py --source 0
```

Press **`q`** in the video window to quit.

Offline sanity check of the color + plate-localization logic (no OCR/YOLO
required):

```bash
python _selftest.py
```

## 3. Project layout

```
main.py        Stream loop, tracking, per-track caching & temporal voting, drawing
color.py       Vehicle color estimation
plate.py       License-plate localization + OCR + cross-frame fusion
utils.py       Backward-compatibility shim re-exporting the above
_selftest.py   Offline unit checks for the pure-CV logic
```

## 4. What changed, and why

### License-plate detection (`plate.py`)

**Before:** the bottom ~55% of the vehicle box was handed straight to EasyOCR.
EasyOCR's detector then fired on *any* text in that region (badges, bumper
stickers, dealer frames, reflections), the plate characters were tiny, and a
tilted plate was never straightened — all of which hurt accuracy.

**Now** there is an explicit localize → rectify → read → fuse pipeline:

1. **Plate localization.** By default a model-free classical detector runs:
   blackhat morphology to bring out dark characters on the light plate, a
   Sobel horizontal-gradient response to catch the dense vertical strokes of
   text, morphological closing to merge characters into a single blob, then
   contour filtering by **aspect ratio (~1.8–6.5), area, and solidity** to keep
   only plate-shaped regions. This restricts OCR to a tight plate crop instead
   of the whole bumper. If you supply `--plate-model`, a trained YOLO plate
   detector is used instead for even better localization.
2. **Deskew.** Each candidate is straightened using its `minAreaRect` angle so
   the text is horizontal before recognition.
3. **Multi-variant OCR.** The plate crop is upscaled and OCR is run on several
   complementary renderings — CLAHE-enhanced grayscale, Otsu binary, and
   adaptive threshold — with an **A–Z 0–9 allowlist** so EasyOCR can't emit
   punctuation/latin-lookalike noise. The best-scoring result across variants
   wins.
4. **Plate-shape scoring.** Candidate strings are scored by a blend of OCR
   confidence and how plate-like they look (length, letter/digit mix, common
   patterns), so a confident read of a non-plate word doesn't beat a slightly
   less confident real plate.
5. **Cross-frame fusion.** Reads for a track are accumulated and fused
   (`fuse_plate_reads`): a confidence-weighted vote over whole strings plus a
   **per-character positional vote** among reads of the modal length. Different
   frames tend to misread *different* characters, so the fused string is
   typically correct even when no single frame was. OCR stops for a track once
   the fused confidence is high enough.

The localized plate is also drawn as a yellow rectangle on the frame.

### Vehicle color detection (`color.py`)

**Before:** one center crop was reduced to a single K-Means centroid and
classified with rigid HSV thresholds. Windshields/shadows often won the vote
(reported as black/blue), glare washed cars out to white/silver, and small
lighting shifts flipped neighbouring colors.

**Now:**

1. **Body-band sampling.** Only the central body band is used — the glassy
   top (windshield/roof), the bumper/plate/road-shadow at the bottom, and the
   side margins are excluded.
2. **Glass / specular masking.** Near-black glass-and-shadow pixels and
   blown-out specular highlights are masked out before clustering — unless
   doing so removes almost everything, which safely preserves genuinely black
   or white cars.
3. **Multi-cluster scoring.** K-Means produces several clusters and *every*
   cluster is scored (size + chroma, with a light penalty on dark near-neutral
   clusters), so a large shadow can no longer outvote the actual paint.
4. **Perceptual classification.** The winning cluster is matched to a small
   palette by nearest neighbour in **CIELAB** space (perceptually uniform),
   which is both simpler and steadier than hand-tuned HSV cut-offs. A chroma
   gate keeps faintly-tinted neutrals from being called a color.
5. **Confidence + temporal voting.** Each estimate returns a confidence, and
   `main.py` accumulates a **confidence-weighted vote** over the first several
   sightings before locking the color, instead of trusting one frame.

Verified on synthetic crops (with a dark windshield and a glare streak
present) the classifier gets red / blue / green / yellow / white / black /
silver all correct; see `_selftest.py`.

## 5. Edge / real-time optimizations (unchanged core)

- **Per-track caching** keyed by ByteTrack ID: color and plate are computed on
  a budget per track, not every frame.
- **Color vote-then-lock** after a few sightings; **OCR retry-then-lock** once
  a fused plate is confident. Vehicle crops below a minimum area skip OCR.
- **Global sparse mode:** after 30 empty frames, detection runs only every 5th
  frame until a vehicle reappears.
- **Cache eviction** after 300 unseen frames keeps memory bounded.
- **RTSP resilience:** automatic reconnect with exponential backoff.

## 6. Known limitations / next steps

- The classical localizer is tuned for roughly frontal/rear plates; extreme
  angles or motion blur still benefit from a trained `--plate-model`.
- Country-specific plate grammars could be added to the scorer to further cut
  misreads.
- No persistence layer yet (e.g. logging fused plate/color to CSV on track
  eviction) — straightforward to add where stale tracks are dropped in
  `main.py`.
