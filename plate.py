"""
License-plate localization and OCR.

Why this is different from feeding the lower half of the car to EasyOCR
----------------------------------------------------------------------
The original pipeline cropped the bottom ~55% of the vehicle box and ran
EasyOCR on the whole thing. That has three problems:

  * EasyOCR's text detector fires on *any* text -- badges, bumper stickers,
    dealer frames, reflections -- not just the plate, so the "best" read is
    often not the plate at all;
  * the plate occupies a tiny fraction of that region, so the characters are
    small and the recognizer is working near its resolution limit;
  * a skewed/angled plate (very common from a roadside camera) is never
    rectified, which hurts recognition badly.

This module adds an explicit *localization* stage before OCR:

  1. Find plate candidates. By default this is a classical, model-free
     detector (blackhat + Sobel gradient + morphological closing + contour
     filtering by aspect ratio / solidity). Optionally, if you pass a trained
     YOLO plate model path, it is used instead for higher accuracy.
  2. Deskew each candidate with its minAreaRect angle so text is horizontal.
  3. Upscale and run OCR on several preprocessing variants (grayscale+CLAHE,
     Otsu binary, adaptive threshold) with an alphanumeric allowlist, keeping
     the best-scoring result.
  4. Validate the string against plausible plate shape and return it with a
     confidence and the plate bbox (in vehicle-crop coordinates) for drawing.

The public entry point is ``PlateReader.read(vehicle_crop)``.
"""

import re
import cv2
import numpy as np


_ALNUM_RE = re.compile(r"[^A-Z0-9]")
# Common confusions when OCR-ing plate glyphs. Used only to build an
# *alternative* normalized candidate, never to overwrite the raw read.
_OCR_CONFUSION = str.maketrans({"O": "0", "I": "1", "Q": "0", "Z": "2", "B": "8"})

# Plausible plate strings: a mix of letters and digits, or an all-digit plate,
# 4-9 chars. Used for scoring, not as a hard gate (so unusual formats still
# surface, just with a lower score).
_PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$"),
    re.compile(r"^[0-9]{1,4}[A-Z]{1,3}[0-9]{0,4}$"),
    re.compile(r"^[0-9]{4,8}$"),
]


def init_ocr_reader(languages=("en",), gpu=False):
    """Construct the EasyOCR reader once. Heavy -- call at startup only."""
    import easyocr  # local import so the cost is only paid when actually used
    return easyocr.Reader(list(languages), gpu=gpu, verbose=False)


def _score_plate_text(text, ocr_conf):
    """Blend OCR confidence with how plate-shaped the string looks (0..1)."""
    if not text:
        return 0.0
    n = len(text)
    if n < 4 or n > 9:
        length_score = 0.2
    else:
        length_score = 1.0
    has_digit = any(c.isdigit() for c in text)
    has_alpha = any(c.isalpha() for c in text)
    mix_score = 1.0 if (has_digit and (has_alpha or n >= 4)) else 0.4
    pattern_score = 1.0 if any(p.match(text) for p in _PLATE_PATTERNS) else 0.6
    shape = (length_score + mix_score + pattern_score) / 3.0
    return float(0.5 * ocr_conf + 0.5 * shape)


def _deskew(plate_bgr):
    """Rotate a plate crop so its dominant text baseline is horizontal."""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thr)
    if coords is None or len(coords) < 20:
        return plate_bgr
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    if abs(angle) < 1.5:      # not worth rotating
        return plate_bgr
    h, w = plate_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(plate_bgr, M, (w, h),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def fuse_plate_reads(reads):
    """
    Fuse several noisy plate reads of the *same* vehicle into one consensus.

    A single frame's OCR often gets one character wrong; different frames tend
    to get *different* characters wrong. Combining them across time is far more
    reliable than trusting the single highest-confidence frame.

    Two mechanisms are combined:
      * weighted string vote -- sum each distinct string's scores;
      * per-character positional vote among reads whose length equals the modal
        length, weighted by score, to synthesize a consensus string that may
        beat any individual read.

    Parameters
    ----------
    reads : list[tuple[str, float]]
        (text, score) pairs accumulated for one track.

    Returns
    -------
    (text, score) or (None, 0.0)
    """
    if not reads:
        return None, 0.0

    # Weighted vote over whole strings.
    string_score = {}
    for text, score in reads:
        string_score[text] = string_score.get(text, 0.0) + score
    top_text = max(string_score, key=string_score.get)
    top_score = string_score[top_text]

    # Per-character consensus among reads sharing the modal length.
    len_weight = {}
    for text, score in reads:
        len_weight[len(text)] = len_weight.get(len(text), 0.0) + score
    modal_len = max(len_weight, key=len_weight.get)
    same_len = [(t, s) for t, s in reads if len(t) == modal_len]

    consensus = top_text
    if len(same_len) >= 3:
        chars = []
        for i in range(modal_len):
            col = {}
            for t, s in same_len:
                col[t[i]] = col.get(t[i], 0.0) + s
            chars.append(max(col, key=col.get))
        consensus = "".join(chars)

    # Prefer the consensus if it is itself well-supported.
    if consensus in string_score and string_score[consensus] >= top_score:
        chosen = consensus
    elif consensus not in string_score and len(same_len) >= 3:
        chosen = consensus  # synthesized, backed by positional majority
    else:
        chosen = top_text

    total = sum(string_score.values()) or 1.0
    support = string_score.get(chosen, top_score) / total
    # Confidence grows with agreement and number of reads (saturating).
    n_conf = min(len(reads) / 4.0, 1.0)
    confidence = float(min(0.4 + 0.6 * support * n_conf + 0.05 * len(reads), 1.0))
    return chosen, confidence


class PlateReader:
    """Localize + OCR license plates from vehicle crops."""

    def __init__(self, ocr_reader, plate_model=None, device="cpu",
                 min_plate_area=350, max_candidates=3):
        self.reader = ocr_reader
        self.device = device
        self.min_plate_area = min_plate_area
        self.max_candidates = max_candidates

        # Optional dedicated plate detector (trained YOLO weights).
        self.detector = None
        if plate_model:
            from ultralytics import YOLO
            self.detector = YOLO(plate_model)

        # Reusable morphology kernels for the classical localizer.
        self._rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        self._sq_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    # ------------------------------------------------------------------ #
    # Candidate localization
    # ------------------------------------------------------------------ #
    def _candidates_model(self, crop):
        """Plate boxes from the optional trained YOLO detector."""
        res = self.detector.predict(crop, device=self.device, verbose=False)
        boxes = []
        if res and res[0].boxes is not None:
            for b in res[0].boxes.xyxy.cpu().numpy().astype(int):
                x1, y1, x2, y2 = b[:4]
                if (x2 - x1) * (y2 - y1) >= self.min_plate_area:
                    boxes.append((x1, y1, x2, y2))
        return boxes[: self.max_candidates]

    def _candidates_classical(self, crop):
        """
        Model-free plate localization via morphology + contour geometry.
        Searches the lower ~65% of the vehicle box (plates sit low, front or
        rear) and returns boxes in full-crop coordinates.
        """
        h, w = crop.shape[:2]
        y_off = int(h * 0.35)
        region = crop[y_off:h, 0:w]
        if region.size == 0:
            return []

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 40, 40)

        # Blackhat reveals dark characters on a lighter plate background.
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self._rect_kern)

        # Bright plate-body regions (helps mask the gradient result).
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, self._sq_kern)
        light = cv2.threshold(light, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Horizontal gradient emphasises the vertical strokes of characters.
        gx = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gx = np.absolute(gx)
        mn, mx = gx.min(), gx.max()
        if mx - mn < 1e-6:
            return []
        gx = (255 * (gx - mn) / (mx - mn)).astype("uint8")

        gx = cv2.GaussianBlur(gx, (5, 5), 0)
        gx = cv2.morphologyEx(gx, cv2.MORPH_CLOSE, self._rect_kern)
        thr = cv2.threshold(gx, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        thr = cv2.erode(thr, None, iterations=2)
        thr = cv2.dilate(thr, None, iterations=2)
        thr = cv2.bitwise_and(thr, thr, mask=light)
        thr = cv2.dilate(thr, None, iterations=2)
        thr = cv2.erode(thr, None, iterations=1)

        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:12]

        region_area = region.shape[0] * region.shape[1]
        cands = []
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw == 0 or ch == 0:
                continue
            ar = cw / float(ch)
            area = cw * ch
            solidity = cv2.contourArea(c) / float(area + 1e-6)
            # Plates: wide, not too tall, reasonable fill, sensible size.
            if not (1.8 <= ar <= 6.5):
                continue
            if area < self.min_plate_area or area > 0.5 * region_area:
                continue
            if solidity < 0.25:
                continue
            pad_x, pad_y = int(0.04 * cw), int(0.20 * ch)
            x1 = max(x - pad_x, 0)
            y1 = max(y - pad_y + y_off, 0)
            x2 = min(x + cw + pad_x, w)
            y2 = min(y + ch + pad_y + y_off, h)
            score = area * min(ar, 4.0)      # favour bigger, plate-shaped boxes
            cands.append((score, (x1, y1, x2, y2)))

        cands.sort(key=lambda t: t[0], reverse=True)
        return [b for _, b in cands[: self.max_candidates]]

    def _candidates(self, crop):
        if self.detector is not None:
            boxes = self._candidates_model(crop)
            if boxes:
                return boxes
        return self._candidates_classical(crop)

    # ------------------------------------------------------------------ #
    # OCR on a localized plate
    # ------------------------------------------------------------------ #
    def _preprocess_variants(self, plate_bgr):
        """Yield a few complementary grayscale renderings for OCR."""
        # Upscale so character height is comfortably readable.
        target_h = 96
        h = plate_bgr.shape[0]
        if h < target_h:
            s = target_h / float(max(h, 1))
            plate_bgr = cv2.resize(plate_bgr, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_CUBIC)

        plate_bgr = _deskew(plate_bgr)
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

        variants = []
        variants.append(self._clahe.apply(gray))                       # contrast
        variants.append(cv2.threshold(                                 # Otsu
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        variants.append(cv2.adaptiveThreshold(                         # adaptive
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 9))
        return variants

    def _ocr_best(self, plate_bgr):
        best_text, best_score, best_conf = None, 0.0, 0.0
        for var in self._preprocess_variants(plate_bgr):
            try:
                results = self.reader.readtext(
                    var, detail=1, paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            except Exception:
                continue
            # A plate may be split across tokens; consider each token and also
            # the left-to-right concatenation of tokens on a similar row.
            tokens = []
            for bbox, text, conf in results:
                cleaned = _ALNUM_RE.sub("", text.upper())
                if cleaned:
                    ys = [p[1] for p in bbox]
                    xs = [p[0] for p in bbox]
                    tokens.append((min(xs), np.mean(ys), cleaned, float(conf)))

            candidates = []
            for _x, _y, cleaned, conf in tokens:
                candidates.append((cleaned, conf))
            if len(tokens) >= 2:
                tokens.sort(key=lambda t: t[0])          # left-to-right
                joined = "".join(t[2] for t in tokens)
                jconf = float(np.mean([t[3] for t in tokens]))
                candidates.append((joined, jconf))

            for cleaned, conf in candidates:
                if not (4 <= len(cleaned) <= 9):
                    continue
                score = _score_plate_text(cleaned, conf)
                if score > best_score:
                    best_text, best_score, best_conf = cleaned, score, conf

        return best_text, best_score, best_conf

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def read(self, vehicle_crop):
        """
        Detect and read a plate from a vehicle crop.

        Returns
        -------
        (text, confidence, plate_box)
            text       : plate string, or None if nothing plausible was read.
            confidence : blended OCR+shape score in 0..1.
            plate_box  : (x1, y1, x2, y2) of the plate within the vehicle crop,
                         or None. Useful for drawing the plate rectangle.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None, 0.0, None
        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 20:
            return None, 0.0, None

        best = (None, 0.0, None)
        boxes = self._candidates(vehicle_crop)

        if not boxes:
            # Fallback: OCR the lower band directly (old behaviour) so we still
            # get *something* when localization fails on an odd crop.
            band = vehicle_crop[int(h * 0.45):h, 0:w]
            text, score, _conf = self._ocr_best(band)
            if text:
                return text, score, None
            return None, 0.0, None

        for (x1, y1, x2, y2) in boxes:
            plate = vehicle_crop[y1:y2, x1:x2]
            if plate.size == 0:
                continue
            text, score, _conf = self._ocr_best(plate)
            if text and score > best[1]:
                best = (text, score, (x1, y1, x2, y2))

        return best
