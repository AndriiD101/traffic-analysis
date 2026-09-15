"""
Vehicle color estimation.

Design goals
------------
Keep it model-free and cheap enough to run on a CPU/edge device, but far more
robust than a single-centroid + hard-HSV-threshold classifier. The previous
version failed in three common situations:

  * windshields / windows and dark shadowed panels dominated the crop and got
    reported as the car's color (usually "black" or "blue");
  * specular highlights (sun glare on paint) washed a colored car out to
    "white" / "silver";
  * rigid HSV cut-offs flipped between neighbouring colors (e.g. gray<->blue,
    red<->brown) whenever lighting shifted slightly.

The approach here addresses each:

  1. Sample only the *body* band of the bounding box (skip roof/windshield at
     the top, bumper/plate/road-shadow at the bottom, and the side margins).
  2. Mask out pixels that are almost certainly not paint: near-black glass /
     shadow and blown-out specular highlights -- unless doing so would remove
     almost everything (genuinely black or white cars), in which case we keep
     them.
  3. Cluster the *remaining* pixels (K-Means) and score every cluster, not just
     the biggest one, so a large shadow can't outvote the actual paint. The
     score rewards cluster size and chroma (colored paint) and lightly
     penalises glassy/near-achromatic-dark clusters.
  4. Classify the winning cluster by nearest neighbour in CIELAB space against
     a small fixed palette. Lab is perceptually uniform, so nearest-anchor is
     both simpler and steadier than hand-tuned HSV boundaries. Achromatic
     colors are gated by a chroma test first so a faintly tinted gray doesn't
     read as a color.

Returns a (name, draw_bgr, confidence) triple. The confidence lets the caller
do temporal voting across frames instead of hard-locking after N samples.
"""

import cv2
import numpy as np


# Vivid, visually distinct BGR values used purely for *drawing* the box, so the
# overlay is consistent and readable regardless of the noisy sampled pixel.
COLOR_DRAW_BGR = {
    "white":  (245, 245, 245),
    "black":  (30, 30, 30),
    "gray":   (128, 128, 128),
    "silver": (200, 200, 200),
    "red":    (0, 0, 220),
    "blue":   (220, 70, 0),
    "green":  (0, 160, 0),
    "yellow": (0, 220, 220),
    "orange": (0, 140, 255),
    "brown":  (30, 65, 115),
    "unknown": (0, 255, 0),
}

# Palette anchors as sRGB (R, G, B). These are converted to CIELAB once at
# import time. Two shades are given for several colors (e.g. bright vs. dark
# red) so a single anchor doesn't have to cover a wide lightness range; both
# map back to the same reported name.
_PALETTE_RGB = {
    "black":  [(20, 20, 22), (55, 55, 58)],
    "white":  [(245, 245, 245), (220, 222, 225)],
    "gray":   [(120, 120, 122), (90, 90, 92)],
    "silver": [(180, 182, 185), (200, 202, 205)],
    "red":    [(190, 30, 34), (130, 25, 28)],
    "blue":   [(30, 60, 170), (20, 40, 95)],
    "green":  [(35, 130, 60), (20, 80, 45)],
    "yellow": [(235, 215, 40)],
    "orange": [(235, 130, 30)],
    "brown":  [(110, 70, 45), (80, 55, 40)],
}


def _build_lab_anchors():
    names, labs = [], []
    for name, rgb_list in _PALETTE_RGB.items():
        for r, g, b in rgb_list:
            bgr = np.uint8([[[b, g, r]]])
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
            names.append(name)
            labs.append(lab)
    return names, np.stack(labs, axis=0)


_ANCHOR_NAMES, _ANCHOR_LABS = _build_lab_anchors()
_ACHROMATIC = {"black", "white", "gray", "silver"}


def _classify_lab(lab_pixel):
    """
    Map one CIELAB pixel (OpenCV scale: L in [0,255], a,b in [0,255] centred
    at 128) to a palette name.

    Chroma is measured as distance of (a,b) from the neutral point 128. Low
    chroma -> restrict the match to achromatic anchors so a slightly tinted
    neutral doesn't get called "blue"/"green". Otherwise match against the
    full palette.
    """
    L, a, b = float(lab_pixel[0]), float(lab_pixel[1]), float(lab_pixel[2])
    chroma = np.hypot(a - 128.0, b - 128.0)

    if chroma < 12.0:
        mask = np.array([n in _ACHROMATIC for n in _ANCHOR_NAMES])
    else:
        mask = np.ones(len(_ANCHOR_NAMES), dtype=bool)

    d = np.linalg.norm(_ANCHOR_LABS - lab_pixel.astype(np.float32), axis=1)
    d = np.where(mask, d, np.inf)
    idx = int(np.argmin(d))
    return _ANCHOR_NAMES[idx], float(d[idx]), float(chroma)


def _body_region(crop):
    """Return the central body band, skipping windshield/roof and bumper."""
    h, w = crop.shape[:2]
    y0, y1 = int(h * 0.30), int(h * 0.82)   # drop glassy top and bumper/plate
    x0, x1 = int(w * 0.15), int(w * 0.85)   # drop side background/mirrors
    band = crop[y0:y1, x0:x1]
    return band if band.size else crop


def get_dominant_color(crop, k=4, sample_size=64):
    """
    Estimate a vehicle's paint color.

    Parameters
    ----------
    crop : np.ndarray
        BGR image of the vehicle bounding box.
    k : int
        Number of K-Means clusters.
    sample_size : int
        Side length the body band is resized to before clustering.

    Returns
    -------
    (name, draw_bgr, confidence)
        name       : color name, or "unknown".
        draw_bgr   : fixed BGR tuple to draw the box with.
        confidence : 0..1 estimate of how reliable this reading is (cluster
                     dominance blended with palette-match tightness). Intended
                     for temporal voting by the caller.
    """
    if crop is None or crop.size == 0:
        return "unknown", COLOR_DRAW_BGR["unknown"], 0.0

    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return "unknown", COLOR_DRAW_BGR["unknown"], 0.0

    band = _body_region(crop)
    small = cv2.resize(band, (sample_size, sample_size), interpolation=cv2.INTER_AREA)
    # Mild blur to suppress texture/JPEG noise before clustering.
    small = cv2.GaussianBlur(small, (3, 3), 0)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.int32)
    s = hsv[:, :, 1].astype(np.int32)

    # Candidate "paint" mask: drop near-black glass/shadow and blown highlights.
    paint_mask = (v > 45) & (v < 245)
    # A specular highlight is bright *and* desaturated; exclude those too.
    paint_mask &= ~((v > 225) & (s < 30))

    flat = small.reshape(-1, 3).astype(np.float32)
    mask_flat = paint_mask.reshape(-1)

    # If masking removed almost everything, the car is probably genuinely very
    # dark or very light -- fall back to using all pixels rather than guessing.
    if mask_flat.sum() < 0.15 * mask_flat.size:
        pixels = flat
    else:
        pixels = flat[mask_flat]

    if len(pixels) < k:
        pixels = flat  # too few to cluster meaningfully

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    k_eff = max(1, min(k, len(pixels)))

    try:
        if k_eff == 1:
            centers = pixels.mean(axis=0, keepdims=True)
            counts = np.array([len(pixels)])
        else:
            _c, labels, centers = cv2.kmeans(
                pixels, k_eff, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(labels.flatten(), minlength=k_eff)
    except cv2.error:
        centers = pixels.mean(axis=0, keepdims=True)
        counts = np.array([len(pixels)])

    total = float(counts.sum()) or 1.0

    # Score every cluster: prefer large, chromatic (colorful) paint clusters;
    # gently down-weight dark near-neutral clusters (residual glass/shadow).
    best = None
    for center, cnt in zip(centers, counts):
        frac = cnt / total
        bgr_px = np.uint8([[np.clip(center, 0, 255)]])
        lab_px = cv2.cvtColor(bgr_px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        name, dist, chroma = _classify_lab(lab_px)

        L = float(lab_px[0])
        chroma_bonus = min(chroma / 60.0, 1.0)          # 0..1, colorful paint
        dark_neutral_pen = 0.5 if (chroma < 12 and L < 60) else 1.0
        score = frac * (0.55 + 0.45 * chroma_bonus) * dark_neutral_pen

        # Match tightness -> palette confidence (closer anchor = more certain).
        match_conf = float(np.clip(1.0 - dist / 90.0, 0.0, 1.0))
        cand = {"name": name, "frac": frac, "score": score,
                "match_conf": match_conf}
        if best is None or score > best["score"]:
            best = cand

    if best is None:
        return "unknown", COLOR_DRAW_BGR["unknown"], 0.0

    confidence = float(np.clip(0.5 * best["frac"] + 0.5 * best["match_conf"], 0.0, 1.0))
    name = best["name"]
    return name, COLOR_DRAW_BGR.get(name, COLOR_DRAW_BGR["unknown"]), confidence
