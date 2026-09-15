"""
Backward-compatibility shim.

The pipeline logic now lives in two focused modules:
    * color.py -- vehicle color estimation
    * plate.py -- license-plate localization + OCR

This module re-exports the most commonly used names so older imports such as
``from utils import get_dominant_color, init_ocr_reader`` keep working.
"""

from color import get_dominant_color, COLOR_DRAW_BGR          # noqa: F401
from plate import init_ocr_reader, PlateReader, fuse_plate_reads  # noqa: F401


def extract_license_plate(image_crop, reader, **_kwargs):
    """
    Legacy one-shot helper kept for compatibility.

    Wraps the new PlateReader (classical localizer + multi-variant OCR) so old
    call sites that expect ``(text, confidence)`` still function. New code
    should construct a PlateReader once and call ``.read()`` to benefit from
    temporal fusion across frames.
    """
    reader_obj = PlateReader(reader)
    text, score, _box = reader_obj.read(image_crop)
    return text, score
