import numpy as np
import easyocr
from utils.image_utils import preprocess_for_ocr, trim_plate_crop
from utils.plate_format import clean_plate_text, format_indonesian_plate


class OCRReader:
    def __init__(self, gpu: bool = True):
        self.reader = easyocr.Reader(['en'], gpu=gpu)

    @staticmethod
    def _is_edge_artifact(bbox, text: str, img_width: int, edge_frac: float = 0.12) -> bool:
        """
        Plate covers/holders often have a vertical bar or screw right at
        the left/right edge of the detected plate box. EasyOCR frequently
        misreads that thin vertical stroke as a stray 'I', '1', or 'L'
        fragment, which then gets glued onto the real plate text (e.g.
        "H 1234 UG" -> "IH 1234 UGI"). We drop a fragment only when it is
        BOTH a lone artifact-like character AND sits right against the
        crop's left/right boundary — genuine plate characters in the
        middle of the plate are never affected by this check.
        """
        if len(text) > 1 or text not in ('I', '1', 'L', '|'):
            return False

        xs = [p[0] for p in bbox]
        x_min, x_max = min(xs), max(xs)
        width = x_max - x_min

        near_left = x_min < img_width * edge_frac
        near_right = x_max > img_width * (1 - edge_frac)
        narrow = width < img_width * 0.15

        return narrow and (near_left or near_right)

    def _read_single(self, image: np.ndarray) -> tuple[str | None, float]:
        """
        Run EasyOCR on one preprocessed image variant and return
        (formatted_plate_text, confidence).

        IMPORTANT: paragraph=False is used deliberately. The original code
        used paragraph=True, which merges text blocks but EasyOCR does NOT
        return confidence scores in that mode — so the pipeline had zero
        signal about how trustworthy a reading was, and could not tell a
        good read from a garbage one. Here we read word/line boxes
        individually (with confidence), sort them left-to-right, and
        combine them ourselves.
        """
        results = self.reader.readtext(
            image,
            paragraph=False,
            detail=1,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        if not results:
            return None, 0.0

        # Sort fragments left-to-right by their bounding box x position,
        # since a plate may be split into multiple text fragments
        # (e.g. "B" / "1234" / "ABC").
        results.sort(key=lambda r: r[0][0][0])

        img_width = image.shape[1]
        last_idx = len(results) - 1

        parts = []
        confs = []
        for idx, (bbox, text, conf) in enumerate(results):
            cleaned = clean_plate_text(text)
            if not cleaned:
                continue

            # Only the first/last fragment can be a frame artifact — text
            # in the middle of the plate is never the cover/screw bar.
            if idx in (0, last_idx) and self._is_edge_artifact(bbox, cleaned, img_width):
                continue

            parts.append(cleaned)
            confs.append(conf)

        if not parts:
            return None, 0.0

        combined = ''.join(parts)
        formatted = format_indonesian_plate(combined)
        if not formatted:
            return None, 0.0

        avg_conf = float(np.mean(confs))
        return formatted, avg_conf

    def read_plate(
        self, vehicle_crop: np.ndarray, px1: int, py1: int, px2: int, py2: int
    ) -> tuple[str | None, float]:
        """
        Baca plat dari vehicle_crop dengan koordinat plat.
        Returns (formatted plate string atau None, confidence 0..1).

        Tries several preprocessing variants of the same crop and keeps
        the single best-confidence valid reading, instead of trusting
        whatever a single fragile threshold happens to produce.
        """
        ocr_crop = trim_plate_crop(vehicle_crop, px1, py1, px2, py2)
        if ocr_crop.size == 0:
            return None, 0.0

        variants = preprocess_for_ocr(ocr_crop)
        if not variants:
            return None, 0.0

        best_text, best_conf = None, 0.0
        for variant in variants:
            text, conf = self._read_single(variant)
            if text and conf > best_conf:
                best_text, best_conf = text, conf

        return best_text, best_conf