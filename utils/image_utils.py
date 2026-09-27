import cv2
import numpy as np


def crop_region(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop region dari image. Clamp agar tidak out of bounds."""
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


def _upscale(gray: np.ndarray, target_height: int = 120) -> np.ndarray:
    """Plate crops are often tiny (<40px tall) which kills OCR accuracy.
    Upscale with cubic interpolation so character strokes are thick enough
    for EasyOCR's recognizer to resolve them."""
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return gray
    if h >= target_height:
        return gray
    scale = target_height / h
    return cv2.resize(gray, (int(w * scale), target_height), interpolation=cv2.INTER_CUBIC)


def preprocess_for_ocr(crop: np.ndarray) -> list[np.ndarray]:
    """
    Generate several preprocessed variants of the plate crop and let the
    OCR stage try all of them, keeping whichever yields the highest
    confidence reading. A single Otsu threshold is fragile to glare,
    shadows, and motion blur — this is why readings used to jump around
    frame to frame. Returns a list of grayscale/binary images (RGB->GRAY).
    """
    if crop.size == 0:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = _upscale(gray)

    # Mild denoise before any thresholding
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # CLAHE improves local contrast, helps with uneven lighting/glare
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    variants = []

    # 1. CLAHE-enhanced grayscale (often best for EasyOCR directly)
    variants.append(contrast)

    # 2. Otsu threshold on the contrast-enhanced image
    _, otsu = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # 3. Inverted Otsu (helps when plate is dark text on light bg vs reverse)
    variants.append(cv2.bitwise_not(otsu))

    # 4. Adaptive threshold, robust to uneven illumination across the plate
    adaptive = cv2.adaptiveThreshold(
        contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    variants.append(adaptive)

    return variants


def trim_plate_crop(vehicle_crop: np.ndarray, px1, py1, px2, py2) -> np.ndarray:
    """Trim bagian atas plat (area kota) dan sisi kiri/kanan (bingkai/cover
    dudukan plat) agar OCR lebih akurat. Bingkai plat (mis. cover plastik,
    baut) yang ikut terdeteksi sebagai bagian box sering muncul sebagai
    garis vertikal tipis di tepi kiri/kanan dan terbaca OCR sebagai 'I'."""
    w, h = px2 - px1, py2 - py1
    px1_new = max(0, px1 + int(0.07 * w))
    px2_new = min(vehicle_crop.shape[1], px2 - int(0.07 * w))
    py1_new = max(0, py1 + int(0.08 * h))
    py2_new = min(vehicle_crop.shape[0], py2 - int(0.35 * h))
    return vehicle_crop[py1_new:py2_new, px1_new:px2_new]