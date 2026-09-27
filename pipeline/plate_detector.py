import numpy as np
from ultralytics import YOLO


class PlateDetector:
    def __init__(self, weights: str = "./models/best2.pt", conf_threshold: float = 0.4):
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold

    def detect(self, vehicle_crop: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """
        Returns list of (x1, y1, x2, y2, conf) relative to vehicle_crop,
        sorted by confidence (highest first), filtered by conf_threshold.
        Sorting + thresholding matters: previously the first raw box from
        the model was always used regardless of confidence, which let
        low-quality / spurious detections feed garbage crops into OCR.
        """
        results = self.model(vehicle_crop, verbose=False)[0]
        plates = []
        for box in results.boxes:
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plates.append((x1, y1, x2, y2, conf))

        plates.sort(key=lambda p: p[4], reverse=True)
        return plates