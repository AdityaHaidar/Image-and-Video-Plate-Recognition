import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, weights: str = "./models/best.pt", conf_threshold: float = 0.3):
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """
        Returns list of (x1, y1, x2, y2, conf) untuk setiap mobil terdeteksi.
        Model custom (YOLO11) dilatih khusus untuk objek 'car', jadi semua
        deteksi model ini sudah pasti kendaraan — tidak perlu filter class
        COCO (VEHICLE_CLASSES) seperti versi yolov8n.pt sebelumnya.
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append((x1, y1, x2, y2, conf))
        return detections