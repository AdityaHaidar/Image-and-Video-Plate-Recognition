import numpy as np
import sys
import os

# Pastikan folder sort/ ada di path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sort'))
from sort import Sort


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class VehicleTracker:
    """
    Wraps SORT with two stability improvements:

    1. Looser/longer SORT parameters: raw SORT only emits a track in a
       given frame when it was matched that exact frame, and silently
       drops (loses the ID forever) any track unmatched for more than
       `max_age` frames. With the old defaults (max_age=10) a single
       short occlusion, motion blur, or a couple of low-confidence
       detection frames was enough for a vehicle's ID to reset, which
       made the plate cache/stabilizer think it was a brand-new vehicle.
       max_age is raised and iou_threshold relaxed so brief gaps survive.

    2. An ID re-linking layer on top of SORT: if SORT *still* drops a
       track (e.g. occlusion longer than max_age) and a new raw ID shows
       up shortly after in roughly the same place, we transparently remap
       it back to the old stable ID instead of exposing a new one to the
       rest of the pipeline. This is what keeps plate_cache/stabilizer
       continuity even across a full SORT track reset.
    """

    def __init__(
        self,
        max_age: int = 40,
        min_hits: int = 3,
        iou_threshold: float = 0.2,
        relink_iou: float = 0.3,
        relink_window: int = 90,
    ):
        self.tracker = Sort(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )

        self.relink_iou = relink_iou
        self.relink_window = relink_window

        self.frame_idx = 0
        self.raw_to_stable: dict[int, int] = {}      # SORT id -> our stable id
        self.last_bbox: dict[int, tuple] = {}         # SORT id -> last known bbox
        self.lost_tracks: dict[int, dict] = {}        # stable id -> {"bbox", "frame"}
        self._next_stable_id = 1

    def update(self, detections: list[tuple[int, int, int, int, float]]) -> np.ndarray:
        """
        Input:  list of (x1, y1, x2, y2, conf)
        Output: numpy array shape (N, 5) -> [x1, y1, x2, y2, stable_track_id]
        """
        self.frame_idx += 1

        if len(detections) == 0:
            raw_tracked = self.tracker.update(np.empty((0, 5)))
        else:
            dets = np.array(detections, dtype=float)[:, :5]
            raw_tracked = self.tracker.update(dets)

        current_raw_ids = set(int(t[4]) for t in raw_tracked)

        # Any raw id we were tracking that vanished this frame becomes
        # "lost" (kept around for relink_window frames in case it reappears)
        vanished_raw_ids = set(self.raw_to_stable.keys()) - current_raw_ids
        for raw_id in vanished_raw_ids:
            stable_id = self.raw_to_stable.pop(raw_id)
            bbox = self.last_bbox.get(raw_id)
            if bbox is not None:
                self.lost_tracks[stable_id] = {"bbox": bbox, "frame": self.frame_idx}

        output = []
        for trk in raw_tracked:
            x1, y1, x2, y2, raw_id = trk
            raw_id = int(raw_id)
            bbox = (x1, y1, x2, y2)
            self.last_bbox[raw_id] = bbox

            if raw_id in self.raw_to_stable:
                stable_id = self.raw_to_stable[raw_id]
            else:
                stable_id = self._resolve_stable_id(bbox)
                self.raw_to_stable[raw_id] = stable_id

            output.append([x1, y1, x2, y2, stable_id])

        # Expire lost tracks that have been gone too long to be relinked
        self.lost_tracks = {
            sid: info for sid, info in self.lost_tracks.items()
            if self.frame_idx - info["frame"] <= self.relink_window
        }

        if not output:
            return np.empty((0, 5))
        return np.array(output, dtype=float)

    def _resolve_stable_id(self, bbox) -> int:
        """For a brand-new raw SORT id, check whether it overlaps closely
        with a recently lost track's last position. If so, reuse that
        stable id instead of minting a new one."""
        best_sid, best_iou = None, 0.0
        for sid, info in self.lost_tracks.items():
            score = _iou(bbox, info["bbox"])
            if score > best_iou:
                best_iou, best_sid = score, sid

        if best_sid is not None and best_iou >= self.relink_iou:
            del self.lost_tracks[best_sid]
            return best_sid

        new_id = self._next_stable_id
        self._next_stable_id += 1
        return new_id