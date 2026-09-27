from collections import defaultdict


class _TrackState:
    __slots__ = ("votes", "locked_text", "locked_score", "hits")

    def __init__(self):
        self.votes = defaultdict(float)
        self.locked_text = None
        self.locked_score = 0.0
        self.hits = 0


class PlateStabilizer:
    """
    Keeps a confidence-weighted vote tally of OCR readings per tracked
    vehicle, and only changes the displayed plate text when a new
    candidate clearly outweighs the current one.

    This fixes the original behaviour where `plate_text = new_text` simply
    overwrote the cached value on every OCR pass — a single noisy or
    blurry frame could instantly replace an already-correct reading.
    Here, a reading needs to accumulate enough weighted evidence (and beat
    the incumbent by `switch_margin`) before it takes over, and old votes
    decay slowly so a single bad frame can't dominate.
    """

    def __init__(
        self,
        min_confidence: float = 0.35,
        switch_margin: float = 1.4,
        decay: float = 0.9,
        min_hits_to_lock: int = 2,
    ):
        self.min_confidence = min_confidence
        self.switch_margin = switch_margin
        self.decay = decay
        self.min_hits_to_lock = min_hits_to_lock
        self.tracks: dict[int, _TrackState] = {}

    def update(self, track_id: int, text: str | None, confidence: float) -> str | None:
        """Feed a new OCR reading for a track and return the current best
        (stable) plate text for that track."""
        state = self.tracks.setdefault(track_id, _TrackState())

        if text and confidence >= self.min_confidence:
            # Decay existing votes so stale/incorrect readings fade over time
            for key in list(state.votes.keys()):
                state.votes[key] *= self.decay
                if state.votes[key] < 1e-3:
                    del state.votes[key]

            state.votes[text] += confidence
            state.hits += 1

            best_text = max(state.votes, key=state.votes.get)
            best_score = state.votes[best_text]

            if state.locked_text is None:
                # Not locked yet — require a couple of hits before
                # committing to anything, so a single lucky/unlucky read
                # doesn't decide the outcome.
                if state.hits >= self.min_hits_to_lock:
                    state.locked_text = best_text
                    state.locked_score = best_score
            elif best_text != state.locked_text:
                # Only steal the lock if the challenger clearly dominates
                if best_score >= state.locked_score * self.switch_margin:
                    state.locked_text = best_text
                    state.locked_score = best_score
            else:
                state.locked_score = best_score

        return state.locked_text

    def get(self, track_id: int) -> str | None:
        state = self.tracks.get(track_id)
        return state.locked_text if state else None

    def cleanup(self, active_ids) -> None:
        """Drop state for tracks that are no longer active (left the frame)."""
        active = set(int(i) for i in active_ids)
        for tid in list(self.tracks.keys()):
            if tid not in active:
                del self.tracks[tid]