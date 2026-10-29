"""Temporal confirmation rule used for every reported result in the MSc project.

A clip counts as violent when its violence probability is at least
``VIOLENCE_THRESHOLD``. A video, or the live stream, raises an alert once
``ALERT_CONSECUTIVE_CLIPS`` violent clips occur in a row; a non-violent clip
resets the count. These are the dissertation settings (0.50 and 3 clips).

The module is pure Python (no PyTorch), so it can be unit-tested anywhere.
``predict.py``, ``ucf_evaluate.py`` and ``xd_evaluate.py`` use it directly.
``live_camera.py`` shares the same constants and applies the same rule inside
its streaming loop, which also tracks alert timing.
"""

VIOLENCE_THRESHOLD = 0.50
ALERT_CONSECUTIVE_CLIPS = 3


def is_violent_clip(violence_probability, threshold=VIOLENCE_THRESHOLD):
    """Return True when one clip's violence probability reaches the threshold."""
    return violence_probability >= threshold


class ConsecutiveClipCounter:
    """Count consecutive violent clips as predictions arrive, one clip at a time."""

    def __init__(self, threshold=VIOLENCE_THRESHOLD, required=ALERT_CONSECUTIVE_CLIPS):
        if required < 1:
            raise ValueError("required must be at least 1")
        self.threshold = threshold
        self.required = required
        self.count = 0
        self.max_count = 0

    def update(self, violence_probability):
        """Add one clip's probability and return whether an alert is now active."""
        if is_violent_clip(violence_probability, self.threshold):
            self.count += 1
            self.max_count = max(self.max_count, self.count)
        else:
            self.count = 0
        return self.alert_active

    @property
    def alert_active(self):
        return self.count >= self.required


def max_consecutive_violent_clips(probabilities, threshold=VIOLENCE_THRESHOLD):
    """Longest run of consecutive violent clips in a sequence of probabilities."""
    counter = ConsecutiveClipCounter(threshold=threshold)
    for violence_probability in probabilities:
        counter.update(violence_probability)
    return counter.max_count


def confirm_alert(
    probabilities,
    threshold=VIOLENCE_THRESHOLD,
    required=ALERT_CONSECUTIVE_CLIPS,
):
    """Video-level decision: True if any run of violent clips reaches ``required``."""
    if required < 1:
        raise ValueError("required must be at least 1")
    return max_consecutive_violent_clips(probabilities, threshold) >= required
