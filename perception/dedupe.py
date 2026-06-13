"""Spatial detection dedupe: each (class, region of radius R) is reported once."""
import numpy as np


class DetectionDeduper:
    def __init__(self, radius=1.0):
        self.radius = radius
        self.seen = []  # list[(cls, x, y)]

    def is_new(self, cls, xy):
        for c, x, y in self.seen:
            if c == cls and np.hypot(xy[0] - x, xy[1] - y) < self.radius:
                return False
        self.seen.append((cls, xy[0], xy[1]))
        return True

    def reset(self):
        self.seen.clear()
