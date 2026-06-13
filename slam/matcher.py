"""Correlative scan matching on an occupancy grid (coarse-to-fine grid search)."""
import cv2
import numpy as np


class ScanMatcher:
    def __init__(self, grid, window_xy=0.15, window_th=0.08):
        self.grid = grid
        self.wxy = window_xy
        self.wth = window_th
        self._field_version = -1
        self._field = None

    def match(self, ranges, angles, guess):
        """Return (pose_est, score). guess = (x,y,yaw) predicted from odometry."""
        g = self.grid
        if self._field is None or self._field_version != getattr(g, "version", None):
            occ = (g.logodds > 0.7).astype(np.float32)
            self._field = cv2.GaussianBlur(occ, (7, 7), 1.5)
            self._field_version = getattr(g, "version", None)
        field = self._field
        hit = ranges < g.max_range - 1e-3
        r, a = ranges[hit], angles[hit]
        if len(r) < 20:
            return np.asarray(guess, dtype=float), 0.0

        def score(pose):
            x, y, th = pose
            ex = x + r * np.cos(th + a)
            ey = y + r * np.sin(th + a)
            ix = ((ex - g.origin[0]) / g.res).astype(int)
            iy = ((ey - g.origin[1]) / g.res).astype(int)
            ok = (ix >= 0) & (ix < g.w) & (iy >= 0) & (iy < g.h)
            if not np.any(ok):
                return 0.0
            return float(field[iy[ok], ix[ok]].sum()) / len(r)

        best = np.asarray(guess, dtype=float)
        best_s = score(best)
        # 3 coarse-to-fine passes, 5x5x5 grid around best
        for sx, sth in (
            (self.wxy / 2, self.wth / 2),
            (self.wxy / 8, self.wth / 8),
            (self.wxy / 32, self.wth / 32),
        ):
            center = best.copy()
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        c = center + np.array([i * sx / 2, j * sx / 2, k * sth / 2])
                        s = score(c)
                        if s > best_s:
                            best_s, best = s, c
        return best, best_s
