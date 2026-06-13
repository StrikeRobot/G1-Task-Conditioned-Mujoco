"""Occupancy grid log-odds, vectorized update (no per-cell bresenham)."""
import numpy as np


class OccupancyGrid:
    def __init__(self, size_m, resolution, l_occ, l_free, l_clamp, max_range):
        self.res = resolution
        self.w = int(round(size_m[0] / resolution))
        self.h = int(round(size_m[1] / resolution))
        self.origin = np.array([-size_m[0] / 2.0, -size_m[1] / 2.0])
        self.l_occ, self.l_free, self.l_clamp = l_occ, l_free, l_clamp
        self.max_range = max_range
        self.logodds = np.zeros((self.h, self.w), dtype=np.float32)
        self.version = 0

    def world_to_map(self, x, y):
        ix = int(np.floor((x - self.origin[0]) / self.res))
        iy = int(np.floor((y - self.origin[1]) / self.res))
        return ix, iy

    def map_to_world(self, ix, iy):
        return (self.origin[0] + (ix + 0.5) * self.res,
                self.origin[1] + (iy + 0.5) * self.res)

    def update(self, pose, ranges, angles):
        """pose=(x,y,yaw) world; ranges/angles in robot frame."""
        x, y, th = pose
        ranges = np.asarray(ranges, dtype=np.float32)
        dirs = np.stack([np.cos(th + angles), np.sin(th + angles)], axis=1)
        step = self.res * 0.8
        n_steps = int(self.max_range / step)
        ts = (np.arange(1, n_steps + 1) * step)[None, :]
        hit_mask = ranges < self.max_range - 1e-3
        free_len = np.where(hit_mask, ranges - self.res, ranges)[:, None]
        valid = ts < free_len
        px = x + dirs[:, 0:1] * ts
        py = y + dirs[:, 1:2] * ts
        self._add(np.stack([px[valid], py[valid]], -1), self.l_free)
        ex = x + dirs[hit_mask, 0] * ranges[hit_mask]
        ey = y + dirs[hit_mask, 1] * ranges[hit_mask]
        self._add(np.stack([ex, ey], -1), self.l_occ)
        np.clip(self.logodds, -self.l_clamp, self.l_clamp, out=self.logodds)
        self.version += 1

    def _add(self, pts_xy, val):
        if len(pts_xy) == 0:
            return
        ij = ((pts_xy - self.origin) / self.res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.w) & (ij[:, 1] >= 0) & (ij[:, 1] < self.h)
        ij = ij[ok]
        # np.unique deduplicates flat indices so each cell gets exactly +val once per scan (no double-weight)
        flat = np.unique(ij[:, 1] * self.w + ij[:, 0])
        self.logodds.flat[flat] += val

    def occupied_mask(self, thr=0.7):
        return self.logodds > thr

    def to_image(self):
        """uint8 HxW: free=255, occupied=0, unknown=200. Row 0 = smallest y."""
        img = np.full((self.h, self.w), 200, np.uint8)
        img[self.logodds < -0.3] = 255
        img[self.logodds > 0.7] = 0
        return img

    def save(self, path):
        np.savez_compressed(path, logodds=self.logodds,
                            origin=self.origin, res=self.res)

    def load(self, path):
        z = np.load(path)
        self.logodds = z["logodds"].astype(np.float32)
        self.h, self.w = self.logodds.shape
        self.origin = z["origin"]
        self.res = float(z["res"])
        self.version += 1
