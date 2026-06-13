import numpy as np

from slam.grid import OccupancyGrid
from slam.matcher import ScanMatcher


def _room_scan(pose, n=180):
    """Synthetic scan of an 8x8 square room (walls at ±4) seen from pose."""
    x, y, th = pose
    angles = np.linspace(-np.pi, np.pi, n, endpoint=False)
    ranges = np.zeros(n, dtype=np.float32)
    for i, a in enumerate(angles):
        dx, dy = np.cos(th + a), np.sin(th + a)
        ts = []
        if dx > 1e-9: ts.append((4 - x) / dx)
        if dx < -1e-9: ts.append((-4 - x) / dx)
        if dy > 1e-9: ts.append((4 - y) / dy)
        if dy < -1e-9: ts.append((-4 - y) / dy)
        ranges[i] = min(t for t in ts if t > 0)
    return ranges, angles


def test_matcher_recovers_offset():
    g = OccupancyGrid((12, 12), 0.05, 0.85, -0.4, 4.0, max_range=10.0)
    true_pose = np.array([0.5, -0.3, 0.2])
    r, a = _room_scan(true_pose)
    for _ in range(8):
        g.update(true_pose, r, a)
    m = ScanMatcher(g, window_xy=0.15, window_th=0.08)
    guess = true_pose + np.array([0.10, -0.08, 0.05])
    est, score = m.match(r, a, guess)
    assert np.linalg.norm(est[:2] - true_pose[:2]) < 0.04, f"xy off: {est}"
    assert abs(est[2] - true_pose[2]) < 0.03, f"theta off: {est}"
