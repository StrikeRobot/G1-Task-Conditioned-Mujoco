"""Snapshot state shared between sim/perception/web — guarded by a single lock."""
import threading


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._d = {
            "rgb_jpeg": None, "depth_jpeg": None, "map_png": None,
            "rgb_raw": None, "depth_raw": None, "cam_pos": None, "cam_mat": None,
            "lidar_ranges": [], "lidar_angles": [], "imu": None,
            "pose": (0.0, 0.0, 0.0), "mode": "manual", "fallen": False,
            "slam_state": "idle", "nav_status": "IDLE", "telegram": "MOCK",
            "detections": [], "path": [], "goal": None, "map_meta": None,
        }

    def update(self, **kw):
        with self._lock:
            self._d.update(kw)

    def snapshot(self):
        with self._lock:
            return dict(self._d)

    def get(self, key):
        with self._lock:
            return self._d[key]
