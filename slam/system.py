"""SLAM orchestrator: odometry predict -> scan match correct -> grid update."""
import numpy as np

from slam.grid import OccupancyGrid
from slam.matcher import ScanMatcher

# Number of scans that must be ingested before scan-matching is trusted.
MATCHER_WARMUP_SCANS = 5


class SlamSystem:
    """States: idle -> building <-> paused -> localized (end_build/load).

    Note: the ``localized`` state is terminal — the system stays in it until
    reset() is called.  This is intentional for the demo build-once workflow.
    """

    def __init__(self, cfg, max_range):
        self.cfg = cfg
        self.max_range = max_range
        self.state = "idle"
        self.pose = np.zeros(3)
        self._scan_count = 0
        self._new_grid()

    def _new_grid(self):
        c = self.cfg
        self.grid = OccupancyGrid(tuple(c["size_m"]), c["resolution"], c["l_occ"],
                                  c["l_free"], c["l_clamp"], self.max_range)
        self.matcher = ScanMatcher(self.grid, c["match_window_xy"],
                                   c["match_window_th"])

    # ---- controls ----
    def start(self):
        if self.state in ("idle", "paused"):
            self.state = "building"

    def pause(self):
        if self.state == "building":
            self.state = "paused"

    def end_build(self):
        if self.state in ("building", "paused"):
            self.state = "localized"

    def reset(self):
        self.state = "idle"
        self.pose = np.zeros(3)
        self._scan_count = 0
        self._new_grid()

    def set_pose(self, pose):
        self.pose = np.asarray(pose, dtype=float).copy()

    def save(self, path):
        self.grid.save(path)

    def load(self, path):
        self.grid.load(path)
        self.matcher = ScanMatcher(self.grid, self.cfg["match_window_xy"],
                                   self.cfg["match_window_th"])
        self.state = "localized"
        self._scan_count = MATCHER_WARMUP_SCANS

    # ---- main update 10Hz ----
    def on_scan(self, odom_delta, ranges, angles):
        if self.state in ("idle", "paused"):
            return
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        self.pose = self.pose + np.array([
            c * odom_delta[0] - s * odom_delta[1],
            s * odom_delta[0] + c * odom_delta[1],
            odom_delta[2]])
        if self._scan_count >= MATCHER_WARMUP_SCANS:
            self.pose, _ = self.matcher.match(ranges, angles, self.pose)
        if self.state == "building":
            self.grid.update(self.pose, ranges, angles)
            self._scan_count += 1
