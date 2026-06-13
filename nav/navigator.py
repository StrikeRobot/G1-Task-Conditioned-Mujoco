"""Navigator FSM: world goal -> A* on the inflated grid -> follow the path.

Coordinate note: world_to_map returns (ix, iy) but astar uses (row, col) = (iy, ix).
The Navigator swaps the order before calling astar then swaps back for map_to_world.
"""
import numpy as np

from nav.astar import astar, inflate
from nav.controller import PurePursuit


class Navigator:
    def __init__(self, cfg, slam):
        self.cfg = cfg
        self.slam = slam
        self.pp = PurePursuit(cfg["lookahead"], cfg["v_max"], cfg["w_max"],
                              cfg["goal_tol"])
        self.path = None
        self.goal = None
        self.waypoints = list(cfg.get("patrol_waypoints") or [])
        self.wp_index = 0
        self._patrolling = False
        self.status = "IDLE"

    def _wp_suffix(self):
        if self._patrolling and self.waypoints:
            return f" (wp {self.wp_index + 1}/{len(self.waypoints)})"
        return ""

    def set_goal(self, x, y):
        g = self.slam.grid
        occ = inflate(g.logodds > self.cfg["occ_threshold"],
                      max(1, int(self.cfg["robot_radius"] / g.res)))
        # world_to_map returns (ix, iy); astar uses (row, col) = (iy, ix)
        ix_s, iy_s = g.world_to_map(*self.slam.pose[:2])
        ix_g, iy_g = g.world_to_map(x, y)
        if not (0 <= ix_s < g.w and 0 <= iy_s < g.h and
                0 <= ix_g < g.w and 0 <= iy_g < g.h):
            self.status = "GOAL OUT OF MAP"
            return False
        start_rc = (iy_s, ix_s)
        goal_rc = (iy_g, ix_g)
        occ[iy_s, ix_s] = False
        cells = astar(occ, start_rc, goal_rc)
        if cells is None:
            self.status = "NO PATH"
            return False
        # cells is list[(row, col)]; drop the first cell (current position), convert to world
        pts = [g.map_to_world(col, row) for row, col in cells[1:]] or [(x, y)]
        self.path = np.array(pts)
        self.goal = (x, y)
        self.status = "NAVIGATING" + self._wp_suffix()
        return True

    def start_patrol(self):
        if not self.waypoints:
            return False
        self._patrolling = True
        self.wp_index = 0
        return self.set_goal(*self.waypoints[0])

    def stop(self):
        self.path = None
        self.goal = None
        self.wp_index = 0
        self._patrolling = False
        self.status = "IDLE"

    def update(self, pose):
        """Called at 10Hz in auto mode. Returns (vx, vy, wz)."""
        if self.path is None:
            return 0.0, 0.0, 0.0
        vx, wz, done = self.pp.compute(pose, self.path)
        if done:
            if self._patrolling and self.wp_index + 1 < len(self.waypoints):
                self.wp_index += 1
                self.set_goal(*self.waypoints[self.wp_index])
                return 0.0, 0.0, 0.0
            self.path = None
            self._patrolling = False
            self.status = "DONE"
            return 0.0, 0.0, 0.0
        return vx, 0.0, wz
