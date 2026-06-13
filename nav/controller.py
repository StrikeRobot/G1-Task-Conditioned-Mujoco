"""Simplified pure pursuit: heading control toward a lookahead point on the path.

The loco/policy_29dof.pt checkpoint turns in place, so on a large heading error
the robot rotates without moving forward; once roughly aligned it drives toward
the lookahead point and slows down as it nears the goal.
"""
import numpy as np


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class PurePursuit:
    def __init__(self, lookahead, v_max, w_max, goal_tol):
        self.lookahead = lookahead
        self.v_max, self.w_max = v_max, w_max
        self.goal_tol = goal_tol

    def compute(self, pose, path):
        """pose=(x,y,yaw); path=Nx2 world. Returns (vx, wz, done)."""
        p = np.asarray(pose[:2], dtype=float)
        path = np.asarray(path, dtype=float)
        if np.linalg.norm(path[-1] - p) < self.goal_tol:
            return 0.0, 0.0, True

        # Find the closest waypoint, then look AHEAD from there for the lookahead
        # point. (Scanning from path[0] would pick points already behind us once
        # the robot has advanced, making it spin toward a target at its back.)
        i_near = int(np.argmin(np.linalg.norm(path - p, axis=1)))
        target = path[-1]
        for j in range(i_near, len(path)):
            if np.linalg.norm(path[j] - p) > self.lookahead:
                target = path[j]
                break

        err = _wrap(np.arctan2(target[1] - p[1], target[0] - p[0]) - pose[2])
        wz = float(np.clip(2.0 * err, -self.w_max, self.w_max))
        if abs(err) > 0.6:
            return 0.0, wz, False  # large error -> rotate in place first
        # forward speed scaled by heading alignment and slowed near the goal
        d_goal = np.linalg.norm(path[-1] - p)
        vx = self.v_max * (1.0 - abs(err) / 0.6) * min(1.0, d_goal / 0.8)
        return float(vx), wz, False
