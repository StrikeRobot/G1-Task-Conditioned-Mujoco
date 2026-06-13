import numpy as np

from nav.controller import PurePursuit
from nav.navigator import Navigator
from slam.grid import OccupancyGrid


def _pp():
    return PurePursuit(lookahead=0.6, v_max=0.5, w_max=1.0, goal_tol=0.3)


def test_pp_drives_forward_on_straight_path():
    path = np.array([[0.5, 0], [1.0, 0], [2.0, 0], [3.0, 0]])
    vx, wz, done = _pp().compute(np.array([0, 0, 0.0]), path)
    assert vx > 0.25 and abs(wz) < 0.1 and not done


def test_pp_rotates_in_place_when_facing_wrong_way():
    """policy_29dof.pt turns in place -> on a large heading error, rotate without
    forward motion (vx=0) and yaw toward the target."""
    path = np.array([[1.0, 0], [2.0, 0]])
    vx, wz, done = _pp().compute(np.array([0, 0, np.pi / 2]), path)
    assert vx == 0.0 and wz <= -0.5 and not done  # spin back toward +x


def test_pp_done_at_goal():
    path = np.array([[1.0, 0]])
    vx, wz, done = _pp().compute(np.array([0.9, 0.05, 0.0]), path)
    assert done and vx == 0 and wz == 0


class _FakeSlam:
    def __init__(self):
        self.grid = OccupancyGrid((10, 10), 0.1, 0.85, -0.4, 4.0, max_range=8.0)
        self.grid.logodds[:] = -1.0
        self.pose = np.zeros(3)


NAV_CFG = {"robot_radius": 0.2, "lookahead": 0.6, "v_max": 0.5, "w_max": 1.0,
           "goal_tol": 0.3, "occ_threshold": 0.7, "patrol_waypoints": []}


def test_navigator_full_cycle():
    slam = _FakeSlam()
    nav = Navigator(NAV_CFG, slam)
    assert nav.status.startswith("IDLE")
    ok = nav.set_goal(2.0, 0.0)
    assert ok and nav.status.startswith("NAVIGATING")
    vx, vy, wz = nav.update(np.array([0.0, 0.0, 0.0]))
    assert vx > 0
    nav.update(np.array([1.95, 0.0, 0.0]))
    assert nav.status.startswith("DONE")
    nav.stop()
    assert nav.status.startswith("IDLE")


def test_navigator_rejects_goal_in_obstacle():
    slam = _FakeSlam()
    ix, iy = slam.grid.world_to_map(2.0, 2.0)
    slam.grid.logodds[iy - 3:iy + 4, ix - 3:ix + 4] = 4.0
    nav = Navigator(NAV_CFG, slam)
    assert not nav.set_goal(2.0, 2.0)


def test_navigator_rejects_goal_outside_map():
    slam = _FakeSlam()
    nav = Navigator(NAV_CFG, slam)
    assert not nav.set_goal(40.0, 0.0)
    assert nav.status == "GOAL OUT OF MAP"
    slam.pose = np.array([-20.0, 0.0, 0.0])
    assert not nav.set_goal(1.0, 0.0)


def test_navigator_patrol_advances_waypoints():
    slam = _FakeSlam()
    cfg = dict(NAV_CFG, patrol_waypoints=[[1.0, 0.0], [2.0, 0.0]])
    nav = Navigator(cfg, slam)
    assert nav.start_patrol()
    assert "wp 1/2" in nav.status
    nav.update(np.array([0.95, 0.0, 0.0]))   # reach wp1 -> move to wp2
    assert "wp 2/2" in nav.status
    nav.update(np.array([1.95, 0.0, 0.0]))
    assert nav.status.startswith("DONE")
