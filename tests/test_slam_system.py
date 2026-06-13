import numpy as np

from slam.odometry import NoisyOdometry
from slam.system import SlamSystem
from tests.test_matcher import _room_scan

CFG = {
    "size_m": [12.0, 12.0], "resolution": 0.05, "l_occ": 0.85, "l_free": -0.4,
    "l_clamp": 4.0, "match_window_xy": 0.15, "match_window_th": 0.08,
    "odom_noise": {"v": 0.03, "w": 0.02, "bias_w": 0.002},
}


def test_odometry_drifts_but_reasonable():
    odo = NoisyOdometry(noise_v=0.05, noise_w=0.03, bias_w=0.005, seed=1)
    pose = np.zeros(3)
    for _ in range(100):
        d = odo.step(np.array([1.0, 0.0]), 0.0, 0.1)
        c, s = np.cos(pose[2]), np.sin(pose[2])
        pose[0] += c * d[0] - s * d[1]
        pose[1] += s * d[0] + c * d[1]
        pose[2] += d[2]
    err = abs(pose[0] - 10.0) + abs(pose[1])
    assert 0.001 < err < 3.0, f"must have drift but not unreasonable: {pose}"


def test_slam_states():
    s = SlamSystem(CFG, max_range=8.0)
    assert s.state == "idle"
    s.start(); assert s.state == "building"
    s.pause(); assert s.state == "paused"
    s.start(); assert s.state == "building"
    s.end_build(); assert s.state == "localized"
    s.reset(); assert s.state == "idle"


def test_slam_builds_map_and_tracks(tmp_path):
    s = SlamSystem(CFG, max_range=8.0)
    s.start()
    true = np.array([0.0, 0.0, 0.0])
    s.set_pose(true)
    for _ in range(30):
        true = true + np.array([0.05, 0, 0])
        r, a = _room_scan(true)
        s.on_scan(odom_delta=np.array([0.05, 0, 0]), ranges=r, angles=a)
    assert np.sum(s.grid.occupied_mask()) > 100, "map must have walls"
    assert np.linalg.norm(s.pose[:2] - true[:2]) < 0.3, f"tracking wrong: {s.pose} vs {true}"
    p = tmp_path / "room.npz"
    s.save(p)
    s2 = SlamSystem(CFG, max_range=8.0)
    s2.load(p)
    assert s2.state == "localized"
    assert np.sum(s2.grid.occupied_mask()) > 100
