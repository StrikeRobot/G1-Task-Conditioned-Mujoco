import numpy as np
import pytest

from apprt.runtime import PatrolApp


@pytest.fixture(scope="module")
def app():
    return PatrolApp("configs/config.yaml", enable_detector=False)


@pytest.mark.slow
@pytest.mark.render
def test_sim_ticks_produce_state(app):
    app.set_mode("manual")
    app.set_teleop(1.0, 0.0, 0.0)   # full forward
    app.slam.start()
    for _ in range(1500):           # 3s sim
        app._sim_tick()
    snap = app.state.snapshot()
    assert snap["rgb_jpeg"] is not None
    assert snap["depth_jpeg"] is not None
    assert snap["map_png"] is not None
    assert len(snap["lidar_ranges"]) == 360
    assert snap["imu"] is not None
    assert not snap["fallen"]
    assert np.sum(app.slam.grid.occupied_mask()) > 50, "map must have walls"


@pytest.mark.slow
@pytest.mark.render
def test_mode_switch_and_nav(app):
    app.slam.end_build()
    app.set_mode("auto")
    x, y, _ = app.slam.pose
    assert app.nav.set_goal(x + 0.5, y)
    vx, vy, wz = app.nav.update(app.slam.pose)
    assert vx > 0 or abs(wz) > 0
