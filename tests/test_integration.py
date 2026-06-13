"""End-to-end headless: drive robot, build map, detect, mock telegram."""
import glob

import numpy as np
import pytest

from apprt.runtime import PatrolApp


@pytest.mark.slow
@pytest.mark.render
def test_full_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    app = PatrolApp("configs/config.yaml", enable_detector=True)
    app.telegram.out_dir = tmp_path
    # load_dotenv() in __init__ may load a real .env -> force MOCK on the instance
    app.telegram.token = ""
    app.telegram.chat_id = ""
    app.set_mode("manual")
    app.slam.start()

    plan = [((1.0, 0, 0), 4.0), ((0, 0, 1.0), 3.0), ((1.0, 0, 0), 4.0)]
    for cmd, dur in plan:
        app.set_teleop(*cmd)
        for _ in range(int(dur / app.model.opt.timestep)):
            app._sim_tick()
        app._perception_tick()

    snap = app.state.snapshot()
    assert not snap["fallen"], "robot must not fall in the standard scenario"
    assert np.sum(app.slam.grid.occupied_mask()) > 200, "map must have walls/objects"
    gt = app.policy.base_pose2d()
    assert np.linalg.norm(np.array(snap["pose"][:2]) - gt[:2]) < 1.0, \
        "SLAM pose must track the true pose (scan matching curbs drift)"
    app.telegram.close()
    if snap["detections"]:
        assert len(glob.glob(str(tmp_path / "*.jpg"))) >= 1
