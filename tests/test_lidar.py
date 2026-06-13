import mujoco
import numpy as np
import yaml

from sim.sensors import lidar_scan

BOX_SCENE = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    <geom name="wall_e" type="box" pos="3 0 0.5" size="0.1 5 0.5"/>
    <geom name="wall_n" type="box" pos="0 2 0.5" size="5 0.1 0.5"/>
    <body name="rob" pos="0 0 0.5">
      <geom name="rob_body" type="sphere" size="0.2" group="2"/>
    </body>
  </worldbody>
</mujoco>
"""

def test_lidar_distances():
    m = mujoco.MjModel.from_xml_string(BOX_SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ranges, angles = lidar_scan(m, d, origin=[0, 0, 0.5], yaw=0.0,
                                n_rays=360, max_range=8.0, min_range=0.35)
    i_e = np.argmin(np.abs(angles - 0.0))
    i_n = np.argmin(np.abs(angles - np.pi / 2))
    i_w = np.argmin(np.abs(angles - np.pi * 0.999))
    assert abs(ranges[i_e] - 2.9) < 0.05
    assert abs(ranges[i_n] - 1.9) < 0.05
    assert ranges[i_w] >= 7.9
    assert np.all(ranges > 0.35), "must not self-hit"


def test_lidar_in_real_scene():
    from sim.loader import load_scene
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    m, meta = load_scene(cfg["sim"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    origin = d.xpos[meta["lidar_body_id"]] + [0, 0, cfg["lidar"]["mount_z_offset"]]
    r, a = lidar_scan(m, d, origin, 0.0, 360, 8.0, cfg["lidar"]["min_range"])
    assert np.sum(r < 8.0) > 50, f"lidar must see the environment, hits={np.sum(r < 8.0)}"
    assert np.all(r > cfg["lidar"]["min_range"]), "self-hit on the robot"
