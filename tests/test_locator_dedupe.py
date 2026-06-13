import numpy as np

from perception.dedupe import DetectionDeduper
from perception.locator import pixel_to_world


def test_pixel_to_world_center():
    p = pixel_to_world(u=320, v=240, depth=2.0, cam_pos=np.zeros(3),
                       cam_mat=np.eye(3).flatten(), fovy_deg=58.0,
                       img_wh=(640, 480))
    assert np.allclose(p, [0, 0, -2.0], atol=1e-6)


def test_pixel_to_world_offset_direction():
    p = pixel_to_world(u=480, v=240, depth=2.0, cam_pos=np.zeros(3),
                       cam_mat=np.eye(3).flatten(), fovy_deg=58.0,
                       img_wh=(640, 480))
    assert p[0] > 0.3 and abs(p[1]) < 1e-6


def test_dedupe():
    dd = DetectionDeduper(radius=1.0)
    assert dd.is_new("cup", (0.0, 0.0))
    assert not dd.is_new("cup", (0.3, 0.2))
    assert dd.is_new("wrench", (0.3, 0.2))
    assert dd.is_new("cup", (3.0, 0.0))
    dd.reset()
    assert dd.is_new("cup", (0.0, 0.0))
