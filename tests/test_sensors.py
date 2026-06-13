import numpy as np

from sim.sensors import depth_to_hsv_bgr, quat_to_rpy, camera_fy


def test_quat_to_rpy_identity():
    assert np.allclose(quat_to_rpy(np.array([1.0, 0, 0, 0])), [0, 0, 0], atol=1e-9)


def test_quat_to_rpy_yaw90():
    q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    r, p, y = quat_to_rpy(q)
    assert abs(y - np.pi / 2) < 1e-6 and abs(r) < 1e-6 and abs(p) < 1e-6


def test_depth_colormap_shape_and_contrast():
    depth = np.full((10, 10), 0.5, np.float32)
    depth[:, 5:] = 6.0
    img = depth_to_hsv_bgr(depth, near=0.3, far=8.0)
    assert img.shape == (10, 10, 3) and img.dtype == np.uint8
    assert not np.array_equal(img[0, 0], img[0, 9]), "near and far must have different colors"


def test_camera_fy():
    assert abs(camera_fy(58.0, 480) - (240 / np.tan(np.radians(29)))) < 1e-6
