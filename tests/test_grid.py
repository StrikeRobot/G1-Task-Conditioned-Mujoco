import numpy as np

from slam.grid import OccupancyGrid


def _grid():
    return OccupancyGrid(size_m=(10.0, 10.0), resolution=0.1,
                         l_occ=0.85, l_free=-0.4, l_clamp=4.0, max_range=8.0)


def test_world_map_roundtrip():
    g = _grid()
    ix, iy = g.world_to_map(0.0, 0.0)
    assert (ix, iy) == (50, 50)
    x, y = g.map_to_world(50, 50)
    assert abs(x) < 0.1 and abs(y) < 0.1


def test_update_marks_occupied_and_free():
    g = _grid()
    ranges = np.array([2.0], dtype=np.float32)
    angles = np.array([0.0])
    for _ in range(5):
        g.update((0.0, 0.0, 0.0), ranges, angles)
    hit = g.world_to_map(2.0, 0.0)
    mid = g.world_to_map(1.0, 0.0)
    assert g.logodds[hit[1], hit[0]] > 0.7
    assert g.logodds[mid[1], mid[0]] < -0.3


def test_no_endpoint_when_max_range():
    g = _grid()
    g.update((0.0, 0.0, 0.0), np.array([8.0], np.float32), np.array([0.0]))
    assert not np.any(g.logodds > 0.5)


def test_save_load_roundtrip(tmp_path):
    g = _grid()
    g.update((0, 0, 0), np.array([2.0], np.float32), np.array([0.0]))
    p = tmp_path / "m.npz"
    g.save(p)
    g2 = _grid()
    g2.load(p)
    assert np.allclose(g.logodds, g2.logodds)


def test_to_image_values():
    g = _grid()
    for _ in range(5):
        g.update((0, 0, 0), np.array([2.0], np.float32), np.array([0.0]))
    img = g.to_image()
    assert img.dtype == np.uint8 and img.shape == (100, 100)
    hit = g.world_to_map(2.0, 0.0)
    mid = g.world_to_map(1.0, 0.0)
    assert img[hit[1], hit[0]] == 0
    assert img[mid[1], mid[0]] == 255
    assert img[5, 5] == 200
