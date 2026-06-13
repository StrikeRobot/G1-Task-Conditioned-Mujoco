import numpy as np

from nav.astar import astar, inflate


def test_inflate_grows_obstacles():
    occ = np.zeros((20, 20), bool)
    occ[10, 10] = True
    inf = inflate(occ, radius_cells=2)
    assert inf[10, 12] and inf[12, 10] and not inf[10, 14]


def test_astar_straight_line():
    occ = np.zeros((20, 20), bool)
    path = astar(occ, (2, 2), (17, 2))
    assert path is not None and path[0] == (2, 2) and path[-1] == (17, 2)
    assert len(path) <= 17


def test_astar_around_wall():
    occ = np.zeros((20, 20), bool)
    occ[5:20, 10] = True
    path = astar(occ, (5, 2), (5, 18))
    assert path is not None
    assert all(not occ[r, c] for r, c in path)


def test_astar_non_square_grid():
    occ = np.zeros((6, 30), bool)   # 6 rows, 30 cols — catch transpose bug
    occ[0:5, 15] = True             # vertical wall at col 15, gap at row 5
    path = astar(occ, (2, 2), (2, 27))
    assert path is not None
    assert all(not occ[r, c] for r, c in path)
    assert any(r == 5 for r, c in path), "must route through the gap at row 5"


def test_astar_no_path():
    occ = np.zeros((20, 20), bool)
    occ[:, 10] = True
    assert astar(occ, (5, 2), (5, 18)) is None
