"""8-connected A* on a boolean occupancy grid (True = blocked).

Cells use the convention (row, col) = (iy, ix) to stay consistent with NumPy indexing.
The Navigator must swap the order of world_to_map's output before passing it here.
"""
import heapq

import cv2
import numpy as np

SQRT2 = 1.41421356

_NBRS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


def inflate(occ, radius_cells):
    k = 2 * radius_cells + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(occ.astype(np.uint8), kernel).astype(bool)


def _h(a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr + dc) + (SQRT2 - 2) * min(dr, dc)


def astar(occ, start, goal):
    """occ[row, col]; start/goal = (row, col) = (iy, ix). Returns list[(row,col)] or None."""
    h_, w_ = occ.shape
    sr, sc = start
    gr, gc = goal
    if occ[gr, gc] or occ[sr, sc]:
        return None
    g = {start: 0.0}
    came = {}
    pq = [(_h(start, goal), start)]
    seen = set()
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if cur in seen:
            continue
        seen.add(cur)
        for dr, dc, cost in _NBRS:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < h_ and 0 <= nc < w_) or occ[nr, nc]:
                continue
            ng = g[cur] + cost
            if ng < g.get((nr, nc), 1e18):
                g[(nr, nc)] = ng
                came[(nr, nc)] = cur
                heapq.heappush(pq, (ng + _h((nr, nc), goal), (nr, nc)))
    return None
