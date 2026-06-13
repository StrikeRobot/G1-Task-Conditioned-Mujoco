"""Project pixel + depth -> world (MuJoCo camera: x right, y up, looking along -z)."""
import numpy as np


def pixel_to_world(u, v, depth, cam_pos, cam_mat, fovy_deg, img_wh):
    w, h = img_wh
    fy = (h / 2) / np.tan(np.radians(fovy_deg) / 2)
    fx = fy
    x_cam = (u - w / 2) / fx * depth
    y_cam = -(v - h / 2) / fy * depth
    p_cam = np.array([x_cam, y_cam, -depth])
    return np.asarray(cam_pos) + np.asarray(cam_mat).reshape(3, 3) @ p_cam
