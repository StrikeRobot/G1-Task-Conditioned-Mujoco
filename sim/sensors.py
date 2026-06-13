"""Sensor utilities: IMU, depth colormap, camera intrinsics, lidar raycast."""
import cv2
import mujoco
import numpy as np


def quat_to_rpy(q):
    """MuJoCo quat (w,x,y,z) -> roll, pitch, yaw (rad)."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def read_imu(data, base_qadr, base_vadr):
    quat = data.qpos[base_qadr + 3 : base_qadr + 7]
    rpy = quat_to_rpy(quat)
    gyro = data.qvel[base_vadr + 3 : base_vadr + 6].copy()
    acc_w = data.qacc[base_vadr : base_vadr + 3] + np.array([0, 0, 9.81])
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    acc_b = mat.reshape(3, 3).T @ acc_w
    return {"rpy": rpy, "gyro": gyro, "accel": acc_b}


def depth_to_hsv_bgr(depth, near, far):
    """Depth (m) -> HSV color image: near = bright pink, far = dark purple (per design)."""
    norm = (np.clip(depth, near, far) - near) / (far - near)
    hue = (150 - norm * 40).astype(np.uint8)
    sat = np.full_like(hue, 255)
    val = (255 * (1.0 - 0.65 * norm)).astype(np.uint8)
    return cv2.cvtColor(np.stack([hue, sat, val], axis=-1), cv2.COLOR_HSV2BGR)


def camera_fy(fovy_deg, img_h):
    return (img_h / 2) / np.tan(np.radians(fovy_deg) / 2)


def lidar_scan(model, data, origin, yaw, n_rays, max_range, min_range):
    """2D scan around the z axis from the origin point (world).

    Returns (ranges, angles) in the ROBOT frame (angle 0 = robot nose).
    Only casts against group 0/1 (environment); robot.xml uses group 2 (visual)/3 (collision).
    """
    pnt = np.asarray(origin, dtype=np.float64)
    angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
    ranges = np.full(n_rays, max_range, dtype=np.float32)
    geomgroup = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
    geomid = np.zeros(1, dtype=np.int32)
    for i, a in enumerate(angles):
        th = yaw + a
        vec = np.array([np.cos(th), np.sin(th), 0.0])
        dist = mujoco.mj_ray(model, data, pnt, vec, geomgroup, 1, -1, geomid)
        if min_range < dist < max_range:
            ranges[i] = dist
    return ranges, angles
