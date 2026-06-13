"""PatrolApp: wiring all modules + sim/perception threads."""
import logging
import os
import threading
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
import yaml
from dotenv import load_dotenv

from alerts.telegram import TelegramSender
from apprt.state import SharedState
from locomotion.policy import G1WalkPolicy
from nav.navigator import Navigator
from perception.dedupe import DetectionDeduper
from perception.locator import pixel_to_world
from sim.loader import load_scene
from sim.sensors import depth_to_hsv_bgr, lidar_scan, read_imu
from slam.odometry import NoisyOdometry
from slam.system import SlamSystem

log = logging.getLogger(__name__)


class PatrolApp:
    def __init__(self, cfg_path, enable_detector=True, enable_viewer=False):
        load_dotenv()
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        sc = self.cfg["sim"]
        self.model, self.scene_meta = load_scene(sc)
        self.data = mujoco.MjData(self.model)
        self.policy = G1WalkPolicy(self.model, self.data, self.cfg["policy"])
        self.policy.reset(spawn=sc["spawn_pos"])

        self.state = SharedState()
        lz = self.cfg["lidar"]
        self.slam = SlamSystem(self.cfg["slam"], max_range=lz["max_range"])
        self.slam.set_pose(self.policy.base_pose2d())
        on = self.cfg["slam"]["odom_noise"]
        self.odom = NoisyOdometry(on["v"], on["w"], on["bias_w"])
        self.nav = Navigator(self.cfg["nav"], self.slam)
        self.telegram = TelegramSender(os.getenv("TELEGRAM_BOT_TOKEN", ""),
                                       os.getenv("TELEGRAM_CHAT_ID", ""),
                                       retries=self.cfg["telegram"]["retries"])
        self.deduper = DetectionDeduper(self.cfg["perception"]["dedupe_radius"])
        self.detector = None
        if enable_detector:
            from perception.detector import Detector
            pc = self.cfg["perception"]
            self.detector = Detector(pc["classes"], pc["conf"], pc["weights"])

        self._renderer = None
        self._depth_renderer = None
        self._enable_viewer = enable_viewer
        self._viewer = None
        self._teleop = np.zeros(3)
        self._mode = "manual"
        self._step_i = 0
        self._running = False
        g = self.slam.grid
        self.state.update(telegram=self.telegram.status,
                          map_meta={"res": g.res, "w": g.w, "h": g.h,
                                    "origin": [float(v) for v in g.origin]})

    # ---- controls from web ----
    def set_mode(self, mode):
        self._mode = mode
        if mode == "manual":
            self.nav.stop()
        self.state.update(mode=mode)

    def set_teleop(self, vx, vy, wz):
        t = self.cfg["teleop"]
        self._teleop = np.array([vx * t["vx"], vy * t["vy"], wz * t["wz"]])

    def reset_robot(self):
        self.policy.reset(spawn=self.cfg["sim"]["spawn_pos"])
        self.slam.set_pose(self.policy.base_pose2d())

    # ---- sim tick: 1 physics step + scheduled side-effects ----
    def _sim_tick(self):
        sc = self.cfg["sim"]
        if self._mode == "auto":
            cmd = self.nav.update(self.slam.pose)
        else:
            cmd = self._teleop
        self.policy.set_command(*cmd)
        self.policy.step()
        mujoco.mj_step(self.model, self.data)
        self._step_i += 1
        if self._step_i % sc["render_every"] == 0:
            self._render_frames()
        if self._step_i % sc["lidar_every"] == 0:
            self._lidar_slam_tick()

    def _ensure_renderers(self):
        if self._renderer is None:
            sc = self.cfg["sim"]
            self._renderer = mujoco.Renderer(self.model, sc["img_h"], sc["img_w"])
            self._depth_renderer = mujoco.Renderer(self.model, sc["img_h"], sc["img_w"])
            self._depth_renderer.enable_depth_rendering()

    def _render_frames(self):
        self._ensure_renderers()
        sc, dz = self.cfg["sim"], self.cfg["depth"]
        cam = sc["cam_name"]
        self._renderer.update_scene(self.data, cam)
        rgb = self._renderer.render()
        self._depth_renderer.update_scene(self.data, cam)
        depth = self._depth_renderer.render()
        depth_bgr = depth_to_hsv_bgr(depth, dz["near"], dz["far"])
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        _, dj = cv2.imencode(".jpg", depth_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        bgr = rgb[:, :, ::-1].copy()
        _, rj = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        kw = {"rgb_raw": bgr, "depth_raw": depth,
              "cam_pos": self.data.cam_xpos[cam_id].copy(),
              "cam_mat": self.data.cam_xmat[cam_id].copy(),
              "depth_jpeg": dj.tobytes(), "fallen": self.policy.fallen}
        if self.detector is None:
            kw["rgb_jpeg"] = rj.tobytes()   # with a detector, the perception thread draws bboxes
        else:
            kw.setdefault("rgb_jpeg", self.state.get("rgb_jpeg") or rj.tobytes())
        self.state.update(**kw)

    def _lidar_slam_tick(self):
        lz = self.cfg["lidar"]
        dt = self.model.opt.timestep * self.cfg["sim"]["lidar_every"]
        pose_gt = self.policy.base_pose2d()
        origin = (self.data.xpos[self.scene_meta["lidar_body_id"]]
                  + np.array([0, 0, lz["mount_z_offset"]]))
        ranges, angles = lidar_scan(self.model, self.data, origin, pose_gt[2],
                                    lz["n_rays"], lz["max_range"], lz["min_range"])
        vel = self.policy.base_vel_body()
        delta = self.odom.step(vel[:2], vel[2], dt)
        if self.slam.state == "idle":
            self.slam.set_pose(pose_gt)   # not yet running SLAM -> show GT pose
        self.slam.on_scan(delta, ranges, angles)
        imu = read_imu(self.data, self.policy.base_qadr, self.policy.base_vadr)
        _, mp = cv2.imencode(".png", np.flipud(self.slam.grid.to_image()))
        self.state.update(
            lidar_ranges=ranges.tolist(), lidar_angles=angles.tolist(),
            imu={k: v.tolist() for k, v in imu.items()},
            pose=tuple(float(v) for v in self.slam.pose),
            slam_state=self.slam.state, nav_status=self.nav.status,
            map_png=mp.tobytes(),
            path=[[float(a), float(b)] for a, b in
                  (self.nav.path if self.nav.path is not None else [])],
            goal=self.nav.goal)

    # ---- perception tick (separate thread) ----
    def _perception_tick(self):
        if self.detector is None:
            return
        from perception.detector import draw_detections
        snap = self.state.snapshot()
        bgr, depth = snap["rgb_raw"], snap["depth_raw"]
        if bgr is None or depth is None:
            return
        dets = self.detector.detect(bgr)
        vis = draw_detections(bgr.copy(), dets)
        _, rj = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self.state.update(rgb_jpeg=rj.tobytes())
        # Only raise alerts (world-locate -> dedupe -> Telegram) during AUTO patrol;
        # manual mode is for map-building, so the bbox overlay still shows but no
        # Telegram signal is sent.
        if self._mode != "auto":
            return
        sc = self.cfg["sim"]
        events = snap["detections"]
        for d in dets:
            u = (d.xyxy[0] + d.xyxy[2]) // 2
            v = (d.xyxy[1] + d.xyxy[3]) // 2
            z = float(depth[min(v, depth.shape[0] - 1), min(u, depth.shape[1] - 1)])
            if not (0.1 < z < self.cfg["depth"]["far"]):
                continue
            p = pixel_to_world(u, v, z, snap["cam_pos"], snap["cam_mat"],
                               sc["cam_fovy"], (sc["img_w"], sc["img_h"]))
            if self.deduper.is_new(d.cls, (p[0], p[1])):
                caption = (f"G1 patrol: detected {d.cls} (conf {d.conf:.2f}) "
                           f"at ({p[0]:.1f}, {p[1]:.1f})")
                self.telegram.send(vis, caption)
                events = events + [{"cls": d.cls, "conf": round(d.conf, 2),
                                    "x": round(float(p[0]), 2),
                                    "y": round(float(p[1]), 2),
                                    "t": time.time()}]
        self.state.update(detections=events[-50:])

    # ---- threads ----
    def start(self):
        self._running = True
        # Begin SLAM mapping immediately so the map fills in as soon as the robot
        # drives (no hidden "Start" prerequisite); the SLAM BUILD panel still
        # controls pause/end/save/reset.
        self.slam.start()
        threading.Thread(target=self._sim_loop, daemon=True).start()
        threading.Thread(target=self._perception_loop, daemon=True).start()

    def stop(self):
        self._running = False
        self.telegram.close()
        time.sleep(0.05)  # let the sim thread finish its current tick before freeing GL
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        for r in (self._renderer, self._depth_renderer):
            if r is not None:
                r.close()
        self._renderer = self._depth_renderer = None

    def _sim_loop(self):
        dt = self.model.opt.timestep
        if self._enable_viewer:
            # Create the EGL offscreen renderers FIRST so their context exists
            # before GLFW (the viewer) grabs this thread's context; otherwise the
            # EGL renderer fails to make its context current.
            self._ensure_renderers()
            try:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                log.info("MuJoCo viewer window opened")
            except Exception:
                log.exception("could not open MuJoCo viewer; running headless")
                self._viewer = None
        next_t = time.perf_counter()
        while self._running:
            try:
                self._sim_tick()
                if self._viewer is not None and self._step_i % 16 == 0:
                    if self._viewer.is_running():
                        self._viewer.sync()
                    else:  # user closed the window
                        self._viewer = None
            except Exception:  # a failed tick must not kill the sim thread
                log.exception("sim tick failed")
            next_t += dt
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            elif sleep < -0.5:
                next_t = time.perf_counter()

    def _perception_loop(self):
        period = 1.0 / self.cfg["perception"]["rate_hz"]
        while self._running:
            t0 = time.perf_counter()
            try:
                self._perception_tick()
            except Exception:
                log.exception("perception tick failed")
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
