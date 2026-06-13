"""G1 29-DOF velocity walking policy (G1_deploy loco/policy_29dof.pt, TorchScript).

Obs 96 (single frame, no history, no gait phase):
  ang_vel(3) | proj_gravity(3) | cmd(3) | (q-default)(29) | dq(29) | last_action(29)
All obs scales are 1.0. Action: q_target = default + 0.25*action (position targets).
PD: override gainprm/biasprm of the position actuators already present in robot.xml.

This checkpoint DOES turn in place (yaw command works at zero forward velocity;
ang_vel_z range ±1.57 rad/s) and stays stable on the full 29-DOF body — unlike the
earlier unitree_rl_lab velocity/v0 policy, which could only arc.

WARNING: __init__ permanently overwrites the model's actuator gain/bias (PD gains)
in-place; any consumer sharing the same MjModel sees the mutated gains.
"""
import mujoco
import numpy as np
import torch

# Policy joint order (Isaac Lab BFS order) — mapped by MJCF name. This is the
# LocoMode.yaml ordering the policy_29dof.pt checkpoint was trained with.
POLICY_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

# Default joint angles in POLICY_JOINTS (BFS) order — LocoMode.yaml.
DEFAULT_POS = np.array([
    -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42, 0.42, 0.35, 0.35,
    -0.23, -0.23, 0.18, -0.18, 0.0, 0.0, 0.0, 0.0, 0.87, 0.87, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# PD gains in POLICY_JOINTS (BFS) order — LocoMode.yaml.
KP = np.array([200, 200, 200, 150, 150, 200, 150, 150, 200, 200, 200, 100, 100,
               20, 20, 100, 100, 20, 20, 50, 50, 50, 50, 40, 40, 40, 40, 40, 40],
              dtype=np.float64)
KD = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
               2, 2, 2, 2, 2, 2, 2], dtype=np.float64)


def get_gravity_orientation(quat):
    qw, qx, qy, qz = quat
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


class G1WalkPolicy:
    def __init__(self, model, data, cfg):
        self.m, self.d = model, data
        self.cfg = cfg
        try:
            self.policy = torch.jit.load(cfg["policy_path"], map_location="cpu").eval()
        except (OSError, FileNotFoundError, RuntimeError) as e:
            raise FileNotFoundError(
                f"policy checkpoint not found or unreadable: {cfg['policy_path']}"
            ) from e
        self.decimation = int(round(cfg["control_dt"] / model.opt.timestep))
        self.action_scale = cfg["action_scale"]
        self.fall_tilt_g = cfg["fall_tilt_g"]
        lim = cfg["cmd_limits"]
        self.cmd_lo = np.array([lim["vx"][0], lim["vy"][0], lim["wz"][0]])
        self.cmd_hi = np.array([lim["vx"][1], lim["vy"][1], lim["wz"][1]])

        # Index entirely by the robot's joint NAMES — the scene has many objects
        # and dynamic objects (freejoints) may be added later: never slice
        # qpos/qvel globally or scan freejoints across the whole model.
        self.qadr = np.array([model.joint(n).qposadr[0] for n in POLICY_JOINTS])
        self.vadr = np.array([model.joint(n).dofadr[0] for n in POLICY_JOINTS])
        base = model.joint("floating_base_joint")
        if model.jnt_type[base.id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("floating_base_joint must be the robot's freejoint")
        self.base_qadr = int(base.qposadr[0])
        self.base_vadr = int(base.dofadr[0])

        # actuator for each policy joint + override PD gains on the model
        jid2idx = {model.joint(n).id: i for i, n in enumerate(POLICY_JOINTS)}
        self.act_for = np.full(29, -1, dtype=int)
        for a in range(model.nu):
            j = model.actuator_trnid[a, 0]
            if j in jid2idx:
                self.act_for[jid2idx[j]] = a
        missing_names = [n for n, idx in zip(POLICY_JOINTS, self.act_for) if idx == -1]
        if missing_names:
            raise ValueError(
                "missing actuator for policy joint: " + ", ".join(missing_names))
        for i in range(29):
            a = self.act_for[i]
            model.actuator_gainprm[a, 0] = KP[i]
            model.actuator_biasprm[a, 1] = -KP[i]
            model.actuator_biasprm[a, 2] = -KD[i]

        self.cmd = np.zeros(3)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.target = DEFAULT_POS.copy()
        self.counter = 0
        self.fallen = False

    def set_command(self, vx, vy, wz):
        self.cmd[:] = np.clip([vx, vy, wz], self.cmd_lo, self.cmd_hi)

    def reset(self, spawn):
        """Reset only the ROBOT's DOFs — do not touch other objects' qpos/qvel."""
        d = self.d
        d.qpos[self.base_qadr : self.base_qadr + 7] = [*spawn, 1.0, 0.0, 0.0, 0.0]
        d.qvel[self.base_vadr : self.base_vadr + 6] = 0
        d.qpos[self.qadr] = DEFAULT_POS
        d.qvel[self.vadr] = 0
        d.ctrl[self.act_for] = DEFAULT_POS
        self.last_action[:] = 0
        self.target = DEFAULT_POS.copy()
        self.counter = 0
        self.fallen = False
        mujoco.mj_forward(self.m, d)

    def _build_obs(self):
        """96-dim single-frame obs (scales all 1.0)."""
        d = self.d
        quat = d.qpos[self.base_qadr + 3 : self.base_qadr + 7]
        obs = np.zeros(96, dtype=np.float32)
        obs[0:3] = d.qvel[self.base_vadr + 3 : self.base_vadr + 6]
        obs[3:6] = get_gravity_orientation(quat)
        obs[6:9] = self.cmd
        obs[9:38] = d.qpos[self.qadr] - DEFAULT_POS
        obs[38:67] = d.qvel[self.vadr]
        obs[67:96] = self.last_action
        return obs

    def step(self):
        """Call once per physics step, BEFORE mj_step."""
        d = self.d
        quat = d.qpos[self.base_qadr + 3 : self.base_qadr + 7]
        if get_gravity_orientation(quat)[2] > self.fall_tilt_g:
            self.fallen = True
        if self.fallen:
            d.ctrl[self.act_for] = d.qpos[self.qadr]  # hold in place, no flailing
            return

        if self.counter % self.decimation == 0:
            obs = self._build_obs()[None, :]
            with torch.no_grad():
                action = self.policy(torch.from_numpy(obs)).numpy().squeeze()
            self.last_action = action.astype(np.float32)
            self.target = DEFAULT_POS + self.action_scale * action
        d.ctrl[self.act_for] = self.target            # position actuators (PD overridden)
        self.counter += 1

    # ---- helpers for SLAM/nav/IMU ----
    def base_pose2d(self):
        q = self.d.qpos
        quat = q[self.base_qadr + 3 : self.base_qadr + 7]
        yaw = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                         1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
        return np.array([q[self.base_qadr], q[self.base_qadr + 1], yaw])

    def base_vel_body(self):
        v = self.d.qvel[self.base_vadr : self.base_vadr + 2]
        wz = self.d.qvel[self.base_vadr + 5]
        yaw = self.base_pose2d()[2]
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], wz])
