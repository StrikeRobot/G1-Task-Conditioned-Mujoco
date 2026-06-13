# G1 Patrol & Detection Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Robot Unitree G1 29-DOF (có sẵn trong scene MuJoCo, dùng nguyên không sửa) tuần tra phòng: SLAM build map (manual teleop), auto navigation theo waypoint, YOLO-World phát hiện cốc/cờ lê/tua vít dưới sàn, web dashboard realtime, gửi ảnh detection lên Telegram.

**Architecture:** Một process Python. Thread sim (MuJoCo 500Hz physics + RL policy ONNX 50Hz + render RGB/depth ~30Hz + lidar 10Hz), thread perception (YOLO-World ~8Hz), thread telegram; FastAPI + 1 WebSocket multiplex. **Scene dùng nguyên `scene/scene.xml` + `scene/robot/robot.xml` (G1 29-DOF) — KHÔNG sửa file scene**; camera + thông số PD inject lúc runtime qua `mujoco.MjSpec` và override `actuator_gainprm/biasprm`. Locomotion dùng policy velocity-tracking `policy.onnx` từ **unitree_rl_lab `deploy/robots/g1_29dof`**.

**Tech Stack:** Python 3.13, MuJoCo 3.8.1 (EGL headless, MjSpec), onnxruntime (policy 29-DOF), ultralytics YOLO-World (`yolov8s-worldv2.pt`, CUDA), FastAPI + uvicorn, OpenCV, numpy, vanilla JS/HTML/CSS.

**Spec:** `docs/superpowers/specs/2026-06-12-g1-patrol-demo-design.md`

---

## Facts đã verify (research 2026-06-12, từ repo unitree_rl_lab)

- Policy: `deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx` (~1.66MB) — input `obs` `[1, 480]` float32, output `actions` `[1, 29]`. ONNX chuẩn, load bằng onnxruntime CPU. Cấu hình tại `.../velocity/v0/params/deploy.yaml`.
- **Obs 480 = 6 term × history 5, ghép TERM-MAJOR** (mỗi term xuất đủ 5 bước oldest→newest rồi mới sang term kế):
  1. `base_ang_vel` (3) × scale **0.2**
  2. `projected_gravity` (3) × 1.0
  3. `velocity_commands` (3) = [vx, vy, wz] × 1.0
  4. `joint_pos_rel` (29) = q − default × 1.0
  5. `joint_vel_rel` (29) × **0.05**
  6. `last_action` (29) = output RAW của policy lần trước (chưa scale/offset)
  → 15+15+15+145+145+145 = 480. Scale áp TRƯỚC khi đưa vào history buffer. Khi reset: prefill cả 5 slot bằng obs hiện tại.
- **Joint order của policy (29, KHÁC thứ tự SDK/MJCF):**
  ```
  [ 0] left_hip_pitch    [ 1] right_hip_pitch   [ 2] waist_yaw
  [ 3] left_hip_roll     [ 4] right_hip_roll    [ 5] waist_roll
  [ 6] left_hip_yaw      [ 7] right_hip_yaw     [ 8] waist_pitch
  [ 9] left_knee         [10] right_knee        [11] left_shoulder_pitch
  [12] right_shoulder_pitch [13] left_ankle_pitch [14] right_ankle_pitch
  [15] left_shoulder_roll [16] right_shoulder_roll [17] left_ankle_roll
  [18] right_ankle_roll  [19] left_shoulder_yaw [20] right_shoulder_yaw
  [21] left_elbow        [22] right_elbow       [23] left_wrist_roll
  [24] right_wrist_roll  [25] left_wrist_pitch  [26] right_wrist_pitch
  [27] left_wrist_yaw    [28] right_wrist_yaw
  ```
  Trong code map theo TÊN joint MJCF (`<name>_joint`) — không tin index.
- `default_joint_pos` (policy order):
  `[-0.1, -0.1, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, -0.2, -0.2, 0.25, -0.25, 0, 0, 0, 0, 0.97, 0.97, 0.15, -0.15, 0, 0, 0, 0]`
- **PD gains theo nhóm joint** (an toàn nhất là map theo tên):
  kp: hip 100, knee 150, ankle 40, waist 200, shoulder/elbow/wrist 40.
  kd: hip 2, knee 4, ankle 2, waist 5, shoulder/elbow/wrist 10.
- Action: `q_target = default + 0.25 × action` (position target, PD khép vòng; KHÔNG phải torque). `last_action` đưa vào obs là action RAW.
- Control: policy 50Hz (`step_dt=0.02`); physics scene dt=0.002 → decimation 10 (giữ nguyên dt của scene, không sửa).
- Command ranges: vx ∈ [-0.5, 1.0], vy ∈ [-0.3, 0.3], **wz ∈ [-0.2, 0.2]** (xoay chậm — nav controller phải tôn trọng).
- Fall check phía deploy gốc: nghiêng > 1.0 rad → passive.
- Robot trong scene (`scene/robot/robot.xml`): G1 29-DOF, freejoint `floating_base_joint`, position actuators class g1 (`kp=500 dampratio=1`) → **override gain/bias lúc runtime** thành PD đúng gains policy: với position actuator, `gainprm[0]=kp`, `biasprm[1]=−kp`, `biasprm[2]=−kd`, rồi `ctrl = q_target`. Effort limit đã có sẵn qua `actuatorfrcrange` trên joint (88/139/25/5 Nm).
- Geom groups robot.xml: visual group 2, collision group 3 → lidar raycast mask chỉ group 0/1 là loại được robot.

## Ràng buộc từ user (2026-06-12)

1. **KHÔNG sửa bất kỳ file nào trong `scene/`** — camera, lidar, PD setup đều làm lúc runtime.
2. Dùng nguyên robot 29-DOF trong scene + policy unitree_rl_lab g1_29dof.
3. Scene hiện chỉ có **cốc** (`thangtt___coc`) là vật mục tiêu nằm sàn; không có mesh cờ lê/tua vít. Detector vẫn prompt đủ 3 class — demo phát hiện thực tế chủ yếu là cốc.
4. **Scene có rất nhiều object — cẩn thận khi build input tensor cho policy.** Đã kiểm: hiện scene.xml có 0 joint (object đều static), nhưng code PHẢI index obs hoàn toàn theo TÊN joint của robot (qposadr/dofadr từng joint, freejoint base lookup theo tên `floating_base_joint`), tuyệt đối không slice `qpos[7:]`/`qvel[6:]` toàn cục — để obs không bị ô nhiễm nếu sau này scene thêm object động. Có test riêng chốt điều này (Task 4).

## File Structure

```
dsc_lab_g1/
├── app.py                        # entrypoint: PatrolApp + uvicorn
├── requirements.txt
├── .env.example
├── configs/config.yaml           # toàn bộ tham số runtime
├── scene/                        # CÓ SẴN — KHÔNG SỬA
├── third_party/unitree_rl_lab/   # git clone (gitignored) — chỉ lấy policy.onnx
├── sim/
│   ├── loader.py                 # MjSpec: load scene + inject head_cam + tìm body mount lidar
│   └── sensors.py                # imu, depth→HSV colormap, lidar raycast, camera intrinsics
├── locomotion/policy.py          # G1WalkPolicy: obs 480 (history 5) → ONNX → PD targets
├── slam/{grid,matcher,odometry,system}.py
├── nav/{astar,controller,navigator}.py
├── perception/{detector,locator,dedupe}.py
├── alerts/telegram.py
├── apprt/{state,runtime}.py      # SharedState + PatrolApp threads
├── web/{server.py, static/{index.html,style.css,app.js,map.js}}
├── captures/  maps/              # output (gitignored)
└── tests/
```

---

### Task 1: Scaffolding & dependencies

**Files:**
- Create: `requirements.txt`, `configs/config.yaml`, `.env.example`, `pytest.ini`, các thư mục module + `__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Tạo requirements.txt**

```
# đã có sẵn trong env: mujoco==3.8.1, torch==2.11.0, opencv-python, numpy, fastapi
onnxruntime
ultralytics>=8.3.0
uvicorn[standard]
websockets
python-dotenv
requests
pyyaml
pytest
httpx
```

- [ ] **Step 2: Cài đặt + clone unitree_rl_lab (chỉ để lấy policy)**

```bash
cd /home/cuongtdm/Desktop/dsc_lab_g1
pip install -r requirements.txt
mkdir -p third_party
git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_lab third_party/unitree_rl_lab
ls -la third_party/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx
python -c "import onnxruntime as ort; s=ort.InferenceSession('third_party/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx'); print([(i.name,i.shape) for i in s.get_inputs()], [(o.name,o.shape) for o in s.get_outputs()])"
```

Expected: in ra `[('obs', [1, 480])] [('actions', [1, 29])]` (tên input/output có thể khác nhẹ — ghi lại tên thật, code ở Task 4 đọc tên động từ session nên không sao).

- [ ] **Step 3: Tạo cấu trúc thư mục**

```bash
mkdir -p configs locomotion sim slam nav perception alerts apprt web/static tests captures maps
touch locomotion/__init__.py sim/__init__.py slam/__init__.py nav/__init__.py perception/__init__.py alerts/__init__.py apprt/__init__.py web/__init__.py tests/__init__.py
```

- [ ] **Step 4: Viết configs/config.yaml**

```yaml
sim:
  scene_xml: "scene/scene.xml"      # load qua sim/loader.py (MjSpec) — KHÔNG sửa file
  render_every: 16                  # ~31 Hz RGB+depth (dt 0.002)
  lidar_every: 50                   # 10 Hz lidar + SLAM + nav
  cam_name: "head_cam"              # inject runtime
  cam_body: "torso_link"            # body gắn camera; loader fallback sang pelvis nếu không có
  cam_pos: [0.08, 0.0, 0.45]        # offset so với body (nhìn trước, chúi 35°)
  cam_pitch_down_deg: 35.0
  cam_fovy: 58.0
  img_w: 640
  img_h: 480
  spawn_pos: [-4.7279, -0.2254, 0.79]   # đúng vị trí robot gốc trong scene

policy:
  onnx_path: "third_party/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx"
  control_dt: 0.02                  # 50 Hz -> decimation = control_dt / sim dt
  history_len: 5
  action_scale: 0.25
  ang_vel_scale: 0.2
  joint_vel_scale: 0.05
  # PD theo nhóm tên joint (đúng deploy.yaml unitree_rl_lab)
  kp_map: { hip: 100, knee: 150, ankle: 40, waist: 200, shoulder: 40, elbow: 40, wrist: 40 }
  kd_map: { hip: 2, knee: 4, ankle: 2, waist: 5, shoulder: 10, elbow: 10, wrist: 10 }
  cmd_limits: { vx: [-0.5, 1.0], vy: [-0.3, 0.3], wz: [-0.2, 0.2] }
  fall_tilt_g: -0.5                 # gravity_z body frame > -0.5 (~>60°) => fallen

teleop: { vx: 0.5, vy: 0.25, wz: 0.2 }

lidar:
  n_rays: 360
  max_range: 8.0
  min_range: 0.40                   # bỏ self-hit vào thân robot
  mount_body: "torso_link"          # origin = xpos của body này
  mount_z_offset: 0.1

depth: { near: 0.3, far: 8.0 }

slam:
  size_m: [24.0, 18.0]
  resolution: 0.05
  l_occ: 0.85
  l_free: -0.4
  l_clamp: 4.0
  match_window_xy: 0.15
  match_window_th: 0.08
  odom_noise: { v: 0.03, w: 0.02, bias_w: 0.002 }

nav:
  robot_radius: 0.35
  lookahead: 0.6
  v_max: 0.5
  w_max: 0.2                        # giới hạn policy! xoay chậm
  goal_tol: 0.3
  occ_threshold: 0.7
  patrol_waypoints: []              # điền sau khi build map (Task 18)

perception:
  classes: ["cup", "wrench", "screwdriver"]
  weights: "yolov8s-worldv2.pt"
  conf: 0.25
  rate_hz: 8
  dedupe_radius: 1.0

telegram: { retries: 3 }

web: { host: "0.0.0.0", port: 8000 }
```

- [ ] **Step 5: Tạo .env.example và pytest.ini**

`.env.example`:
```
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
markers =
    render: tests cần GL context (chạy với MUJOCO_GL=egl)
    slow: tests chạy lâu (sim nhiều giây / tải model YOLO)
```

- [ ] **Step 6: Thêm vào .gitignore**

Thêm các dòng: `third_party/`, `*.npz`, `yolov8s-worldv2.pt`

- [ ] **Step 7: Commit**

```bash
git add requirements.txt configs/ .env.example pytest.ini .gitignore */__init__.py
git commit -m "chore: scaffold project structure, config, and dependencies"
```

---

### Task 2: Amend spec theo quyết định mới của user

**Files:**
- Modify: `docs/superpowers/specs/2026-06-12-g1-patrol-demo-design.md`

- [ ] **Step 1: Sửa spec**

Cập nhật các điểm sau trong spec:

1. Mục "1. Simulation": thay đoạn "**Bổ sung:** STL/geom cờ lê và tua vít..." và "**Robot:** thay `scene/robot/robot.xml`..." bằng:
   > **Scene dùng nguyên trạng, không sửa file nào trong `scene/`** (yêu cầu user 2026-06-12). Robot = G1 29-DOF có sẵn trong scene. Camera RGB/depth inject lúc runtime bằng `mujoco.MjSpec` (thêm camera vào body torso trước khi compile model — file XML không đổi). Lidar origin lấy từ `xpos` của body torso, không cần site. Vật mục tiêu trên sàn: các cốc có sẵn trong scene; không có mesh cờ lê/tua vít — detector vẫn prompt 3 class nhưng demo phát hiện thực tế là cốc.
2. Mục "2. Locomotion": thay bằng:
   > Policy velocity-tracking `policy.onnx` từ **unitree_rl_lab `deploy/robots/g1_29dof`** (velocity/v0), chạy bằng onnxruntime trong process. Obs 480-dim (6 term × history 5, term-major), action 29 = position targets (`default + 0.25×action`), PD gains theo deploy.yaml (hip 100/knee 150/ankle 40/waist 200/arm 40) — override lên position actuators có sẵn của robot.xml lúc runtime (`actuator_gainprm/biasprm`). 50Hz control, physics 500Hz giữ nguyên. Giới hạn lệnh: vx [-0.5,1.0], vy [-0.3,0.3], wz [-0.2,0.2].
3. Mục "4. Navigation": thêm câu "Tốc độ xoay tối đa 0.2 rad/s theo giới hạn policy — pure pursuit dùng w_max=0.2."
4. Mục "2. Locomotion": thêm câu "Scene chứa nhiều object nên obs tensor build hoàn toàn theo tên joint robot (qposadr/dofadr per-joint, base = `floating_base_joint`), không slice qpos/qvel toàn cục — chống ô nhiễm obs nếu scene thêm object động (yêu cầu user 2026-06-12)."

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-12-g1-patrol-demo-design.md
git commit -m "docs: spec v2 - keep scene untouched, use in-scene 29dof robot + rl_lab onnx policy"
```

---

### Task 3: Scene loader — MjSpec inject camera (không sửa file)

**Files:**
- Create: `sim/loader.py`, `tests/test_loader.py`

- [ ] **Step 1: Viết test fail**

`tests/test_loader.py`:
```python
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import yaml

from sim.loader import load_scene

CFG = yaml.safe_load(open("configs/config.yaml"))


def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def test_loader_does_not_touch_scene_files():
    before = {p: _md5(p) for p in ("scene/scene.xml", "scene/robot/robot.xml")}
    load_scene(CFG["sim"])
    after = {p: _md5(p) for p in before}
    assert before == after, "loader không được sửa file scene"


def test_loader_injects_camera_and_keeps_robot():
    model, meta = load_scene(CFG["sim"])
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam") >= 0
    # robot: freejoint base theo TÊN + đủ 29 joint (KHÔNG đếm toàn cục —
    # scene có thể thêm object động sau này)
    base = model.joint("floating_base_joint")
    assert model.jnt_type[base.id] == mujoco.mjtJoint.mjJNT_FREE
    # kiểm đại diện các nhóm joint (đủ 29 joint check kỹ ở Task 4 theo POLICY_JOINTS)
    for n in ("left_knee_joint", "waist_yaw_joint", "right_wrist_yaw_joint",
              "left_ankle_roll_joint", "right_shoulder_pitch_joint"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0, f"thiếu {n}"
    assert model.nu >= 29, f"phải có >=29 actuator, có {model.nu}"
    # room còn nguyên
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_north") >= 0
    # meta trả về body mount lidar hợp lệ
    assert meta["lidar_body_id"] >= 0


def test_loaded_model_steps():
    model, _ = load_scene(CFG["sim"])
    d = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, d)
    assert np.all(np.isfinite(d.qpos))
```

- [ ] **Step 2: Chạy test, verify fail**

Run: `pytest tests/test_loader.py -v` — Expected: FAIL (module chưa có).

- [ ] **Step 3: Viết sim/loader.py**

```python
"""Load scene.xml qua MjSpec và inject camera head_cam lúc runtime.

KHÔNG ghi/sửa bất kỳ file nào trong scene/ — mọi thay đổi chỉ tồn tại
trong model đã compile trong RAM.
"""
import mujoco
import numpy as np


def _camera_quat(pitch_down_deg):
    """Quat cho camera gắn trên body (x trước, z lên): nhìn về +x, chúi xuống.

    Frame camera MuJoCo: nhìn dọc -z, x phải, y lên.
    """
    p = np.radians(pitch_down_deg)
    view = np.array([np.cos(p), 0.0, -np.sin(p)])   # hướng nhìn trong body frame
    zc = -view                                       # trục z camera = -hướng nhìn
    xc = np.array([0.0, -1.0, 0.0])                  # x camera = bên phải robot
    yc = np.cross(zc, xc)
    R = np.column_stack([xc, yc, zc])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return quat


def load_scene(sim_cfg):
    """Trả (model, meta). meta = {'lidar_body_id', 'cam_body_name'}."""
    spec = mujoco.MjSpec.from_file(sim_cfg["scene_xml"])

    # tìm body gắn camera: ưu tiên cam_body, fallback pelvis
    body = None
    for name in (sim_cfg["cam_body"], "pelvis"):
        body = spec.body(name)
        if body is not None:
            break
    assert body is not None, "không tìm thấy body gắn camera (torso_link/pelvis)"

    cam = body.add_camera()
    cam.name = sim_cfg["cam_name"]
    cam.pos = sim_cfg["cam_pos"]
    cam.fovy = sim_cfg["cam_fovy"]
    cam.quat = _camera_quat(sim_cfg["cam_pitch_down_deg"])

    model = spec.compile()
    meta = {
        "cam_body_name": body.name,
        "lidar_body_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.name),
    }
    return model, meta
```

**Lưu ý API MjSpec (mujoco 3.8):** `spec.body(name)` trả None nếu không có; `body.add_camera()` trả mjsCamera với thuộc tính `name/pos/quat/fovy`. Nếu API khác nhẹ (ví dụ `spec.find_body`), sửa theo `help(mujoco.MjSpec)` — đây là điểm duy nhất phụ thuộc API mới.

**Nếu `MjSpec.from_file` lỗi với include/meshdir:** fallback đã định sẵn — đọc XML bằng ElementTree TRONG RAM, thêm node camera vào body torso, rồi `MjModel.from_xml_string(ET.tostring(...), assets=...)`. Vẫn không ghi file nào.

- [ ] **Step 4: Chạy test, verify pass**

Run: `pytest tests/test_loader.py -v` — Expected: 3 PASS.

Nếu `n_hinge != 29`: in danh sách joint (`[model.joint(i).name for i in range(model.njnt)]`) để xác nhận robot.xml là bản 29dof; điều chỉnh assert nếu robot có thêm hand joints (khi đó vẫn OK — policy chỉ điều khiển 29 joint theo tên, các joint thừa giữ nguyên).

- [ ] **Step 5: Commit**

```bash
git add sim/loader.py tests/test_loader.py
git commit -m "feat: runtime scene loader injecting head camera via MjSpec (scene untouched)"
```

---

### Task 4: Locomotion — G1WalkPolicy (ONNX 29-DOF, obs 480)

Tái hiện đúng deploy của unitree_rl_lab: obs term-major history 5, action = position target, PD qua position actuators (override gains runtime).

**Files:**
- Create: `locomotion/policy.py`, `tests/test_policy.py`

- [ ] **Step 1: Viết test fail**

`tests/test_policy.py`:
```python
import mujoco
import numpy as np
import pytest
import yaml

from locomotion.policy import G1WalkPolicy, POLICY_JOINTS, get_gravity_orientation
from sim.loader import load_scene

CFG = yaml.safe_load(open("configs/config.yaml"))


def test_gravity_orientation_upright():
    g = get_gravity_orientation(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(g, [0, 0, -1], atol=1e-6)


def test_policy_joint_list_has_29():
    assert len(POLICY_JOINTS) == 29


def _make():
    model, _ = load_scene(CFG["sim"])
    d = mujoco.MjData(model)
    pol = G1WalkPolicy(model, d, CFG["policy"])
    pol.reset(spawn=CFG["sim"]["spawn_pos"])
    return model, d, pol


def test_obs_dim_is_480():
    m, d, pol = _make()
    obs = pol._build_obs()
    assert obs.shape == (480,)


def test_obs_indexed_by_robot_joint_names():
    """Scene nhiều object: obs phải lấy đúng DOF robot theo tên, không slice toàn cục."""
    m, d, pol = _make()
    # left_knee là policy index 9 -> trong term qpos (29 phần tử), bước newest
    knee_adr = m.joint("left_knee_joint").qposadr[0]
    d.qpos[knee_adr] += 0.123
    mujoco.mj_forward(m, d)
    terms = pol._current_terms()
    assert abs(terms["qpos"][9] - 0.123) < 1e-6, \
        "qpos term index 9 phải phản ánh đúng left_knee theo tên joint"
    # base address phải đúng freejoint của robot theo tên
    assert pol.base_qadr == m.joint("floating_base_joint").qposadr[0]


@pytest.mark.slow
def test_stands_still_with_zero_command():
    m, d, pol = _make()
    pol.set_command(0.0, 0.0, 0.0)
    for _ in range(int(3.0 / m.opt.timestep)):
        pol.step()
        mujoco.mj_step(m, d)
    assert not pol.fallen
    assert d.qpos[pol.base_qadr + 2] > 0.5, "robot phải đứng vững với lệnh 0"


@pytest.mark.slow
def test_walks_forward():
    m, d, pol = _make()
    pol.set_command(0.5, 0.0, 0.0)
    x0 = d.qpos[pol.base_qadr]
    for _ in range(int(5.0 / m.opt.timestep)):
        pol.step()
        mujoco.mj_step(m, d)
    assert not pol.fallen, "robot ngã khi đi thẳng"
    assert d.qpos[pol.base_qadr] - x0 > 1.0, "phải đi >1m trong 5s"


@pytest.mark.slow
def test_turns():
    m, d, pol = _make()
    pol.set_command(0.0, 0.0, 0.2)
    for _ in range(int(4.0 / m.opt.timestep)):
        pol.step()
        mujoco.mj_step(m, d)
    quat = d.qpos[pol.base_qadr + 3 : pol.base_qadr + 7]
    yaw = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                     1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
    assert not pol.fallen
    assert yaw > 0.3, f"phải xoay >0.3rad trong 4s với wz=0.2, yaw={yaw:.2f}"
```

- [ ] **Step 2: Chạy test, verify fail**

Run: `pytest tests/test_policy.py -v -m ""` — Expected: FAIL (module chưa có).

- [ ] **Step 3: Viết locomotion/policy.py**

```python
"""G1 29-DOF walking policy (unitree_rl_lab velocity/v0, ONNX).

Obs 480 = 6 term x history 5, term-major, oldest->newest:
  ang_vel*0.2 | proj_gravity | cmd | (q-default) | dq*0.05 | last_action_raw
Action: q_target = default + 0.25*action (position targets, PD onboard).
PD: override gainprm/biasprm của position actuators sẵn có trong robot.xml.
"""
from collections import deque

import mujoco
import numpy as np
import onnxruntime as ort

# Thứ tự joint của policy (Isaac Lab BFS order) — map theo tên MJCF
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

DEFAULT_POS = np.array([
    -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3,
    -0.2, -0.2, 0.25, -0.25, 0.0, 0.0, 0.0, 0.0, 0.97, 0.97, 0.15, -0.15,
    0.0, 0.0, 0.0, 0.0])

TERMS = ("ang_vel", "gravity", "cmd", "qpos", "qvel", "action")


def get_gravity_orientation(quat):
    qw, qx, qy, qz = quat
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def _gain_for(name, gain_map):
    for key, val in gain_map.items():
        if key in name:
            return float(val)
    raise KeyError(f"không có gain cho joint {name}")


class G1WalkPolicy:
    def __init__(self, model, data, cfg):
        self.m, self.d = model, data
        self.cfg = cfg
        self.sess = ort.InferenceSession(cfg["onnx_path"],
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.decimation = int(round(cfg["control_dt"] / model.opt.timestep))
        self.hist_len = cfg["history_len"]
        self.action_scale = cfg["action_scale"]
        self.ang_vel_scale = cfg["ang_vel_scale"]
        self.joint_vel_scale = cfg["joint_vel_scale"]
        self.fall_tilt_g = cfg["fall_tilt_g"]
        lim = cfg["cmd_limits"]
        self.cmd_lo = np.array([lim["vx"][0], lim["vy"][0], lim["wz"][0]])
        self.cmd_hi = np.array([lim["vx"][1], lim["vy"][1], lim["wz"][1]])

        # Index hoàn toàn theo TÊN joint của robot — scene có nhiều object,
        # có thể thêm object động (freejoint) sau này: tuyệt đối không slice
        # qpos/qvel toàn cục hay quét freejoint toàn model.
        self.qadr = np.array([model.joint(n).qposadr[0] for n in POLICY_JOINTS])
        self.vadr = np.array([model.joint(n).dofadr[0] for n in POLICY_JOINTS])
        base = model.joint("floating_base_joint")
        assert model.jnt_type[base.id] == mujoco.mjtJoint.mjJNT_FREE, \
            "floating_base_joint phải là freejoint của robot"
        self.base_qadr = int(base.qposadr[0])
        self.base_vadr = int(base.dofadr[0])

        # actuator cho từng policy joint + override PD gains lên model
        jid2idx = {model.joint(n).id: i for i, n in enumerate(POLICY_JOINTS)}
        self.act_for = np.full(29, -1, dtype=int)
        for a in range(model.nu):
            j = model.actuator_trnid[a, 0]
            if j in jid2idx:
                self.act_for[jid2idx[j]] = a
        assert np.all(self.act_for >= 0), "thiếu actuator cho policy joint"
        for i, n in enumerate(POLICY_JOINTS):
            kp = _gain_for(n, cfg["kp_map"])
            kd = _gain_for(n, cfg["kd_map"])
            a = self.act_for[i]
            model.actuator_gainprm[a, 0] = kp
            model.actuator_biasprm[a, 1] = -kp
            model.actuator_biasprm[a, 2] = -kd

        self.cmd = np.zeros(3)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.target = DEFAULT_POS.copy()
        self.counter = 0
        self.fallen = False
        self.hist = {t: deque(maxlen=self.hist_len) for t in TERMS}

    def set_command(self, vx, vy, wz):
        self.cmd[:] = np.clip([vx, vy, wz], self.cmd_lo, self.cmd_hi)

    def reset(self, spawn):
        """Chỉ reset DOF của ROBOT — không đụng qpos/qvel của object khác."""
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
        terms = self._current_terms()
        for t in TERMS:
            self.hist[t].clear()
            for _ in range(self.hist_len):           # prefill như deploy gốc
                self.hist[t].append(terms[t].copy())

    def _current_terms(self):
        d = self.d
        quat = d.qpos[self.base_qadr + 3 : self.base_qadr + 7]
        return {
            "ang_vel": (d.qvel[self.base_vadr + 3 : self.base_vadr + 6]
                        * self.ang_vel_scale).astype(np.float32),
            "gravity": get_gravity_orientation(quat).astype(np.float32),
            "cmd": self.cmd.astype(np.float32),
            "qpos": (d.qpos[self.qadr] - DEFAULT_POS).astype(np.float32),
            "qvel": (d.qvel[self.vadr] * self.joint_vel_scale).astype(np.float32),
            "action": self.last_action.copy(),
        }

    def _build_obs(self):
        """Term-major: mỗi term đủ history (oldest->newest) rồi mới sang term kế."""
        return np.concatenate(
            [np.concatenate(list(self.hist[t])) for t in TERMS]).astype(np.float32)

    def step(self):
        """Gọi 1 lần mỗi physics step, TRƯỚC mj_step."""
        d = self.d
        quat = d.qpos[self.base_qadr + 3 : self.base_qadr + 7]
        if get_gravity_orientation(quat)[2] > self.fall_tilt_g:
            self.fallen = True
        if self.fallen:
            d.ctrl[self.act_for] = d.qpos[self.qadr]  # giữ tại chỗ, không vùng vẫy
            return

        if self.counter % self.decimation == 0:
            terms = self._current_terms()
            for t in TERMS:
                self.hist[t].append(terms[t])
            obs = self._build_obs()[None, :]
            action = self.sess.run([self.out_name], {self.in_name: obs})[0][0]
            self.last_action = action.astype(np.float32)
            self.target = DEFAULT_POS + self.action_scale * action
        d.ctrl[self.act_for] = self.target            # position actuators (PD đã override)
        self.counter += 1

    # ---- helpers cho SLAM/nav/IMU ----
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
```

- [ ] **Step 4: Chạy test, verify pass**

Run: `MUJOCO_GL=egl pytest tests/test_policy.py -v -m ""`
Expected: 6 PASS.

**Debug guide nếu robot ngã (đọc kỹ — đây là điểm rủi ro lớn nhất, sim2sim Isaac→MuJoCo):**
1. Verify joint names: `python -c "...; print([m.joint(i).name for i in range(m.njnt)])"` — nếu tên waist khác (vd `waist_yaw_joint` không tồn tại), sửa POLICY_JOINTS theo tên thật (giữ đúng thứ tự policy).
2. Verify obs từng term bằng cách in 96 phần tử bước cuối (newest) — gravity phải ≈ [0,0,-1] khi đứng.
3. Thử history layout: nếu vẫn ngã sau khi obs đúng, thử layout step-major (5 block × 96) — đổi `_build_obs` thành `np.concatenate([np.concatenate([self.hist[t][k] for t in TERMS]) for k in range(self.hist_len)])`. (Term-major là layout đã verify từ source C++, nhưng đây là fallback rẻ.)
4. Kiểm tra PD: in `model.actuator_gainprm[a]` trước/sau override; thử warmup 1s giữ `ctrl = DEFAULT_POS` trước khi bật policy.
5. Scene khác biệt: sàn friction 0.9, timestep 0.002, impratio 100 — nếu ngã do solver, thử decimation theo control_dt (giữ nguyên scene, KHÔNG sửa file).

- [ ] **Step 5: Commit**

```bash
git add locomotion/policy.py tests/test_policy.py
git commit -m "feat: 29-dof ONNX walking policy (unitree_rl_lab velocity/v0) with runtime PD override"
```

---

### Task 5: Sensors — IMU, depth colormap, intrinsics, lidar

**Files:**
- Create: `sim/sensors.py`, `tests/test_sensors.py`

- [ ] **Step 1: Viết test fail**

`tests/test_sensors.py`:
```python
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
    assert not np.array_equal(img[0, 0], img[0, 9]), "gần và xa phải khác màu"


def test_camera_fy():
    assert abs(camera_fy(58.0, 480) - (240 / np.tan(np.radians(29)))) < 1e-6
```

- [ ] **Step 2: Chạy test, verify fail**

Run: `pytest tests/test_sensors.py -v` — Expected: FAIL.

- [ ] **Step 3: Viết sim/sensors.py**

```python
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
    """Depth (m) -> ảnh màu HSV: gần = hồng sáng, xa = tím tối (theo design)."""
    norm = (np.clip(depth, near, far) - near) / (far - near)
    hue = (150 - norm * 40).astype(np.uint8)
    sat = np.full_like(hue, 255)
    val = (255 * (1.0 - 0.65 * norm)).astype(np.uint8)
    return cv2.cvtColor(np.stack([hue, sat, val], axis=-1), cv2.COLOR_HSV2BGR)


def camera_fy(fovy_deg, img_h):
    return (img_h / 2) / np.tan(np.radians(fovy_deg) / 2)


def lidar_scan(model, data, origin, yaw, n_rays, max_range, min_range):
    """Quét 2D quanh trục z từ điểm origin (world).

    Trả (ranges, angles) trong ROBOT frame (angle 0 = mũi robot).
    Chỉ bắn group 0/1 (môi trường); robot.xml dùng group 2 (visual)/3 (collision).
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
```

- [ ] **Step 4: Chạy test, verify pass**

Run: `pytest tests/test_sensors.py -v` — Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add sim/sensors.py tests/test_sensors.py
git commit -m "feat: sensor utils (imu, depth HSV colormap, intrinsics, lidar)"
```

---

### Task 6: Lidar verification

**Files:**
- Create: `tests/test_lidar.py`

- [ ] **Step 1: Viết test (scene synthetic)**

`tests/test_lidar.py`:
```python
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
    assert np.all(ranges > 0.35), "không được self-hit"


def test_lidar_in_real_scene():
    from sim.loader import load_scene
    cfg = yaml.safe_load(open("configs/config.yaml"))
    m, meta = load_scene(cfg["sim"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    origin = d.xpos[meta["lidar_body_id"]] + [0, 0, cfg["lidar"]["mount_z_offset"]]
    r, a = lidar_scan(m, d, origin, 0.0, 360, 8.0, cfg["lidar"]["min_range"])
    assert np.sum(r < 8.0) > 50, f"lidar phải thấy môi trường, hits={np.sum(r < 8.0)}"
    assert np.all(r > cfg["lidar"]["min_range"]), "self-hit vào robot"
```

- [ ] **Step 2: Chạy test, verify pass**

Run: `pytest tests/test_lidar.py -v` — Expected: 2 PASS.

**Nếu self-hit trong scene thật:** kiểm geom group robot: `grep -o 'group="[0-9]"' scene/robot/robot.xml | sort | uniq -c` — class visual=2, collision=3 như đã verify. Nếu vẫn hit (geom không class), tăng `min_range` lên 0.5 trong config (KHÔNG sửa scene).

- [ ] **Step 3: Commit**

```bash
git add tests/test_lidar.py
git commit -m "test: lidar raycast in synthetic and real scene"
```

---

### Task 7: OccupancyGrid

**Files:**
- Create: `slam/grid.py`, `tests/test_grid.py`

- [ ] **Step 1: Viết test fail**

`tests/test_grid.py`:
```python
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
```

- [ ] **Step 2: Chạy test, verify fail**

Run: `pytest tests/test_grid.py -v` — Expected: FAIL.

- [ ] **Step 3: Viết slam/grid.py**

```python
"""Occupancy grid log-odds, update vectorized (không bresenham per-cell)."""
import numpy as np


class OccupancyGrid:
    def __init__(self, size_m, resolution, l_occ, l_free, l_clamp, max_range):
        self.res = resolution
        self.w = int(round(size_m[0] / resolution))
        self.h = int(round(size_m[1] / resolution))
        self.origin = np.array([-size_m[0] / 2.0, -size_m[1] / 2.0])
        self.l_occ, self.l_free, self.l_clamp = l_occ, l_free, l_clamp
        self.max_range = max_range
        self.logodds = np.zeros((self.h, self.w), dtype=np.float32)

    def world_to_map(self, x, y):
        ix = int((x - self.origin[0]) / self.res)
        iy = int((y - self.origin[1]) / self.res)
        return ix, iy

    def map_to_world(self, ix, iy):
        return (self.origin[0] + (ix + 0.5) * self.res,
                self.origin[1] + (iy + 0.5) * self.res)

    def update(self, pose, ranges, angles):
        """pose=(x,y,yaw) world; ranges/angles trong robot frame."""
        x, y, th = pose
        ranges = np.asarray(ranges, dtype=np.float32)
        dirs = np.stack([np.cos(th + angles), np.sin(th + angles)], axis=1)
        step = self.res * 0.8
        n_steps = int(self.max_range / step)
        ts = (np.arange(1, n_steps + 1) * step)[None, :]
        hit_mask = ranges < self.max_range - 1e-3
        free_len = np.where(hit_mask, ranges - self.res, ranges)[:, None]
        valid = ts < free_len
        px = x + dirs[:, 0:1] * ts
        py = y + dirs[:, 1:2] * ts
        self._add(np.stack([px[valid], py[valid]], -1), self.l_free)
        ex = x + dirs[hit_mask, 0] * ranges[hit_mask]
        ey = y + dirs[hit_mask, 1] * ranges[hit_mask]
        self._add(np.stack([ex, ey], -1), self.l_occ)
        np.clip(self.logodds, -self.l_clamp, self.l_clamp, out=self.logodds)

    def _add(self, pts_xy, val):
        if len(pts_xy) == 0:
            return
        ij = ((pts_xy - self.origin) / self.res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.w) & (ij[:, 1] >= 0) & (ij[:, 1] < self.h)
        ij = ij[ok]
        flat = np.unique(ij[:, 1] * self.w + ij[:, 0])  # mỗi cell 1 lần/scan
        self.logodds.flat[flat] += val

    def occupied_mask(self, thr=0.7):
        return self.logodds > thr

    def to_image(self):
        """uint8 HxW: free=255, occupied=0, unknown=200. Hàng 0 = y nhỏ nhất."""
        img = np.full((self.h, self.w), 200, np.uint8)
        img[self.logodds < -0.3] = 255
        img[self.logodds > 0.7] = 0
        return img

    def save(self, path):
        np.savez_compressed(path, logodds=self.logodds,
                            origin=self.origin, res=self.res)

    def load(self, path):
        z = np.load(path)
        self.logodds = z["logodds"].astype(np.float32)
        self.h, self.w = self.logodds.shape
        self.origin = z["origin"]
        self.res = float(z["res"])
```

- [ ] **Step 4: Chạy test, verify pass**

Run: `pytest tests/test_grid.py -v` — Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add slam/grid.py tests/test_grid.py
git commit -m "feat: log-odds occupancy grid with vectorized update and save/load"
```

---

### Task 8: Correlative scan matcher

**Files:**
- Create: `slam/matcher.py`, `tests/test_matcher.py`

- [ ] **Step 1: Viết test fail**

`tests/test_matcher.py`:
```python
import numpy as np

from slam.grid import OccupancyGrid
from slam.matcher import ScanMatcher


def _room_scan(pose, n=180):
    """Scan synthetic phòng vuông 8x8 (tường tại ±4) nhìn từ pose."""
    x, y, th = pose
    angles = np.linspace(-np.pi, np.pi, n, endpoint=False)
    ranges = np.zeros(n, dtype=np.float32)
    for i, a in enumerate(angles):
        dx, dy = np.cos(th + a), np.sin(th + a)
        ts = []
        if dx > 1e-9: ts.append((4 - x) / dx)
        if dx < -1e-9: ts.append((-4 - x) / dx)
        if dy > 1e-9: ts.append((4 - y) / dy)
        if dy < -1e-9: ts.append((-4 - y) / dy)
        ranges[i] = min(t for t in ts if t > 0)
    return ranges, angles


def test_matcher_recovers_offset():
    g = OccupancyGrid((12, 12), 0.05, 0.85, -0.4, 4.0, max_range=10.0)
    true_pose = np.array([0.5, -0.3, 0.2])
    r, a = _room_scan(true_pose)
    for _ in range(8):
        g.update(true_pose, r, a)
    m = ScanMatcher(g, window_xy=0.15, window_th=0.08)
    guess = true_pose + np.array([0.10, -0.08, 0.05])
    est, score = m.match(r, a, guess)
    assert np.linalg.norm(est[:2] - true_pose[:2]) < 0.04, f"xy lệch: {est}"
    assert abs(est[2] - true_pose[2]) < 0.03, f"theta lệch: {est}"
```

- [ ] **Step 2: Chạy test, verify fail**

Run: `pytest tests/test_matcher.py -v` — Expected: FAIL.

- [ ] **Step 3: Viết slam/matcher.py**

```python
"""Correlative scan matching trên occupancy grid (coarse-to-fine grid search)."""
import cv2
import numpy as np


class ScanMatcher:
    def __init__(self, grid, window_xy=0.15, window_th=0.08):
        self.grid = grid
        self.wxy = window_xy
        self.wth = window_th

    def match(self, ranges, angles, guess):
        """Trả (pose_est, score). guess = (x,y,yaw) dự đoán từ odometry."""
        g = self.grid
        occ = (g.logodds > 0.7).astype(np.float32)
        field = cv2.GaussianBlur(occ, (7, 7), 1.5)
        hit = ranges < g.max_range - 1e-3
        r, a = ranges[hit], angles[hit]
        if len(r) < 20:
            return np.asarray(guess, dtype=float), 0.0

        def score(pose):
            x, y, th = pose
            ex = x + r * np.cos(th + a)
            ey = y + r * np.sin(th + a)
            ix = ((ex - g.origin[0]) / g.res).astype(int)
            iy = ((ey - g.origin[1]) / g.res).astype(int)
            ok = (ix >= 0) & (ix < g.w) & (iy >= 0) & (iy < g.h)
            if not np.any(ok):
                return 0.0
            return float(field[iy[ok], ix[ok]].sum()) / len(r)

        best = np.asarray(guess, dtype=float)
        best_s = score(best)
        # 2 vòng coarse-to-fine, lưới 5x5x5 quanh best
        for sx, sth in ((self.wxy / 2, self.wth / 2), (self.wxy / 8, self.wth / 8)):
            center = best.copy()
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        c = center + np.array([i * sx / 2, j * sx / 2, k * sth / 2])
                        s = score(c)
                        if s > best_s:
                            best_s, best = s, c
        return best, best_s
```

- [ ] **Step 4: Chạy test, verify pass**

Run: `pytest tests/test_matcher.py -v` — Expected: PASS (125 cand × 2 vòng, <50ms).

- [ ] **Step 5: Commit**

```bash
git add slam/matcher.py tests/test_matcher.py
git commit -m "feat: correlative scan matcher (coarse-to-fine grid search)"
```

---

### Task 9: Odometry nhiễu + SlamSystem

**Files:**
- Create: `slam/odometry.py`, `slam/system.py`, `tests/test_slam_system.py`

- [ ] **Step 1: Viết test fail**

`tests/test_slam_system.py`:
```python
import numpy as np

from slam.odometry import NoisyOdometry
from slam.system import SlamSystem
from tests.test_matcher import _room_scan

CFG = {
    "size_m": [12.0, 12.0], "resolution": 0.05, "l_occ": 0.85, "l_free": -0.4,
    "l_clamp": 4.0, "match_window_xy": 0.15, "match_window_th": 0.08,
    "odom_noise": {"v": 0.03, "w": 0.02, "bias_w": 0.002},
}


def test_odometry_drifts_but_reasonable():
    odo = NoisyOdometry(noise_v=0.05, noise_w=0.03, bias_w=0.005, seed=1)
    pose = np.zeros(3)
    for _ in range(100):
        d = odo.step(np.array([1.0, 0.0]), 0.0, 0.1)
        c, s = np.cos(pose[2]), np.sin(pose[2])
        pose[0] += c * d[0] - s * d[1]
        pose[1] += s * d[0] + c * d[1]
        pose[2] += d[2]
    err = abs(pose[0] - 10.0) + abs(pose[1])
    assert 0.001 < err < 3.0, f"phải có drift nhưng không vô lý: {pose}"


def test_slam_states():
    s = SlamSystem(CFG, max_range=8.0)
    assert s.state == "idle"
    s.start(); assert s.state == "building"
    s.pause(); assert s.state == "paused"
    s.start(); assert s.state == "building"
    s.end_build(); assert s.state == "localized"
    s.reset(); assert s.state == "idle"


def test_slam_builds_map_and_tracks(tmp_path):
    s = SlamSystem(CFG, max_range=8.0)
    s.start()
    true = np.array([0.0, 0.0, 0.0])
    s.set_pose(true)
    for _ in range(30):
        true = true + np.array([0.05, 0, 0])
        r, a = _room_scan(true)
        s.on_scan(odom_delta=np.array([0.05, 0, 0]), ranges=r, angles=a)
    assert np.sum(s.grid.occupied_mask()) > 100, "map phải có tường"
    assert np.linalg.norm(s.pose[:2] - true[:2]) < 0.3, f"track sai: {s.pose} vs {true}"
    p = tmp_path / "room.npz"
    s.save(p)
    s2 = SlamSystem(CFG, max_range=8.0)
    s2.load(p)
    assert s2.state == "localized"
    assert np.sum(s2.grid.occupied_mask()) > 100
```

- [ ] **Step 2: Chạy test, verify fail**

Run: `pytest tests/test_slam_system.py -v` — Expected: FAIL.

- [ ] **Step 3: Viết slam/odometry.py**

```python
"""Odometry nhiễu: nhận vận tốc thật từ sim, trả delta pose (body frame) noise+bias."""
import numpy as np


class NoisyOdometry:
    def __init__(self, noise_v=0.03, noise_w=0.02, bias_w=0.002, seed=0):
        self.rng = np.random.default_rng(seed)
        self.noise_v, self.noise_w = noise_v, noise_w
        self.bias_w = self.rng.normal(0, bias_w)

    def step(self, v_body_xy, wz, dt):
        """Trả (dx, dy, dyaw) body frame, đã nhiễu."""
        v = np.asarray(v_body_xy) * (1 + self.rng.normal(0, self.noise_v, 2))
        w = wz + self.rng.normal(0, self.noise_w) + self.bias_w
        return np.array([v[0] * dt, v[1] * dt, w * dt])
```

- [ ] **Step 4: Viết slam/system.py**

```python
"""SLAM orchestrator: odometry predict -> scan match correct -> grid update."""
import numpy as np

from slam.grid import OccupancyGrid
from slam.matcher import ScanMatcher


class SlamSystem:
    """States: idle -> building <-> paused -> localized (end_build/load)."""

    def __init__(self, cfg, max_range):
        self.cfg = cfg
        self.max_range = max_range
        self.state = "idle"
        self.pose = np.zeros(3)
        self._scan_count = 0
        self._new_grid()

    def _new_grid(self):
        c = self.cfg
        self.grid = OccupancyGrid(tuple(c["size_m"]), c["resolution"], c["l_occ"],
                                  c["l_free"], c["l_clamp"], self.max_range)
        self.matcher = ScanMatcher(self.grid, c["match_window_xy"],
                                   c["match_window_th"])

    # ---- controls ----
    def start(self):
        if self.state in ("idle", "paused"):
            self.state = "building"

    def pause(self):
        if self.state == "building":
            self.state = "paused"

    def end_build(self):
        if self.state in ("building", "paused"):
            self.state = "localized"

    def reset(self):
        self.state = "idle"
        self.pose = np.zeros(3)
        self._scan_count = 0
        self._new_grid()

    def set_pose(self, pose):
        self.pose = np.asarray(pose, dtype=float).copy()

    def save(self, path):
        self.grid.save(path)

    def load(self, path):
        self.grid.load(path)
        self.matcher = ScanMatcher(self.grid, self.cfg["match_window_xy"],
                                   self.cfg["match_window_th"])
        self.state = "localized"
        self._scan_count = 50

    # ---- main update 10Hz ----
    def on_scan(self, odom_delta, ranges, angles):
        if self.state in ("idle", "paused"):
            return
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        self.pose = self.pose + np.array([
            c * odom_delta[0] - s * odom_delta[1],
            s * odom_delta[0] + c * odom_delta[1],
            odom_delta[2]])
        if self._scan_count >= 5:
            self.pose, _ = self.matcher.match(ranges, angles, self.pose)
        if self.state == "building":
            self.grid.update(self.pose, ranges, angles)
            self._scan_count += 1
```

- [ ] **Step 5: Chạy test, verify pass**

Run: `pytest tests/test_slam_system.py -v` — Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add slam/odometry.py slam/system.py tests/test_slam_system.py
git commit -m "feat: noisy odometry and SLAM system FSM with save/load"
```

---

### Task 10: A* + inflate

**Files:**
- Create: `nav/astar.py`, `tests/test_astar.py`

- [ ] **Step 1: Viết test fail**

`tests/test_astar.py`:
```python
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
    assert all(not occ[iy, ix] for ix, iy in path)


def test_astar_no_path():
    occ = np.zeros((20, 20), bool)
    occ[:, 10] = True
    assert astar(occ, (5, 2), (5, 18)) is None
```

- [ ] **Step 2: Chạy test, verify fail**

Run: `pytest tests/test_astar.py -v` — Expected: FAIL.

- [ ] **Step 3: Viết nav/astar.py**

```python
"""A* 8-hướng trên occupancy bool grid (True = blocked). Cells là (ix, iy)."""
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
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (SQRT2 - 2) * min(dx, dy)


def astar(occ, start, goal):
    """occ[iy, ix]; start/goal = (ix, iy). Trả list[(ix,iy)] hoặc None."""
    h_, w_ = occ.shape
    if occ[goal[1], goal[0]] or occ[start[1], start[0]]:
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
        for dx, dy, cost in _NBRS:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < w_ and 0 <= ny < h_) or occ[ny, nx]:
                continue
            ng = g[cur] + cost
            if ng < g.get((nx, ny), 1e18):
                g[(nx, ny)] = ng
                came[(nx, ny)] = cur
                heapq.heappush(pq, (ng + _h((nx, ny), goal), (nx, ny)))
    return None
```

- [ ] **Step 4: Chạy test, verify pass**

Run: `pytest tests/test_astar.py -v` — Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add nav/astar.py tests/test_astar.py
git commit -m "feat: A* path planning with obstacle inflation"
```

---

### Task 11: PurePursuit + Navigator (w_max = 0.2 theo giới hạn policy)

**Files:**
- Create: `nav/controller.py`, `nav/navigator.py`, `tests/test_nav.py`

- [ ] **Step 1: Viết test fail**

`tests/test_nav.py`:
```python
import numpy as np

from nav.controller import PurePursuit
from nav.navigator import Navigator
from slam.grid import OccupancyGrid


def _pp():
    return PurePursuit(lookahead=0.6, v_max=0.5, w_max=0.2, goal_tol=0.3)


def test_pp_drives_forward_on_straight_path():
    path = np.array([[0.5, 0], [1.0, 0], [2.0, 0], [3.0, 0]])
    vx, wz, done = _pp().compute(np.array([0, 0, 0.0]), path)
    assert vx > 0.25 and abs(wz) < 0.1 and not done


def test_pp_arcs_when_facing_wrong_way():
    """Policy KHÔNG xoay tại chỗ được (Task 4 finding) -> phải đi cung: vx>=v_turn."""
    path = np.array([[1.0, 0], [2.0, 0]])
    vx, wz, done = _pp().compute(np.array([0, 0, np.pi / 2]), path)
    assert abs(vx - 0.3) < 1e-6 and wz <= -0.15 and not done  # cung phải về +x


def test_pp_done_at_goal():
    path = np.array([[1.0, 0]])
    vx, wz, done = _pp().compute(np.array([0.9, 0.05, 0.0]), path)
    assert done and vx == 0 and wz == 0


class _FakeSlam:
    def __init__(self):
        self.grid = OccupancyGrid((10, 10), 0.1, 0.85, -0.4, 4.0, max_range=8.0)
        self.grid.logodds[:] = -1.0
        self.pose = np.zeros(3)


NAV_CFG = {"robot_radius": 0.2, "lookahead": 0.6, "v_max": 0.5, "w_max": 0.2,
           "goal_tol": 0.3, "occ_threshold": 0.7, "patrol_waypoints": []}


def test_navigator_full_cycle():
    slam = _FakeSlam()
    nav = Navigator(NAV_CFG, slam)
    assert nav.status.startswith("IDLE")
    ok = nav.set_goal(2.0, 0.0)
    assert ok and nav.status.startswith("NAVIGATING")
    vx, vy, wz = nav.update(np.array([0.0, 0.0, 0.0]))
    assert vx > 0
    nav.update(np.array([1.95, 0.0, 0.0]))
    assert nav.status.startswith("DONE")
    nav.stop()
    assert nav.status.startswith("IDLE")


def test_navigator_rejects_goal_in_obstacle():
    slam = _FakeSlam()
    ix, iy = slam.grid.world_to_map(2.0, 2.0)
    slam.grid.logodds[iy - 3:iy + 4, ix - 3:ix + 4] = 4.0
    nav = Navigator(NAV_CFG, slam)
    assert not nav.set_goal(2.0, 2.0)


def test_navigator_patrol_advances_waypoints():
    slam = _FakeSlam()
    cfg = dict(NAV_CFG, patrol_waypoints=[[1.0, 0.0], [2.0, 0.0]])
    nav = Navigator(cfg, slam)
    assert nav.start_patrol()
    assert "wp 1/2" in nav.status
    nav.update(np.array([0.95, 0.0, 0.0]))   # tới wp1 -> sang wp2
    assert "wp 2/2" in nav.status
    nav.update(np.array([1.95, 0.0, 0.0]))
    assert nav.status.startswith("DONE")
```

- [ ] **Step 2: Chạy test, verify fail**

Run: `pytest tests/test_nav.py -v` — Expected: FAIL.

- [ ] **Step 3: Viết nav/controller.py**

```python
"""Pure pursuit đơn giản hoá: heading control tới điểm lookahead.

QUAN TRỌNG (finding Task 4): policy 29dof KHÔNG xoay tại chỗ được (gait tắt
khi vx~0). Khi lệch hướng lớn phải đi CUNG: giữ vx = v_turn (~0.3) + wz bão
hoà. Bán kính quay tối thiểu ~ v_turn/w_max = 1.5m — nav phải chừa khoảng.
"""
import numpy as np

V_TURN = 0.3  # vx tối thiểu để gait sống khi cần quay gắt


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class PurePursuit:
    def __init__(self, lookahead, v_max, w_max, goal_tol):
        self.lookahead = lookahead
        self.v_max, self.w_max = v_max, w_max
        self.goal_tol = goal_tol

    def compute(self, pose, path):
        """pose=(x,y,yaw); path=Nx2 world. Trả (vx, wz, done)."""
        p = np.asarray(pose[:2])
        if np.linalg.norm(np.asarray(path[-1]) - p) < self.goal_tol:
            return 0.0, 0.0, True
        target = path[-1]
        for pt in path:
            if np.linalg.norm(np.asarray(pt) - p) > self.lookahead:
                target = pt
                break
        err = _wrap(np.arctan2(target[1] - p[1], target[0] - p[0]) - pose[2])
        wz = float(np.clip(2.0 * err, -self.w_max, self.w_max))
        if abs(err) > 0.8:
            return V_TURN, wz, False  # lệch nhiều -> đi cung (không xoay tại chỗ được)
        vx = max(V_TURN, self.v_max * (1.0 - abs(err) / 0.8))
        return float(vx), wz, False
```

- [ ] **Step 4: Viết nav/navigator.py**

```python
"""Navigator FSM: goal world -> A* trên grid inflate -> bám path."""
import numpy as np

from nav.astar import astar, inflate
from nav.controller import PurePursuit


class Navigator:
    def __init__(self, cfg, slam):
        self.cfg = cfg
        self.slam = slam
        self.pp = PurePursuit(cfg["lookahead"], cfg["v_max"], cfg["w_max"],
                              cfg["goal_tol"])
        self.path = None
        self.goal = None
        self.waypoints = list(cfg.get("patrol_waypoints") or [])
        self.wp_index = 0
        self._patrolling = False
        self.status = "IDLE"

    def _wp_suffix(self):
        if self._patrolling and self.waypoints:
            return f" (wp {self.wp_index + 1}/{len(self.waypoints)})"
        return ""

    def set_goal(self, x, y):
        g = self.slam.grid
        occ = inflate(g.logodds > self.cfg["occ_threshold"],
                      max(1, int(self.cfg["robot_radius"] / g.res)))
        start = g.world_to_map(*self.slam.pose[:2])
        goal = g.world_to_map(x, y)
        occ[start[1], start[0]] = False
        cells = astar(occ, start, goal)
        if cells is None:
            self.status = "NO PATH"
            return False
        pts = [g.map_to_world(ix, iy) for ix, iy in cells[1:]] or [(x, y)]
        self.path = np.array(pts)
        self.goal = (x, y)
        self.status = "NAVIGATING" + self._wp_suffix()
        return True

    def start_patrol(self):
        if not self.waypoints:
            return False
        self._patrolling = True
        self.wp_index = 0
        return self.set_goal(*self.waypoints[0])

    def stop(self):
        self.path = None
        self.goal = None
        self.wp_index = 0
        self._patrolling = False
        self.status = "IDLE"

    def update(self, pose):
        """Gọi 10Hz ở auto mode. Trả (vx, vy, wz)."""
        if self.path is None:
            return 0.0, 0.0, 0.0
        vx, wz, done = self.pp.compute(pose, self.path)
        if done:
            if self._patrolling and self.wp_index + 1 < len(self.waypoints):
                self.wp_index += 1
                self.set_goal(*self.waypoints[self.wp_index])
                return 0.0, 0.0, 0.0
            self.path = None
            self._patrolling = False
            self.status = "DONE"
            return 0.0, 0.0, 0.0
        return vx, 0.0, wz
```

- [ ] **Step 5: Chạy test, verify pass**

Run: `pytest tests/test_nav.py -v` — Expected: 6 PASS.

- [ ] **Step 6: Commit**

```bash
git add nav/controller.py nav/navigator.py tests/test_nav.py
git commit -m "feat: pure pursuit (w_max 0.2) and navigator FSM with patrol waypoints"
```

---

### Task 12: YOLO-World detector + bbox overlay

**Files:**
- Create: `perception/detector.py`, `tests/test_detector.py`

- [ ] **Step 1: Viết test fail**

`tests/test_detector.py`:
```python
import numpy as np
import pytest

from perception.detector import Detection, draw_detections


def test_draw_detections_marks_pixels():
    img = np.zeros((100, 100, 3), np.uint8)
    dets = [Detection(cls="cup", conf=0.9, xyxy=(10, 10, 50, 50))]
    out = draw_detections(img.copy(), dets)
    assert out.sum() > 0


@pytest.mark.slow
def test_yolo_world_loads_and_runs():
    from perception.detector import Detector
    det = Detector(classes=["cup", "wrench", "screwdriver"], conf=0.25)
    img = np.full((480, 640, 3), 128, np.uint8)
    out = det.detect(img)
    assert isinstance(out, list)
```

- [ ] **Step 2: Chạy test, verify fail**

Run: `pytest tests/test_detector.py -v -m "not slow"` — Expected: FAIL.

- [ ] **Step 3: Viết perception/detector.py**

```python
"""YOLO-World open-vocabulary detector."""
from dataclasses import dataclass

import cv2


@dataclass
class Detection:
    cls: str
    conf: float
    xyxy: tuple  # (x1, y1, x2, y2) pixel


COLORS = {"cup": (60, 200, 60), "wrench": (60, 120, 255),
          "screwdriver": (220, 80, 220)}


class Detector:
    def __init__(self, classes, conf, weights="yolov8s-worldv2.pt", device=None):
        from ultralytics import YOLOWorld
        self.model = YOLOWorld(weights)
        self.model.set_classes(list(classes))
        self.classes = list(classes)
        self.conf = conf
        self.device = device

    def detect(self, bgr):
        res = self.model.predict(bgr, conf=self.conf, verbose=False,
                                 device=self.device)[0]
        out = []
        for b in res.boxes:
            out.append(Detection(
                cls=self.classes[int(b.cls.item())],
                conf=float(b.conf.item()),
                xyxy=tuple(int(v) for v in b.xyxy[0].tolist())))
        return out


def draw_detections(bgr, dets):
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        color = COLORS.get(d.cls, (0, 255, 255))
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(bgr, f"{d.cls} {d.conf:.2f}", (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return bgr
```

- [ ] **Step 4: Chạy test, verify pass**

Run: `pytest tests/test_detector.py -v -m ""` (lần đầu tải `yolov8s-worldv2.pt` + CLIP text encoder — cần internet)
Expected: 2 PASS.

- [ ] **Step 5: Sanity check trên scene thật — QUAN TRỌNG**

```bash
MUJOCO_GL=egl python - <<'EOF'
import mujoco, cv2, yaml
from sim.loader import load_scene
from perception.detector import Detector, draw_detections
cfg = yaml.safe_load(open("configs/config.yaml"))
m, meta = load_scene(cfg["sim"])
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 480, 640)
r.update_scene(d, "head_cam")
img = r.render()[:, :, ::-1].copy()
cv2.imwrite("/tmp/cam_view.jpg", img)
det = Detector(cfg["perception"]["classes"], 0.05)  # conf thấp để khảo sát
out = det.detect(img)
print([(x.cls, round(x.conf, 2)) for x in out])
cv2.imwrite("/tmp/det_check.jpg", draw_detections(img, out))
EOF
```

Xem `/tmp/cam_view.jpg` (camera nhìn đúng hướng/độ chúi chưa?) và `/tmp/det_check.jpg`. Ghi lại confidence thực tế trên cốc trong scene → chỉnh `perception.conf` (rủi ro STL không texture: hạ xuống 0.1–0.2 nếu cần). Có thể cần lái robot tới gần cốc để kiểm tra — làm lại check này ở Task 17 khi có teleop.

- [ ] **Step 6: Commit**

```bash
git add perception/detector.py tests/test_detector.py
git commit -m "feat: YOLO-World detector with bbox overlay"
```

---

### Task 13: Locator (pixel→world) + DetectionDeduper

**Files:**
- Create: `perception/locator.py`, `perception/dedupe.py`, `tests/test_locator_dedupe.py`

- [ ] **Step 1: Viết test fail**

`tests/test_locator_dedupe.py`:
```python
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
```

- [ ] **Step 2: Chạy test, verify fail**

Run: `pytest tests/test_locator_dedupe.py -v` — Expected: FAIL.

- [ ] **Step 3: Viết perception/locator.py và perception/dedupe.py**

`perception/locator.py`:
```python
"""Chiếu pixel + depth -> world (camera MuJoCo: x phải, y lên, nhìn dọc -z)."""
import numpy as np


def pixel_to_world(u, v, depth, cam_pos, cam_mat, fovy_deg, img_wh):
    w, h = img_wh
    fy = (h / 2) / np.tan(np.radians(fovy_deg) / 2)
    fx = fy
    x_cam = (u - w / 2) / fx * depth
    y_cam = -(v - h / 2) / fy * depth
    p_cam = np.array([x_cam, y_cam, -depth])
    return np.asarray(cam_pos) + np.asarray(cam_mat).reshape(3, 3) @ p_cam
```

`perception/dedupe.py`:
```python
"""Dedupe detection theo không gian: mỗi (class, vùng bán kính R) báo 1 lần."""
import numpy as np


class DetectionDeduper:
    def __init__(self, radius=1.0):
        self.radius = radius
        self.seen = []  # list[(cls, x, y)]

    def is_new(self, cls, xy):
        for c, x, y in self.seen:
            if c == cls and np.hypot(xy[0] - x, xy[1] - y) < self.radius:
                return False
        self.seen.append((cls, xy[0], xy[1]))
        return True

    def reset(self):
        self.seen.clear()
```

- [ ] **Step 4: Chạy test, verify pass**

Run: `pytest tests/test_locator_dedupe.py -v` — Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add perception/locator.py perception/dedupe.py tests/test_locator_dedupe.py
git commit -m "feat: pixel-to-world projection and spatial detection dedupe"
```

---

### Task 14: Telegram sender (ARMED/MOCK)

**Files:**
- Create: `alerts/telegram.py`, `tests/test_telegram.py`

- [ ] **Step 1: Viết test fail**

`tests/test_telegram.py`:
```python
import numpy as np

import alerts.telegram as tg
from alerts.telegram import TelegramSender


def _img():
    return np.zeros((10, 10, 3), np.uint8)


def test_mock_mode_saves_locally(tmp_path):
    s = TelegramSender(token="", chat_id="", out_dir=str(tmp_path))
    assert s.status == "MOCK"
    s.send(_img(), "cup 0.91 at (1.0, 2.0)")
    s.close()
    assert len(list(tmp_path.glob("*.jpg"))) == 1


def test_armed_mode_posts(tmp_path, monkeypatch):
    calls = []

    def fake_post(url, data=None, files=None, timeout=None):
        calls.append((url, data))
        class R: status_code = 200
        return R()

    monkeypatch.setattr(tg.requests, "post", fake_post)
    s = TelegramSender(token="TOK", chat_id="123", out_dir=str(tmp_path))
    assert s.status == "ARMED"
    s.send(_img(), "wrench 0.8 at (0, 0)")
    s.close()
    assert len(calls) == 1
    assert "botTOK/sendPhoto" in calls[0][0]
    assert calls[0][1]["chat_id"] == "123"


def test_armed_retries_on_failure(tmp_path, monkeypatch):
    n = {"count": 0}

    def fail_post(url, data=None, files=None, timeout=None):
        n["count"] += 1
        raise ConnectionError("boom")

    monkeypatch.setattr(tg.requests, "post", fail_post)
    s = TelegramSender(token="TOK", chat_id="1", out_dir=str(tmp_path), retries=3)
    s.send(_img(), "x")
    s.close()
    assert n["count"] == 3
    assert len(list(tmp_path.glob("*.jpg"))) == 1  # vẫn lưu local
```

- [ ] **Step 2: Chạy test, verify fail**

Run: `pytest tests/test_telegram.py -v` — Expected: FAIL.

- [ ] **Step 3: Viết alerts/telegram.py**

```python
"""Gửi ảnh detection lên Telegram (thread + queue, không block sim).

Tạo bot: nhắn @BotFather -> /newbot -> lấy TOKEN.
Lấy chat_id: nhắn bot 1 tin bất kỳ rồi mở
https://api.telegram.org/bot<TOKEN>/getUpdates -> message.chat.id.
Điền vào .env (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID). Thiếu -> MOCK mode.
"""
import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import requests

log = logging.getLogger(__name__)


class TelegramSender:
    def __init__(self, token, chat_id, out_dir="captures", retries=3):
        self.token, self.chat_id, self.retries = token, chat_id, retries
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.q = queue.Queue()
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    @property
    def status(self):
        return "ARMED" if (self.token and self.chat_id) else "MOCK"

    def send(self, image_bgr, caption):
        self.q.put((image_bgr.copy(), caption))

    def close(self):
        self.q.put(None)
        self._t.join(timeout=10)

    def _worker(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            img, caption = item
            fname = self.out_dir / f"det_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(fname), img)
            if self.status == "MOCK":
                log.info("MOCK telegram: %s -> %s", caption, fname)
                continue
            ok, buf = cv2.imencode(".jpg", img)
            url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
            for i in range(self.retries):
                try:
                    r = requests.post(url, data={"chat_id": self.chat_id,
                                                 "caption": caption},
                                      files={"photo": ("det.jpg", buf.tobytes())},
                                      timeout=10)
                    if r.status_code == 200:
                        break
                    log.warning("telegram HTTP %s (try %d)", r.status_code, i + 1)
                except Exception as e:
                    log.warning("telegram error: %s (try %d)", e, i + 1)
                time.sleep(1)
```

- [ ] **Step 4: Chạy test, verify pass**

Run: `pytest tests/test_telegram.py -v` — Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add alerts/telegram.py tests/test_telegram.py
git commit -m "feat: telegram photo alert sender with mock fallback and retries"
```

---

### Task 15: SharedState + PatrolApp runtime

**Files:**
- Create: `apprt/state.py`, `apprt/runtime.py`, `app.py`, `tests/test_runtime.py`

- [ ] **Step 1: Viết test fail**

`tests/test_runtime.py`:
```python
import numpy as np
import pytest

from apprt.runtime import PatrolApp


@pytest.fixture(scope="module")
def app():
    return PatrolApp("configs/config.yaml", enable_detector=False)


@pytest.mark.slow
@pytest.mark.render
def test_sim_ticks_produce_state(app):
    app.set_mode("manual")
    app.set_teleop(1.0, 0.0, 0.0)   # full tiến
    app.slam.start()
    for _ in range(1500):           # 3s sim
        app._sim_tick()
    snap = app.state.snapshot()
    assert snap["rgb_jpeg"] is not None
    assert snap["depth_jpeg"] is not None
    assert snap["map_png"] is not None
    assert len(snap["lidar_ranges"]) == 360
    assert snap["imu"] is not None
    assert not snap["fallen"]
    assert np.sum(app.slam.grid.occupied_mask()) > 50, "map phải có tường"


@pytest.mark.slow
@pytest.mark.render
def test_mode_switch_and_nav(app):
    app.slam.end_build()
    app.set_mode("auto")
    x, y, _ = app.slam.pose
    assert app.nav.set_goal(x + 0.5, y)
    vx, vy, wz = app.nav.update(app.slam.pose)
    assert vx > 0 or abs(wz) > 0
```

- [ ] **Step 2: Chạy test, verify fail**

Run: `MUJOCO_GL=egl pytest tests/test_runtime.py -v -m ""` — Expected: FAIL.

- [ ] **Step 3: Viết apprt/state.py**

```python
"""Snapshot state chia sẻ giữa sim/perception/web — qua 1 lock."""
import threading


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._d = {
            "rgb_jpeg": None, "depth_jpeg": None, "map_png": None,
            "rgb_raw": None, "depth_raw": None, "cam_pos": None, "cam_mat": None,
            "lidar_ranges": [], "lidar_angles": [], "imu": None,
            "pose": (0.0, 0.0, 0.0), "mode": "manual", "fallen": False,
            "slam_state": "idle", "nav_status": "IDLE", "telegram": "MOCK",
            "detections": [], "path": [], "goal": None, "map_meta": None,
        }

    def update(self, **kw):
        with self._lock:
            self._d.update(kw)

    def snapshot(self):
        with self._lock:
            return dict(self._d)

    def get(self, key):
        with self._lock:
            return self._d[key]
```

- [ ] **Step 4: Viết apprt/runtime.py**

```python
"""PatrolApp: wiring mọi module + sim/perception threads."""
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
    def __init__(self, cfg_path, enable_detector=True):
        load_dotenv()
        self.cfg = yaml.safe_load(open(cfg_path))
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
        self._teleop = np.zeros(3)
        self._mode = "manual"
        self._step_i = 0
        self._running = False
        g = self.slam.grid
        self.state.update(telegram=self.telegram.status,
                          map_meta={"res": g.res, "w": g.w, "h": g.h,
                                    "origin": [float(v) for v in g.origin]})

    # ---- controls từ web ----
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

    # ---- sim tick: 1 physics step + side-effects theo lịch ----
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
            kw["rgb_jpeg"] = rj.tobytes()   # có detector thì perception thread vẽ bbox
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
            self.slam.set_pose(pose_gt)   # chưa SLAM -> hiển thị pose GT
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

    # ---- perception tick (thread riêng) ----
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
                caption = (f"G1 patrol: phát hiện {d.cls} (conf {d.conf:.2f}) "
                           f"tại ({p[0]:.1f}, {p[1]:.1f})")
                self.telegram.send(vis, caption)
                events = events + [{"cls": d.cls, "conf": round(d.conf, 2),
                                    "x": round(float(p[0]), 2),
                                    "y": round(float(p[1]), 2),
                                    "t": time.time()}]
        self.state.update(detections=events[-50:])

    # ---- threads ----
    def start(self):
        self._running = True
        threading.Thread(target=self._sim_loop, daemon=True).start()
        threading.Thread(target=self._perception_loop, daemon=True).start()

    def stop(self):
        self._running = False
        self.telegram.close()

    def _sim_loop(self):
        dt = self.model.opt.timestep
        next_t = time.perf_counter()
        while self._running:
            self._sim_tick()
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
```

- [ ] **Step 5: Viết app.py**

```python
"""Entrypoint: MUJOCO_GL=egl python app.py [--config configs/config.yaml]"""
import argparse
import logging

import uvicorn

from apprt.runtime import PatrolApp
from web.server import create_app


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--no-detector", action="store_true")
    args = ap.parse_args()

    patrol = PatrolApp(args.config, enable_detector=not args.no_detector)
    patrol.start()
    web_cfg = patrol.cfg["web"]
    try:
        uvicorn.run(create_app(patrol), host=web_cfg["host"], port=web_cfg["port"])
    finally:
        patrol.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Chạy test, verify pass**

Run: `MUJOCO_GL=egl pytest tests/test_runtime.py -v -m ""`
Expected: 2 PASS.

- [ ] **Step 7: Commit**

```bash
git add apprt/ app.py tests/test_runtime.py
git commit -m "feat: PatrolApp runtime wiring sim, slam, nav, perception, telegram"
```

---

### Task 16: Web backend (FastAPI + WS)

Server nhận object `patrol` duck-typed → test bằng fake, không cần MuJoCo.

**Files:**
- Create: `web/server.py`, `tests/test_server.py`

- [ ] **Step 1: Viết test fail**

`tests/test_server.py`:
```python
from fastapi.testclient import TestClient

from web.server import create_app


class _FakeSlam:
    state = "idle"
    def start(self): self.state = "building"
    def pause(self): self.state = "paused"
    def end_build(self): self.state = "localized"
    def reset(self): self.state = "idle"
    def save(self, p): self.saved = str(p)
    def load(self, p): self.loaded = str(p)


class _FakeNav:
    status = "IDLE"
    def set_goal(self, x, y): self.goal = (x, y); return True
    def stop(self): self.status = "IDLE"
    def start_patrol(self): return True


class _FakeState:
    @staticmethod
    def snapshot(): return {"mode": "manual", "pose": (0, 0, 0)}


class _FakePatrol:
    def __init__(self):
        self.slam, self.nav, self.state = _FakeSlam(), _FakeNav(), _FakeState()
    def set_mode(self, m): self.mode = m
    def set_teleop(self, vx, vy, wz): self.teleop = (vx, vy, wz)
    def reset_robot(self): self.reset_called = True


def _client():
    fake = _FakePatrol()
    return TestClient(create_app(fake)), fake


def test_mode_endpoint():
    c, fake = _client()
    assert c.post("/api/mode", json={"mode": "auto"}).status_code == 200
    assert fake.mode == "auto"


def test_teleop_endpoint():
    c, fake = _client()
    assert c.post("/api/teleop", json={"vx": 1, "vy": 0, "wz": -1}).status_code == 200
    assert fake.teleop == (1, 0, -1)


def test_slam_endpoints():
    c, fake = _client()
    assert c.post("/api/slam/start").status_code == 200
    assert fake.slam.state == "building"
    assert c.post("/api/slam/save", json={"name": "map1"}).status_code == 200
    assert fake.slam.saved.endswith("map1.npz")


def test_nav_goal():
    c, fake = _client()
    assert c.post("/api/nav/goal", json={"x": 1.5, "y": -2.0}).status_code == 200
    assert fake.nav.goal == (1.5, -2.0)
```

- [ ] **Step 2: Chạy test, verify fail**

Run: `pytest tests/test_server.py -v` — Expected: FAIL.

- [ ] **Step 3: Viết web/server.py**

```python
"""FastAPI: REST controls + 1 WebSocket multiplex.

WS /ws/stream:
  - binary: 1 byte kênh + payload (0x01 rgb jpeg, 0x02 depth jpeg, 0x03 map png)
  - text: JSON telemetry
"""
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

MAPS_DIR = Path("maps")
STATIC_DIR = Path(__file__).parent / "static"


def create_app(patrol):
    app = FastAPI()

    @app.post("/api/mode")
    async def set_mode(body: dict):
        patrol.set_mode(body["mode"])
        return {"ok": True}

    @app.post("/api/teleop")
    async def teleop(body: dict):
        patrol.set_teleop(body.get("vx", 0), body.get("vy", 0), body.get("wz", 0))
        return {"ok": True}

    @app.post("/api/robot/reset")
    async def robot_reset():
        patrol.reset_robot()
        return {"ok": True}

    @app.post("/api/nav/goal")
    async def nav_goal(body: dict):
        return {"ok": patrol.nav.set_goal(float(body["x"]), float(body["y"]))}

    @app.post("/api/nav/stop")
    async def nav_stop():
        patrol.nav.stop()
        return {"ok": True}

    @app.post("/api/nav/patrol")
    async def nav_patrol():
        return {"ok": patrol.nav.start_patrol()}

    @app.get("/api/slam/maps")
    async def list_maps():
        MAPS_DIR.mkdir(exist_ok=True)
        return {"maps": sorted(p.stem for p in MAPS_DIR.glob("*.npz"))}

    @app.post("/api/slam/{action}")
    async def slam_action(action: str, body: dict | None = None):
        MAPS_DIR.mkdir(exist_ok=True)
        if action == "start":
            patrol.slam.start()
        elif action == "pause":
            patrol.slam.pause()
        elif action == "end":
            patrol.slam.end_build()
        elif action == "reset":
            patrol.slam.reset()
        elif action == "save":
            patrol.slam.save(MAPS_DIR / f"{(body or {}).get('name', 'map')}.npz")
        elif action == "load":
            patrol.slam.load(MAPS_DIR / f"{(body or {}).get('name', 'map')}.npz")
        else:
            return {"ok": False, "error": "unknown action"}
        return {"ok": True}

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        i = 0
        try:
            while True:
                snap = patrol.state.snapshot()
                if snap.get("rgb_jpeg"):
                    await ws.send_bytes(b"\x01" + snap["rgb_jpeg"])
                if snap.get("depth_jpeg"):
                    await ws.send_bytes(b"\x02" + snap["depth_jpeg"])
                if i % 8 == 0 and snap.get("map_png"):     # ~2Hz
                    await ws.send_bytes(b"\x03" + snap["map_png"])
                telemetry = {k: snap.get(k) for k in (
                    "pose", "mode", "slam_state", "nav_status", "detections",
                    "path", "goal", "telegram", "fallen", "map_meta", "imu",
                    "lidar_ranges", "lidar_angles")}
                await ws.send_text(json.dumps(telemetry))
                i += 1
                await asyncio.sleep(1 / 15)
        except WebSocketDisconnect:
            pass

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
```

Lưu ý: route `GET /api/slam/maps` phải khai báo TRƯỚC `POST /api/slam/{action}` như trên (khác method nên không xung đột, nhưng giữ thứ tự này cho rõ).

- [ ] **Step 4: Chạy test, verify pass**

Run: `pytest tests/test_server.py -v` — Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add web/server.py tests/test_server.py
git commit -m "feat: FastAPI server with REST controls and multiplexed WS stream"
```

---

### Task 17: Frontend dashboard

Layout theo ảnh design `photo_2026-06-12_21-10-42.jpg`: header status dots; hàng panel CAMERA/DEPTH/LIDAR/IMU; SLAM MAP lớn + panel phải DRIVE MODE/NAVIGATION/SLAM BUILD; nền sáng, card trắng, accent đỏ/đen.

**Files:**
- Create: `web/static/index.html`, `web/static/style.css`, `web/static/map.js`, `web/static/app.js`

- [ ] **Step 1: Viết index.html**

```html
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8"/>
<title>G1 PATROL</title>
<link rel="stylesheet" href="style.css"/>
</head>
<body>
<header>
  <div class="logo">⚡ G1·PATROL</div>
  <div class="dots">
    <span id="dot-cam" class="dot"></span>Camera
    <span id="dot-depth" class="dot"></span>Depth
    <span id="dot-lidar" class="dot"></span>LIDAR
    <span id="dot-imu" class="dot"></span>IMU
  </div>
</header>

<main>
  <section class="row-top">
    <div class="card"><h3>CAMERA <small>bbox overlay</small></h3>
      <img id="cam" alt="camera"/></div>
    <div class="card"><h3>DEPTH <small>colormap</small></h3>
      <img id="depth" alt="depth"/></div>
    <div class="card"><h3>LIDAR <small>top-down</small></h3>
      <canvas id="lidar" width="300" height="300"></canvas></div>
    <div class="card imu-card"><h3>IMU <small>orientation</small></h3>
      <div class="rpy">
        <div><label>ROLL</label><b id="roll">0.0</b></div>
        <div><label>PITCH</label><b id="pitch">0.0</b></div>
        <div><label>YAW</label><b id="yaw">0.0</b></div>
      </div>
      <table class="imu-table">
        <tr><th>Accel</th><td id="ax">0</td><td id="ay">0</td><td id="az">0</td></tr>
        <tr><th>Gyro</th><td id="gx">0</td><td id="gy">0</td><td id="gz">0</td></tr>
      </table></div>
  </section>

  <section class="row-main">
    <div class="card map-card">
      <h3>SLAM MAP <small>drag: pan · wheel: zoom · double-click: fit ·
        right-click (Auto): set goal</small></h3>
      <canvas id="map"></canvas>
    </div>
    <aside class="panel">
      <div class="card"><h3>DRIVE MODE</h3>
        <div class="toggle">
          <button id="btn-manual" class="active">Manual</button>
          <button id="btn-auto">Auto</button>
        </div>
        <p class="hint">Manual: W/S tiến lùi · A/D xoay · Q/E đi ngang</p>
        <p>Robot: <b id="robot-state">OK</b>
          <button id="btn-reset-robot" class="mini">Reset robot</button></p>
      </div>
      <div class="card"><h3>NAVIGATION <small>(Auto mode)</small></h3>
        <div class="btnrow">
          <button id="btn-nav-stop" class="danger">■ Stop</button>
          <button id="btn-nav-patrol">Patrol</button>
        </div>
        <p>Status: <b id="nav-status">IDLE</b></p>
        <p>Telegram alert: <b id="tg-status">MOCK</b></p>
      </div>
      <div class="card"><h3>SLAM BUILD</h3>
        <p>State: <b id="slam-state">idle</b></p>
        <div class="btnrow">
          <button id="btn-slam-start">Start</button>
          <button id="btn-slam-pause">Pause</button>
          <button id="btn-slam-end" class="danger">End build</button>
        </div>
        <div class="btnrow">
          <button id="btn-slam-save">Save map</button>
          <button id="btn-slam-reset">Reset</button>
        </div>
        <div class="btnrow">
          <select id="map-select"></select>
          <button id="btn-slam-load">Load</button>
        </div>
      </div>
      <div class="card"><h3>DETECTIONS</h3>
        <ul id="det-list"></ul>
      </div>
    </aside>
  </section>
</main>
<script src="map.js"></script>
<script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Viết style.css**

```css
* { box-sizing: border-box; margin: 0; }
body { font-family: "Segoe UI", system-ui, sans-serif; background: #e8e8ea;
       color: #111; font-size: 13px; }
header { display: flex; justify-content: space-between; align-items: center;
         background: #fff; padding: 8px 16px; border-bottom: 2px solid #111; }
.logo { font-weight: 800; letter-spacing: 1px; font-size: 16px;
        border: 2px solid #111; padding: 2px 10px; }
.dots { display: flex; gap: 14px; align-items: center; }
.dot { width: 9px; height: 9px; border-radius: 50%; background: #c33;
       display: inline-block; margin-right: 4px; }
.dot.ok { background: #2c5; }
main { padding: 10px; display: flex; flex-direction: column; gap: 10px; }
.card { background: #fff; border-radius: 6px; padding: 8px 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,.12); }
.card h3 { font-size: 11px; letter-spacing: 1px; margin-bottom: 6px; }
.card h3 small { color: #999; font-weight: 400; text-transform: none; }
.row-top { display: grid; grid-template-columns: 1.2fr 1.2fr 1fr 1fr; gap: 10px; }
.row-top img, #lidar { width: 100%; aspect-ratio: 4/3; object-fit: cover;
                       background: #111; border-radius: 4px; }
#lidar { aspect-ratio: 1/1; background: #fff; border: 1px solid #eee; }
.rpy { display: flex; justify-content: space-around; margin: 10px 0; }
.rpy label { display: block; color: #999; font-size: 10px; text-align: center; }
.rpy b { font-size: 26px; font-weight: 700; }
.imu-table { width: 100%; font-size: 12px; }
.imu-table td { background: #f3f3f5; padding: 3px 6px; text-align: right; }
.imu-table th { text-align: left; color: #777; font-weight: 600; }
.row-main { display: grid; grid-template-columns: 1fr 280px; gap: 10px; }
.map-card { min-height: 480px; display: flex; flex-direction: column; }
#map { flex: 1; width: 100%; background: #b9b9bd; border-radius: 4px;
       cursor: grab; }
.panel { display: flex; flex-direction: column; gap: 10px; }
.toggle { display: flex; border: 1px solid #ccc; border-radius: 6px;
          overflow: hidden; }
.toggle button { flex: 1; padding: 8px; border: 0; background: #f5f5f7;
                 cursor: pointer; font-weight: 600; }
.toggle button.active { background: #111; color: #fff; }
.btnrow { display: flex; gap: 6px; margin: 6px 0; }
.btnrow button, .mini { flex: 1; padding: 6px 8px; border: 1px solid #ccc;
        border-radius: 5px; background: #fff; cursor: pointer; font-weight: 600; }
.btnrow button:hover { background: #f0f0f2; }
button.danger { color: #c33; border-color: #c33; }
.hint { color: #888; font-size: 11px; margin: 6px 0; }
#det-list { list-style: none; max-height: 140px; overflow-y: auto; }
#det-list li { padding: 3px 0; border-bottom: 1px solid #eee; }
select { flex: 1; padding: 6px; border: 1px solid #ccc; border-radius: 5px; }
```

- [ ] **Step 3: Viết map.js**

```javascript
// MapView: vẽ occupancy PNG + robot + path + goal; pan/zoom/right-click goal.
class MapView {
  constructor(canvas, onGoal) {
    this.cv = canvas; this.ctx = canvas.getContext("2d");
    this.onGoal = onGoal;
    this.img = null; this.meta = null;
    this.scale = 3.0; this.tx = 0; this.ty = 0;
    this.pose = [0, 0, 0]; this.path = []; this.goal = null;
    this._fitted = false;
    canvas.addEventListener("mousedown", e => {
      if (e.button !== 0) return;
      this._drag = [e.offsetX, e.offsetY];
    });
    canvas.addEventListener("mousemove", e => {
      if (!this._drag) return;
      this.tx += e.offsetX - this._drag[0];
      this.ty += e.offsetY - this._drag[1];
      this._drag = [e.offsetX, e.offsetY];
    });
    window.addEventListener("mouseup", () => this._drag = null);
    canvas.addEventListener("wheel", e => {
      e.preventDefault();
      const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      this.scale *= f;
      this.tx = e.offsetX - (e.offsetX - this.tx) * f;
      this.ty = e.offsetY - (e.offsetY - this.ty) * f;
    });
    canvas.addEventListener("dblclick", () => this.fit());
    canvas.addEventListener("contextmenu", e => {
      e.preventDefault();
      const w = this.canvasToWorld(e.offsetX, e.offsetY);
      if (w) this.onGoal(w[0], w[1]);
    });
  }

  setMap(blob, meta) {
    this.meta = meta;
    createImageBitmap(blob).then(b => {
      this.img = b;
      if (!this._fitted) { this.fit(); this._fitted = true; }
    });
  }

  fit() {
    if (!this.img) return;
    const s = Math.min(this.cv.width / this.img.width,
                       this.cv.height / this.img.height);
    this.scale = s;
    this.tx = (this.cv.width - this.img.width * s) / 2;
    this.ty = (this.cv.height - this.img.height * s) / 2;
  }

  // PNG đã flipud: pixel row 0 = y world lớn nhất
  worldToCanvas(x, y) {
    const m = this.meta; if (!m) return null;
    const px = (x - m.origin[0]) / m.res;
    const py = m.h - (y - m.origin[1]) / m.res;
    return [px * this.scale + this.tx, py * this.scale + this.ty];
  }

  canvasToWorld(cx, cy) {
    const m = this.meta; if (!m) return null;
    const px = (cx - this.tx) / this.scale;
    const py = (cy - this.ty) / this.scale;
    return [m.origin[0] + px * m.res, m.origin[1] + (m.h - py) * m.res];
  }

  draw() {
    const { ctx, cv } = this;
    cv.width = cv.clientWidth; cv.height = cv.clientHeight;
    ctx.fillStyle = "#b9b9bd"; ctx.fillRect(0, 0, cv.width, cv.height);
    if (this.img) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(this.img, this.tx, this.ty,
                    this.img.width * this.scale, this.img.height * this.scale);
    }
    if (this.path.length > 1) {
      ctx.strokeStyle = "#39f"; ctx.lineWidth = 2; ctx.beginPath();
      this.path.forEach((p, i) => {
        const c = this.worldToCanvas(p[0], p[1]);
        i ? ctx.lineTo(c[0], c[1]) : ctx.moveTo(c[0], c[1]);
      });
      ctx.stroke();
    }
    if (this.goal) {
      const g = this.worldToCanvas(this.goal[0], this.goal[1]);
      if (g) { ctx.fillStyle = "#f90"; ctx.beginPath();
        ctx.arc(g[0], g[1], 5, 0, 7); ctx.fill(); }
    }
    const r = this.worldToCanvas(this.pose[0], this.pose[1]);
    if (r) {
      ctx.fillStyle = "#e22"; ctx.beginPath(); ctx.arc(r[0], r[1], 6, 0, 7); ctx.fill();
      ctx.strokeStyle = "#e22"; ctx.lineWidth = 2; ctx.beginPath();
      ctx.moveTo(r[0], r[1]);
      ctx.lineTo(r[0] + 14 * Math.cos(-this.pose[2]),
                 r[1] + 14 * Math.sin(-this.pose[2]));
      ctx.stroke();
    }
  }
}
```

**Lưu ý dấu yaw:** canvas y hướng xuống, world y hướng lên → mũi hướng dùng `-pose[2]`. Verify bằng mắt ở Step 5; nếu mũi quay ngược chiều thật → bỏ dấu trừ.

- [ ] **Step 4: Viết app.js**

```javascript
// WS + điều khiển + render panels.
const $ = id => document.getElementById(id);
const api = (path, body) => fetch(path, { method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body || {}) });

const mapView = new MapView($("map"), (x, y) => {
  if (mode === "auto") api("/api/nav/goal", { x, y });
});
let mode = "manual";
let lastMsg = {};

// ---- WebSocket ----
const ws = new WebSocket(`ws://${location.host}/ws/stream`);
ws.binaryType = "blob";
const urls = { cam: null, depth: null };
ws.onmessage = async ev => {
  if (typeof ev.data === "string") { onTelemetry(JSON.parse(ev.data)); return; }
  const ch = new Uint8Array(await ev.data.slice(0, 1).arrayBuffer())[0];
  const payload = ev.data.slice(1);
  if (ch === 1) setImg("cam", payload);
  else if (ch === 2) setImg("depth", payload);
  else if (ch === 3) mapView.setMap(payload, lastMsg.map_meta);
};
function setImg(id, blob) {
  const old = urls[id];
  urls[id] = URL.createObjectURL(blob);
  $(id).src = urls[id];
  if (old) URL.revokeObjectURL(old);
}

function onTelemetry(m) {
  lastMsg = m;
  $("dot-cam").className = "dot ok"; $("dot-depth").className = "dot ok";
  $("dot-lidar").className = m.lidar_ranges?.length ? "dot ok" : "dot";
  $("dot-imu").className = m.imu ? "dot ok" : "dot";
  if (m.imu) {
    const d = r => (r * 180 / Math.PI).toFixed(1);
    $("roll").textContent = d(m.imu.rpy[0]);
    $("pitch").textContent = d(m.imu.rpy[1]);
    $("yaw").textContent = d(m.imu.rpy[2]);
    ["ax","ay","az"].forEach((id, i) => $(id).textContent = m.imu.accel[i].toFixed(2));
    ["gx","gy","gz"].forEach((id, i) => $(id).textContent = m.imu.gyro[i].toFixed(3));
  }
  $("slam-state").textContent = m.slam_state;
  $("nav-status").textContent = m.nav_status;
  $("tg-status").textContent = m.telegram;
  $("tg-status").style.color = m.telegram === "ARMED" ? "#2c5" : "#c80";
  $("robot-state").textContent = m.fallen ? "FALLEN" : "OK";
  $("robot-state").style.color = m.fallen ? "#c33" : "#2c5";
  mapView.pose = m.pose; mapView.path = m.path || []; mapView.goal = m.goal;
  drawLidar(m.lidar_ranges, m.lidar_angles);
  $("det-list").innerHTML = (m.detections || []).slice(-8).reverse()
    .map(d => `<li>⚠ <b>${d.cls}</b> ${d.conf} @ (${d.x}, ${d.y})</li>`).join("");
}

// ---- Lidar polar ----
function drawLidar(ranges, angles) {
  const cv = $("lidar"), ctx = cv.getContext("2d");
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, cv.width, cv.height);
  const cx = cv.width / 2, cy = cv.height / 2, sc = cv.width / 2 / 8.0;
  ctx.strokeStyle = "#dde";
  for (let r = 2; r <= 8; r += 2) {
    ctx.beginPath(); ctx.arc(cx, cy, r * sc, 0, 7); ctx.stroke();
  }
  ctx.strokeStyle = "#3a3"; ctx.beginPath();
  ctx.moveTo(cx, cy); ctx.lineTo(cx, cy - 8 * sc); ctx.stroke();
  if (!ranges) return;
  ctx.fillStyle = "#36c";
  for (let i = 0; i < ranges.length; i++) {
    if (ranges[i] >= 7.99) continue;
    const a = angles[i] + Math.PI / 2;   // angle 0 = mũi robot = hướng lên
    ctx.fillRect(cx + Math.cos(a) * ranges[i] * sc - 1,
                 cy - Math.sin(a) * ranges[i] * sc - 1, 2, 2);
  }
  ctx.fillStyle = "#e22"; ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 7); ctx.fill();
}

// ---- Drive mode + teleop ----
$("btn-manual").onclick = () => setMode("manual");
$("btn-auto").onclick = () => setMode("auto");
function setMode(m) {
  mode = m;
  $("btn-manual").classList.toggle("active", m === "manual");
  $("btn-auto").classList.toggle("active", m === "auto");
  api("/api/mode", { mode: m });
}
const keys = {};
onkeydown = e => { keys[e.key.toLowerCase()] = true; };
onkeyup = e => { keys[e.key.toLowerCase()] = false; };
setInterval(() => {
  if (mode !== "manual") return;
  const vx = (keys.w ? 1 : 0) + (keys.s ? -1 : 0);
  const vy = (keys.q ? 1 : 0) + (keys.e ? -1 : 0);
  const wz = (keys.a ? 1 : 0) + (keys.d ? -1 : 0);
  api("/api/teleop", { vx, vy, wz });
}, 100);

// ---- Buttons ----
$("btn-reset-robot").onclick = () => api("/api/robot/reset");
$("btn-nav-stop").onclick = () => api("/api/nav/stop");
$("btn-nav-patrol").onclick = () => api("/api/nav/patrol");
$("btn-slam-start").onclick = () => api("/api/slam/start");
$("btn-slam-pause").onclick = () => api("/api/slam/pause");
$("btn-slam-end").onclick = () => api("/api/slam/end");
$("btn-slam-reset").onclick = () => api("/api/slam/reset");
$("btn-slam-save").onclick = async () => {
  const name = prompt("Tên map:", "map1");
  if (name) { await api("/api/slam/save", { name }); loadMapList(); }
};
$("btn-slam-load").onclick = () =>
  api("/api/slam/load", { name: $("map-select").value });
async function loadMapList() {
  const r = await fetch("/api/slam/maps").then(r => r.json());
  $("map-select").innerHTML = r.maps.map(m => `<option>${m}</option>`).join("");
}
loadMapList();

// ---- render loop ----
(function loop() { mapView.draw(); requestAnimationFrame(loop); })();
```

- [ ] **Step 5: Chạy thử end-to-end**

```bash
MUJOCO_GL=egl python app.py
```

Mở http://localhost:8000. Verify checklist (sửa JS/CSS nếu lỗi):
1. 4 status dots xanh, camera + depth stream chạy.
2. Lidar polar có điểm xanh (tường/đồ vật).
3. IMU số nhảy.
4. Manual: giữ W → robot BƯỚC ĐI THẬT (chân RL policy), SLAM Start → map hiện dần.
5. Pan/zoom/double-click map OK; mũi hướng robot đúng chiều.
6. Auto + right-click map → path xanh + robot tự đi (xoay chậm — wz max 0.2 là bình thường) → DONE.
7. Save map → có trong dropdown → Reset → Load lại OK.
8. Lái robot tới gần cốc trong scene → bbox hiện, DETECTIONS có entry, `captures/` có ảnh.

- [ ] **Step 6: Commit**

```bash
git add web/static/
git commit -m "feat: web dashboard (camera/depth/lidar/imu panels, slam map, controls)"
```

---

### Task 18: Integration test + README + patrol waypoints

**Files:**
- Create: `tests/test_integration.py`, `README.md`
- Modify: `configs/config.yaml` (điền patrol_waypoints sau khi có map)

- [ ] **Step 1: Viết integration test**

`tests/test_integration.py`:
```python
"""End-to-end headless: lái robot, build map, detect, mock telegram."""
import glob

import numpy as np
import pytest

from apprt.runtime import PatrolApp


@pytest.mark.slow
@pytest.mark.render
def test_full_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    app = PatrolApp("configs/config.yaml", enable_detector=True)
    app.telegram.out_dir = tmp_path
    app.set_mode("manual")
    app.slam.start()

    plan = [((1.0, 0, 0), 4.0), ((0, 0, 1.0), 3.0), ((1.0, 0, 0), 4.0)]
    for cmd, dur in plan:
        app.set_teleop(*cmd)
        for _ in range(int(dur / app.model.opt.timestep)):
            app._sim_tick()
        app._perception_tick()

    snap = app.state.snapshot()
    assert not snap["fallen"], "robot không được ngã trong kịch bản chuẩn"
    assert np.sum(app.slam.grid.occupied_mask()) > 200, "map phải có tường/đồ"
    gt = app.policy.base_pose2d()
    assert np.linalg.norm(np.array(snap["pose"][:2]) - gt[:2]) < 1.0, \
        "SLAM pose phải bám pose thật (scan matching kìm drift)"
    app.telegram.close()
    if snap["detections"]:
        assert len(glob.glob(str(tmp_path / "*.jpg"))) >= 1
```

- [ ] **Step 2: Chạy integration test**

Run: `MUJOCO_GL=egl pytest tests/test_integration.py -v -m "" -s`
Expected: PASS (~2-4 phút: YOLO load + 11s sim).

- [ ] **Step 3: Chốt patrol waypoints**

Chạy app, build map hoàn chỉnh (lái 1 vòng phòng), save map `default`. Nhìn map + vị trí các cốc trong scene, chọn 4–6 điểm quét phủ vùng sàn trống, điền vào `configs/config.yaml` `nav.patrol_waypoints` (toạ độ world mét, ví dụ `[[2.0, 1.0], [4.0, -2.0], [0.0, -4.0], [-3.0, 3.0]]` — chỉnh theo map thật).

- [ ] **Step 4: Viết README.md**

Nội dung bắt buộc:
1. Mô tả demo + ảnh chụp dashboard.
2. Cài đặt: `pip install -r requirements.txt`; clone `unitree_rl_lab` vào `third_party/` (chỉ dùng `policy.onnx`); **không cần build scene — dùng nguyên `scene/scene.xml`**.
3. **Hướng dẫn tạo Telegram bot:** @BotFather → `/newbot` → đặt tên → copy TOKEN; nhắn bot 1 tin → mở `https://api.telegram.org/bot<TOKEN>/getUpdates` → copy `message.chat.id`; tạo `.env` từ `.env.example` điền 2 giá trị. Không có `.env` → MOCK mode (ảnh lưu `captures/`).
4. Chạy: `MUJOCO_GL=egl python app.py` → http://localhost:8000.
5. Kịch bản demo: (a) Manual + SLAM Start → lái W/A/S/D vòng quanh phòng build map; (b) End build + Save map; (c) chuyển Auto → Patrol hoặc right-click chọn điểm; (d) robot quét, gặp cốc → bbox + ảnh lên Telegram.
6. Ghi chú: robot KHÔNG xoay tại chỗ được (đặc tính checkpoint policy — gait tắt khi vx~0), chỉ quay theo cung bán kính ~1.5m với wz max 0.2 rad/s; scene chỉ có cốc làm vật mục tiêu (không có mesh cờ lê/tua vít — prompt detector vẫn đủ 3 class).
7. Test: `MUJOCO_GL=egl pytest -m "not slow"` (nhanh) / `pytest -m ""` (full).

- [ ] **Step 5: Chạy toàn bộ test suite lần cuối**

Run: `MUJOCO_GL=egl pytest -v -m ""`
Expected: tất cả PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_integration.py README.md configs/config.yaml
git commit -m "feat: integration test, README with telegram guide, patrol waypoints"
```

---

## Rủi ro & cách xử lý (đọc trước khi code)

| Rủi ro | Phát hiện ở | Xử lý |
|---|---|---|
| **Sim2sim gap: policy 29dof Isaac→MuJoCo ngã** (rủi ro lớn nhất) | Task 4 Step 4 | Debug guide 5 bước trong Task 4; thử obs layout fallback; warmup đứng 1s; nếu bế tắc hẳn → hỏi user trước khi đổi hướng |
| MjSpec API khác kỳ vọng (add_camera/quat) | Task 3 Step 4 | Fallback ElementTree in-RAM + `from_xml_string` (vẫn không sửa file) |
| Tên joint waist khác POLICY_JOINTS | Task 4 Step 4 | In danh sách joint, sửa list theo tên thật |
| Lidar self-hit (geom robot không group 2/3) | Task 6 Step 2 | Tăng min_range; kiểm group bằng grep (không sửa scene) |
| YOLO-World conf thấp trên cốc STL không texture | Task 12 Step 5 | Hạ conf 0.1–0.2; nếu vẫn kém → báo user, đề xuất fine-tune synthetic (ngoài scope) |
| Scan matching trượt khi phòng ít feature | Task 18 Step 2 | Tăng window search, giảm odom noise |
| Sim chậm hơn realtime khi bật YOLO | Task 17 Step 5 | Giảm rate_hz, YOLO device="cuda:0", giảm img size |
| Robot 29dof đi chậm/xoay chậm làm demo dài | Task 17 Step 5 | Chấp nhận (giới hạn policy); teleop vx max 0.5 đủ nhanh |

## Self-review đã thực hiện

- **Spec coverage:** scene nguyên trạng + camera runtime ✓(T3) locomotion 29dof onnx ✓(T4) depth-HSV/lidar/IMU ✓(T5,T6) SLAM scan-matching + save/load + states ✓(T7–T9) navigation A*+pure-pursuit (w_max 0.2)+patrol ✓(T10,T11) YOLO-World 3 class + bbox ✓(T12) world-projection + dedupe + capture ✓(T13) Telegram ARMED/MOCK + guide ✓(T14,T18) web dashboard đủ panel theo design + manual/auto + slam build controls ✓(T16,T17) error handling (fallen+reset, perception crash-safe, telegram retry, WS disconnect) ✓(T15,T16) testing ✓(từng task + T18).
- **Placeholder scan:** không còn TBD/TODO; mọi step code đều có code đầy đủ; các bước "verify bằng mắt" có lệnh + tiêu chí cụ thể.
- **Type consistency:** `OccupancyGrid(size_m, resolution, l_occ, l_free, l_clamp, max_range)` thống nhất T7/T9/T11; `SlamSystem.{start,pause,end_build,reset,save,load,on_scan,set_pose,pose,grid,state}` T9/T15/T16; `Navigator.{set_goal,stop,start_patrol,update,status,path,goal}` T11/T15/T16; `G1WalkPolicy.{set_command,step,reset,fallen,base_pose2d,base_vel_body,base_qadr,base_vadr}` T4/T15; `load_scene(sim_cfg) -> (model, meta)` T3/T6/T15; `Detection(cls,conf,xyxy)` T12/T15; telemetry keys server (T16) khớp app.js (T17).

