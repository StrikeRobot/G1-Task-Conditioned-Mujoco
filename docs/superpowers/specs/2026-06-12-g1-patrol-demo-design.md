# G1 Patrol & Detection Demo — Design Spec

**Date:** 2026-06-12
**Status:** Approved by user

## Goal

Demo robot Unitree G1 (mô phỏng MuJoCo) tuần tra trong phòng: build map bằng SLAM (điều khiển thủ công), sau đó chuyển sang chế độ auto navigation quét các waypoint chỉ định để tìm vật thể vứt dưới sàn (cốc, cờ lê, tua vít). Web dashboard hiển thị camera RGB (bbox overlay), depth map (HSV colormap), lidar, SLAM map, IMU. Khi phát hiện vật thể: capture ảnh và gửi lên Telegram.

Layout web theo ảnh tham chiếu `photo_2026-06-12_21-10-42.jpg` (kiểu "STRIKE ROBOT" dashboard).

## Decisions (đã chốt với user)

| Quyết định | Lựa chọn |
|---|---|
| Locomotion | **RL walking policy 29-DOF** `loco/policy_29dof.pt` từ G1_deploy (TorchScript, obs 96, yaw ±1.57) — **quay tại chỗ + ổn định**. (Ban đầu dùng unitree_rl_lab velocity/v0 nhưng nó chỉ arc được; đổi 2026-06-13.) |
| Robot & Scene | **Dùng nguyên scene + robot 29-DOF có sẵn, KHÔNG sửa file nào trong `scene/`** (user 2026-06-12) |
| Detection | **YOLO26** (ultralytics, COCO) lọc theo class quan tâm. Ban đầu chọn YOLO-World open-vocab nhưng conf trên cốc STL chỉ ~0.22; YOLO26 cho ~0.85–0.93 nên đổi sang (2026-06-13). COCO chỉ có "cup"; wrench/screwdriver không có (scene cũng không có mesh) → bỏ qua. |
| SLAM | **Scan-matching SLAM thật** (odometry noise + correlative matching + occupancy grid) |
| Telegram | Chưa có bot — kèm hướng dẫn tạo qua @BotFather; code chạy mock khi thiếu token |

## Architecture

Một process Python duy nhất + web client:

```
Sim loop (thread): MuJoCo 500Hz physics → RL policy 50Hz → PD position targets
  Sensors: RGB camera (head, 640×480 ~30Hz), depth render → HSV colormap,
           lidar 2D (360 rays mj_ray, 8m, đặt ở ngực), IMU (pelvis)
       │
       ├─► Perception thread: YOLO-World 5–10Hz → bbox + detection events
       ├─► SLAM thread: odometry (noisy) + scan matching → occupancy grid (5cm/cell)
       ├─► Navigation: A* (global, grid inflate) + pure pursuit (local) → (vx, vy, ωz)
       └─► FastAPI + WebSocket: stream video (JPEG binary WS), telemetry/map (JSON WS)
              └─► Telegram alert: sendPhoto khi có detection mới (dedupe theo vị trí world)
```

## Components

### 1. Simulation (`sim/`)
- **Scene dùng nguyên trạng, không sửa file nào trong `scene/`** (yêu cầu user 2026-06-12). Robot = G1 29-DOF có sẵn trong scene (`scene/robot/robot.xml`).
- Camera RGB/depth inject lúc runtime bằng `mujoco.MjSpec` (thêm camera `head_cam` vào body torso trước khi compile — file XML không đổi). Depth từ MuJoCo render, normalize rồi áp HSV colormap.
- Lidar: 360 tia raycast 360°, tầm 8m, 10Hz; origin từ `xpos` body torso (không cần site).
- IMU: quaternion → RPY, accel m/s², gyro rad/s.
- Vật mục tiêu trên sàn: các cốc có sẵn trong scene; không có mesh cờ lê/tua vít — detector vẫn prompt 3 class nhưng demo phát hiện thực tế là cốc.

### 2. Locomotion (`locomotion/`)
- Policy velocity-tracking `policy.onnx` từ **unitree_rl_lab `deploy/robots/g1_29dof`** (velocity/v0), chạy bằng onnxruntime in-process, 50Hz.
- Obs 480-dim (6 term × history 5, term-major): ang_vel×0.2, projected gravity, cmd, (q−default), dq×0.05, last_action raw. Output 29 = position targets (`default + 0.25×action`) → PD gains theo deploy.yaml (hip 100/knee 150/ankle 40/waist 200/arm 40) override lên position actuators sẵn có lúc runtime (`actuator_gainprm/biasprm`).
- **Scene chứa nhiều object nên obs tensor build hoàn toàn theo tên joint robot** (qposadr/dofadr per-joint, base = `floating_base_joint`), không slice qpos/qvel toàn cục — chống ô nhiễm obs nếu scene thêm object động (yêu cầu user 2026-06-12).
- Giới hạn lệnh: vx [-0.5, 1.0], vy [-0.3, 0.3], wz [-0.2, 0.2].
- Interface: `set_command(vx, vy, wz)` — nguồn từ manual teleop hoặc navigation.
- Fall detection (nghiêng > ~60°) → dừng + cho phép reset từ web.

### 3. SLAM (`slam/`)
- Odometry: tích phân vận tốc ước lượng + gyro, cộng noise/drift mô phỏng.
- Scan matching: correlative scan matching / ICP với map hiện tại để hiệu chỉnh pose.
- Occupancy grid log-odds, 5cm/cell, đủ phủ phòng 20×15m.
- API: start/pause/end build, save/load map (`.npz`), reset. State: `idle/building/paused/done`.

### 4. Navigation (`nav/`)
- Goal: right-click trên map (Auto mode) hoặc danh sách waypoint patrol định sẵn (configs).
- Global: A* trên grid đã inflate theo bán kính robot. Local: pure pursuit → (vx, ωz).
- Tốc độ xoay tối đa 0.2 rad/s theo giới hạn policy — pure pursuit dùng w_max=0.2 (xoay tại chỗ sớm khi lệch hướng nhiều).
- Replan khi path bị chặn. Status: `IDLE / NAVIGATING (wp i/N) / DONE`. Controls: Stop, Reset goal.

### 5. Perception (`perception/`)
- ultralytics YOLO-World, classes `["cup", "wrench", "screwdriver"]`, GPU, 5–10Hz.
- BBox overlay vẽ lên frame trước khi stream.
- Detection event khi confidence > threshold (tune được trong config):
  - Chiếu tâm bbox qua depth + camera intrinsics + robot pose → vị trí world.
  - Dedupe không gian: vật trong bán kính R của detection cũ cùng class → bỏ qua.
  - Capture frame (có bbox) → queue gửi Telegram + lưu local `captures/`.
- **Rủi ro đã biết:** STL không texture có thể làm YOLO-World confidence thấp. Mitigation: tune threshold trước; nếu không đạt, fallback là fine-tune YOLO trên ảnh synthetic render từ chính scene (ngoài scope spec này, ghi nhận làm phase sau nếu cần).

### 6. Telegram (`alerts/`)
- HTTP Bot API `sendPhoto`, caption: class, confidence, vị trí (x, y), timestamp.
- Token + chat_id từ `.env` (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`). Thiếu token → mock mode: lưu ảnh local + log, web hiển thị "Telegram: MOCK".
- README kèm hướng dẫn tạo bot: @BotFather → `/newbot` → lấy token; lấy chat_id qua `getUpdates`.
- Gửi async qua queue, không block sim loop.

### 7. Web dashboard (`web/`)
- FastAPI + vanilla JS/HTML/CSS, layout đúng ảnh design:
  - Header: logo + status dots (Camera/Depth/LIDAR/IMU).
  - Hàng trên: CAMERA (bbox overlay) · DEPTH (colormap) · LIDAR (polar top-down) · IMU (ROLL/PITCH/YAW lớn + bảng accel/gyro).
  - Giữa: SLAM MAP canvas — pan (drag), zoom (wheel), double-click fit, right-click đặt goal (Auto); vẽ robot pose, path, waypoints; minimap góc.
  - Phải: DRIVE MODE (Manual/Auto toggle) · NAVIGATION (Stop, Reset goal, status, "Telegram alert: ARMED/MOCK") · SLAM BUILD (state, End build/Pause, Save map/Reset, dropdown chọn map/Load).
  - Panel detection log: ảnh capture gần nhất + danh sách phát hiện.
- Manual control: WASD/phím mũi tên (giữ để đi), hiển thị hướng dẫn phím.
- Transport: 1 WS binary cho video frames (camera + depth, JPEG), 1 WS JSON cho telemetry (pose, lidar, IMU, nav status, detections), map gửi dạng PNG base64/binary khi có thay đổi (throttle ~2Hz).

### Concurrency
- Thread sim (physics + policy), thread perception, thread SLAM; FastAPI async event loop riêng.
- Giao tiếp qua queue/lock-protected snapshot; sim loop không bao giờ bị block bởi I/O.

## Cấu trúc thư mục

```
dsc_lab_g1/
├── sim/          # MuJoCo wrapper, sensors (camera, depth, lidar, imu)
├── locomotion/   # policy runner, velocity command interface
├── slam/         # odometry, scan matching, occupancy grid
├── nav/          # A*, pure pursuit, waypoint manager
├── perception/   # YOLO-World detector, spatial dedupe, capture
├── alerts/       # telegram sender (+ mock)
├── web/          # FastAPI app + static frontend
├── scene/        # (đã có) + thêm wrench/screwdriver assets
├── configs/      # config.yaml (thresholds, waypoints, slam params)
├── captures/     # ảnh detection đã capture
├── maps/         # map đã save
└── tests/        # unit + integration tests
```

## Error handling

- Policy load fail / robot ngã → web hiển thị trạng thái lỗi + nút reset robot về pose đứng.
- YOLO/GPU fail → camera vẫn stream không bbox, status dot perception đỏ.
- Telegram gửi fail → retry 3 lần, log, không làm rơi detection (vẫn lưu local).
- WS client disconnect → sim vẫn chạy bình thường.

## Testing

- **Unit:** occupancy grid update với scan synthetic; scan matching hội tụ với offset biết trước; A* trên grid mẫu (có/không đường đi); spatial dedupe; HSV depth colormap.
- **Integration (headless):** chạy sim N giây — robot nhận lệnh vận tốc và di chuyển, map build ra occupied cells hợp lý, detect được cốc trong scene, mock telegram nhận đúng ảnh + caption.
- **Manual acceptance:** mở web, lái robot build map, save map, chuyển Auto, đặt goal, robot tự đi, phát hiện vật → ảnh lên Telegram thật.

## Environment

- Python 3.13, MuJoCo 3.8.1, torch 2.11 (đã có), cài thêm: ultralytics, uvicorn, websockets, python-dotenv.
- GPU: RTX 5060 Ti 16GB — đủ cho YOLO-World realtime + render.
