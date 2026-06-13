"""FastAPI: REST controls + 1 multiplexed WebSocket.

WS /ws/stream:
  - binary: 1 channel byte + payload (0x01 rgb jpeg, 0x02 depth jpeg, 0x03 map png)
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
        elif action == "delete":
            name = (body or {}).get("name", "")
            p = MAPS_DIR / f"{name}.npz"
            if name and p.exists():
                p.unlink()
            else:
                return {"ok": False, "error": "map not found"}
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
                if i % 2 == 0 and snap.get("map_png"):     # ~7Hz (map regen is 10Hz)
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
