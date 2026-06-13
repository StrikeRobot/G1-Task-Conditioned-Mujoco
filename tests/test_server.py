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


def test_slam_delete():
    from pathlib import Path
    c, _ = _client()
    p = Path("maps/__unittest_delete__.npz")
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(b"x")
    assert c.post("/api/slam/delete", json={"name": "__unittest_delete__"}).json()["ok"]
    assert not p.exists()
    assert c.post("/api/slam/delete", json={"name": "__nope__"}).json()["ok"] is False


def test_nav_goal():
    c, fake = _client()
    assert c.post("/api/nav/goal", json={"x": 1.5, "y": -2.0}).status_code == 200
    assert fake.nav.goal == (1.5, -2.0)
