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
    assert len(list(tmp_path.glob("*.jpg"))) == 1  # still saved locally
