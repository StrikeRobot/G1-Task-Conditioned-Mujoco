"""Send detection images to Telegram (thread + queue, does not block sim).

Create a bot: message @BotFather -> /newbot -> get the TOKEN.
Get chat_id: send the bot any message then open
https://api.telegram.org/bot<TOKEN>/getUpdates -> message.chat.id.
Fill into .env (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID). Missing -> MOCK mode.
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
            try:
                self._handle(item)
            except Exception:  # worker must survive any single-item error
                log.exception("telegram worker item failed")

    def _handle(self, item):
        img, caption = item
        fname = self.out_dir / f"det_{int(time.time() * 1000)}.jpg"
        if not cv2.imwrite(str(fname), img):
            log.warning("telegram: failed to save %s", fname)
        if self.status == "MOCK":
            log.info("MOCK telegram: %s -> %s", caption, fname)
            return
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            log.warning("telegram: imencode failed, skipping send")
            return
        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        for i in range(self.retries):
            try:
                r = requests.post(url, data={"chat_id": self.chat_id,
                                             "caption": caption},
                                  files={"photo": ("det.jpg", buf.tobytes())},
                                  timeout=10)
                if r.status_code == 200:
                    break
                desc = ""
                try:
                    desc = r.json().get("description", "")
                except Exception:
                    pass
                log.warning("telegram HTTP %s: %s (try %d)", r.status_code, desc, i + 1)
                if "chat not found" in desc.lower():
                    log.warning("telegram: open the bot in Telegram and press START, "
                                "then set the correct TELEGRAM_CHAT_ID in .env")
                    break  # config error won't fix on retry
            except Exception as e:
                log.warning("telegram error: %s (try %d)", e, i + 1)
            time.sleep(1)
