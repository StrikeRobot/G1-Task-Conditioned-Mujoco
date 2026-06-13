"""Entrypoint: MUJOCO_GL=egl python app.py [--config configs/config.yaml]

The offscreen renderers (camera/depth for the web dashboard) use the EGL
backend. An optional native MuJoCo 3D viewer window opens on top via GLFW when
a display is available; disable it with --no-viewer (e.g. on a headless server).
"""
import argparse
import logging
import os

import uvicorn

from apprt.runtime import PatrolApp
from web.server import create_app


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--no-detector", action="store_true")
    ap.add_argument("--no-viewer", action="store_true",
                    help="do not open the native MuJoCo 3D window")
    args = ap.parse_args()

    enable_viewer = bool(os.environ.get("DISPLAY")) and not args.no_viewer
    patrol = PatrolApp(args.config, enable_detector=not args.no_detector,
                       enable_viewer=enable_viewer)
    patrol.start()
    web_cfg = patrol.cfg["web"]
    try:
        uvicorn.run(create_app(patrol), host=web_cfg["host"], port=web_cfg["port"])
    finally:
        patrol.stop()


if __name__ == "__main__":
    main()
