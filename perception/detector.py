"""YOLO26 object detector (COCO classes), filtered by the list of classes of interest.

YOLO26 is a fixed 80-class COCO detector (no open-vocab/text prompt). Among the
requested classes (cup/wrench/screwdriver) only "cup" exists in COCO; wrench and
screwdriver do not exist so they are ignored (the scene also lacks those 2 objects).
"""
import logging
from dataclasses import dataclass

import cv2

log = logging.getLogger(__name__)


@dataclass
class Detection:
    cls: str
    conf: float
    xyxy: tuple  # (x1, y1, x2, y2) pixel


COLORS = {"cup": (60, 200, 60), "wrench": (60, 120, 255),
          "screwdriver": (220, 80, 220)}


class Detector:
    def __init__(self, classes, conf, weights="yolo26s.pt", device=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device
        # map id->name of the model, and the set of ids of interest present in the COCO vocab
        names = self.model.names
        wanted = set(classes)
        self.id2name = dict(names)
        self.wanted_ids = {i for i, n in names.items() if n in wanted}
        missing = wanted - {names[i] for i in self.wanted_ids}
        if missing:
            log.warning("YOLO26 has no class %s in COCO — will be ignored", sorted(missing))
        if not self.wanted_ids:
            log.warning("none of the requested classes are in the model vocab")

    def detect(self, bgr):
        res = self.model.predict(bgr, conf=self.conf, verbose=False,
                                 device=self.device)[0]
        out = []
        for b in res.boxes:
            cid = int(b.cls.item())
            if cid not in self.wanted_ids:
                continue
            out.append(Detection(
                cls=self.id2name[cid],
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
