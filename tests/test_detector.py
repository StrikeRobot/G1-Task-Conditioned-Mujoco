import numpy as np
import pytest

from perception.detector import Detection, draw_detections


def test_draw_detections_marks_pixels():
    img = np.zeros((100, 100, 3), np.uint8)
    dets = [Detection(cls="cup", conf=0.9, xyxy=(10, 10, 50, 50))]
    out = draw_detections(img.copy(), dets)
    assert out.sum() > 0


@pytest.mark.slow
def test_yolo26_loads_and_runs():
    from perception.detector import Detector
    det = Detector(classes=["cup", "wrench", "screwdriver"], conf=0.25)
    # "cup" is in COCO; wrench/screwdriver are not -> only cup is kept
    assert det.wanted_ids, "must recognize the cup class in the vocab"
    img = np.full((480, 640, 3), 128, np.uint8)
    out = det.detect(img)
    assert isinstance(out, list)
