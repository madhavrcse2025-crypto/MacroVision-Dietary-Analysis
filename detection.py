"""
YOLOv8 inference wrapper for MacroVision.

Loads the fine-tuned detector (see training/train_yolov8.py) and returns a list of
detected food items with class label, confidence, and bounding-box area in pixels --
the format expected by estimation.estimate_meal().
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "macrovision_yolov8.pt"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.5


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixels, on the resized frame
    bbox_area: float  # pixels^2


class FoodDetector:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model = YOLO(str(model_path))

    @staticmethod
    def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
        """Bilateral filter + resize, matching the pipeline described in the report (5.1.1)."""
        denoised = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)
        resized = cv2.resize(denoised, (INPUT_SIZE, INPUT_SIZE))
        return resized

    def infer(self, image_bgr: np.ndarray) -> List[Detection]:
        frame = self._preprocess(image_bgr)
        results = self.model.predict(source=frame, imgsz=INPUT_SIZE,
                                      conf=CONF_THRESHOLD, verbose=False)
        detections: List[Detection] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = max(0.0, (x2 - x1) * (y2 - y1))
            label = results[0].names[int(box.cls[0])]
            detections.append(Detection(
                label=label,
                confidence=float(box.conf[0]),
                bbox=(x1, y1, x2, y2),
                bbox_area=area,
            ))
        return detections
