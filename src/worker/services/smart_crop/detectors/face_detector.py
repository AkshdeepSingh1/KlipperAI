import os
from typing import List, Dict, Any
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector as MPFaceDetector, FaceDetectorOptions, RunningMode
from mediapipe import Image, ImageFormat

from src.worker.services.smart_crop.detectors.base_detector import BaseDetector
from src.shared.core.logger import get_logger

logger = get_logger(__name__)

class FaceDetector(BaseDetector):
    """
    Face detector using MediaPipe.
    """

    def __init__(self, model_path: str = None, confidence: float = 0.3):
        if model_path is None:
            # Update default path to new structure
            model_path = os.path.join(
                os.path.dirname(__file__), 
                "../../models/face_detection/face_detector.tflite"
            )
        
        self.model_path = model_path
        self.confidence = confidence
        self.detector = None
        self._initialize_detector()

    def _initialize_detector(self):
        if not os.path.exists(self.model_path):
            logger.error(f"Face detection model not found at {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=self.confidence,
        )
        self.detector = MPFaceDetector.create_from_options(options)
        logger.info(f"FaceDetector initialized with model: {self.model_path}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.detector is None:
            logger.warning("FaceDetector not initialized.")
            return []

        try:
            mp_image = Image(image_format=ImageFormat.SRGB, data=frame)
            result = self.detector.detect(mp_image)
            
            detections = []
            if result.detections:
                for detection in result.detections:
                    # MediaPipe returns bounding box in pixels
                    bb = detection.bounding_box
                    x1 = int(bb.origin_x)
                    y1 = int(bb.origin_y)
                    x2 = int(bb.origin_x + bb.width)
                    y2 = int(bb.origin_y + bb.height)

                    detections.append({
                        "type": "face",
                        "bbox": [x1, y1, x2, y2],
                        "weight": 5.0
                    })
            return detections

        except Exception as e:
            logger.error(f"Error during face detection: {str(e)}")
            return []

    def close(self):
        if self.detector:
            self.detector.close()
