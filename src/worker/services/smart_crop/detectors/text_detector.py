from typing import List, Dict, Any, Literal
import numpy as np
import easyocr
import cv2
import os

from src.worker.services.smart_crop.detectors.base_detector import BaseDetector
from src.shared.core.logger import get_logger

logger = get_logger(__name__)

# Paths
MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
EASYOCR_MODELS_DIR = os.path.join(MODELS_ROOT, "text_detection/easyocr")
EAST_MODEL_PATH = os.path.join(MODELS_ROOT, "text_detection/east/frozen_east_text_detection.pb")

class TextDetector(BaseDetector):
    """
    Text detector supporting multiple backends:
    1. 'easyocr' - Accurate but heavy (uses PyTorch).
    2. 'east' - OpenCV EAST text detector (faster, detection only).
    """

    def __init__(
        self, 
        backend: Literal["easyocr", "east"] = "east", 
        languages: List[str] = None, 
        confidence: float = 0.5, 
        use_gpu: bool = True
    ):
        self.backend = backend
        self.confidence = confidence
        self.languages = languages if languages else ['en']
        self.use_gpu = use_gpu
        self.reader = None
        self.east_net = None
        
        logger.info(f"Initializing TextDetector with backend: {self.backend}")

        if self.backend == "easyocr":
            self._init_easyocr()
        elif self.backend == "east":
            self._init_east()
        else:
            logger.warning(f"Unknown backend '{backend}', falling back to EAST")
            self._init_east()

    def _init_easyocr(self):
        """Initialize EasyOCR reader."""
        # Ensure directory exists
        if not os.path.exists(EASYOCR_MODELS_DIR):
             os.makedirs(EASYOCR_MODELS_DIR, exist_ok=True)
             
        try:
            self.reader = easyocr.Reader(
                self.languages, 
                gpu=self.use_gpu, 
                verbose=False,
                model_storage_directory=EASYOCR_MODELS_DIR,
                download_enabled=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            # Fallback to default (allow download if needed)
            self.reader = easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)

    def _init_east(self):
        """Initialize OpenCV EAST detector."""
        if not os.path.exists(EAST_MODEL_PATH):
            logger.error(f"EAST model not found at {EAST_MODEL_PATH}")
            return

        try:
            self.east_net = cv2.dnn.readNet(EAST_MODEL_PATH)
            logger.info("OpenCV EAST text detector loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load EAST model: {e}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.backend == "easyocr":
            return self._detect_easyocr(frame)
        elif self.backend == "east":
            return self._detect_east(frame)
        return []

    def _detect_easyocr(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        try:
            results = self.reader.readtext(frame)
            detections = []
            for bbox_points, text, prob in results:
                if prob < self.confidence:
                    continue

                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))

                detections.append({
                    "type": "text",
                    "bbox": [x1, y1, x2, y2],
                    "weight": 4.0
                })
            return detections
        except Exception as e:
            logger.error(f"EasyOCR detection failed: {e}")
            return []

    def _detect_east(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text using OpenCV EAST.
        Requires resizing image to multiple of 32.
        """
        if self.east_net is None:
            return []
            
        try:
            # EAST requires dimensions to be multiples of 32
            orig_h, orig_w = frame.shape[:2]
            new_w, new_h = (320, 320) # Fixed size for inference speed
            
            r_w = orig_w / float(new_w)
            r_h = orig_h / float(new_h)
            
            blob = cv2.dnn.blobFromImage(
                frame, 1.0, (new_w, new_h), 
                (123.68, 116.78, 103.94), swapRB=True, crop=False
            )
            
            self.east_net.setInput(blob)
            
            # Output layers: 'feature_fusion/Conv_7/Sigmoid' (scores), 'feature_fusion/concat_3' (geometry)
            layer_names = [
                "feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"
            ]
            scores, geometry = self.east_net.forward(layer_names)
            
            # Decode results
            return self._decode_east(scores, geometry, r_w, r_h)
            
        except Exception as e:
            logger.error(f"EAST detection failed: {e}")
            return []

    def _decode_east(self, scores, geometry, r_w, r_h):
        """
        Decode EAST output to bounding boxes.
        """
        detections = []
        num_rows, num_cols = scores.shape[2:4]
        
        boxes = []
        confidences = []
        
        for y in range(num_rows):
            scores_data = scores[0, 0, y]
            x0_data = geometry[0, 0, y]
            x1_data = geometry[0, 1, y]
            x2_data = geometry[0, 2, y]
            x3_data = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]
            
            for x in range(num_cols):
                score = scores_data[x]
                if score < self.confidence:
                    continue
                
                # Calculate offset
                offset_x, offset_y = x * 4.0, y * 4.0
                
                # Extract rotation angle
                angle = angles_data[x]
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]
                
                end_x = int(offset_x + (cos_a * x1_data[x]) + (sin_a * x2_data[x]))
                end_y = int(offset_y - (sin_a * x1_data[x]) + (cos_a * x2_data[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)
                
                boxes.append([start_x, start_y, end_x, end_y])
                confidences.append(float(score))

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(
            [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes], # Convert to [x, y, w, h] for NMS
            confidences, 
            self.confidence, 
            0.4
        )
        
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                start_x, start_y, end_x, end_y = box
                
                # Scale back to original image size
                x1 = int(start_x * r_w)
                y1 = int(start_y * r_h)
                x2 = int(end_x * r_w)
                y2 = int(end_y * r_h)
                
                detections.append({
                    "type": "text",
                    "bbox": [x1, y1, x2, y2],
                    "weight": 4.0
                })
                
        return detections
