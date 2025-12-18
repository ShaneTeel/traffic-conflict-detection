import numpy as np
from ultralytics import YOLO

from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class TargetDetector:

    def __init__(self, model_path:str="yolov8n.pt", confidence:float=0.5):

        self.model = YOLO(model=model_path, verbose=False)
        self.confidence = confidence

        logger.debug("Initialied detector.")

    def detect(self, frame:np.ndarray):
        results = self.model(frame, conf=self.confidence, verbose=False)

        results_lst = []

        if len(results[0].boxes) == 0:
            logger.debug("No objects detected in frame.")
        else:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                class_id = box.cls[0].item()
                class_name = results[0].names[class_id]
                box_dict = {
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "class_id": class_id,
                    "class_name": class_name
                }
                results_lst.append(box_dict)
                    
        return results_lst