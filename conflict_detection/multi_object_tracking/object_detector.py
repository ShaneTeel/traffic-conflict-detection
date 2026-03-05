from ultralytics import YOLO
from numpy.typing import NDArray

from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class ObjectDetector:

    _VIC_CLASSES = [2, 3, 5, 7]

    def __init__(self, model_path:str="yolov8m.pt", confidence:float=0.5):

        self.model = YOLO(model=model_path, verbose=False)
        self.confidence = confidence

        logger.debug("Initialied detector.")

    def detect(self, frame:NDArray):
        results = self.model(frame, conf=self.confidence, verbose=False)

        results_lst = []

        if len(results[0].boxes) == 0:
            logger.debug("No objects detected in frame.")
        else:
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                if class_id not in self._VIC_CLASSES:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                class_name = results[0].names[class_id]
                box_dict = {
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "class_id": class_id,
                    "class_name": class_name
                }
                results_lst.append(box_dict)
                    
        return results_lst