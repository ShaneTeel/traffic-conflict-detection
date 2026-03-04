    
import cv2
import numpy as np

from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class Illustrator:
    '''Superimposes shapes/lines on an image'''

    def __init__(self, object_color:tuple = (0, 255, 0), conflict_color:tuple = (0, 255, 0)):
        
        self.object_color = self._hex_to_bgr(object_color)
        self.conflict_color = self._hex_to_bgr(conflict_color)

    def draw_boxes(self, frame:np.ndarray, pt1:tuple, pt2:tuple, class_name:str, conf:float, track_id:int):
        frame = self._channel_checker(frame)
        cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=self.object_color, thickness=2, lineType=cv2.LINE_AA)
        if track_id is not None:
            label = f"Class: {class_name}, Confidence: {conf:.2f}, Track: {track_id}"
        else:
            label = f"Class: {class_name}, Confidence: {conf:.2f}, Track: None"
        cv2.putText(img=frame, text=label, org=(pt1[0], pt1[1]-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        return frame
    
    def draw_marks(self, frame:np.ndarray, center_pts:tuple, label:str):
        frame = self._channel_checker(frame)
        cv2.drawMarker(frame, center_pts, markerType=cv2.MARKER_CROSS, thickness=5, color=self.conflict_color)

        cv2.putText(img=frame, text=label, org=(center_pts[0], center_pts[1]-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        return frame

    def _hex_to_bgr(self, color):
        if isinstance(color, tuple) and len(color) == 3:
            if len(color) == 3:
                return color
            else:
                return color[:3]
        
        if color.startswith("#"):
            hex_color = color[1:7]

            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)

            return (b, g, r)

    def _channel_checker(self, frame):
        if len(frame.shape) < 3:
            frame = cv2.merge([frame, frame, frame])
            return frame
        else:
            return frame