    
import cv2
import numpy as np

from typing import List

from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class Illustrator:
    '''Superimposes shapes/lines on an image'''

    def __init__(self, stroke_color:tuple = (0, 0, 0), fill_color:tuple = (0, 255, 0)):
        
        self.stroke_color = self._hex_to_bgr(stroke_color)
        self.fill_color = self._hex_to_bgr(fill_color)

    def draw_boxes(self, frame:np.ndarray, pt1:tuple, pt2:tuple, class_name:str, conf:float, track_id:int):
        frame = self._channel_checker(frame)
        cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=self.stroke_color, thickness=2, lineType=cv2.LINE_AA)
        if track_id is not None:
            label = f"Class:{class_name}, Confidence: {conf:.2f}, Track: {track_id}"
        else:
            label = f"Class:{class_name}, Confidence: {conf:.2f}, Track: None"
        cv2.putText(img=frame, text=label, org=(pt1[0], pt1[1]-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=self.stroke_color, thickness=2, lineType=cv2.LINE_AA)
        return frame
    
    def _draw_banner_text(self, frame, text):
        frame = self._channel_checker(frame)
        h, w = frame.shape[:2]
        banner_height = int(0.08 * h)
        cv2.rectangle(frame, (0, 0), (w, banner_height), (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

        cv2.putText(frame, text, (int(w // 2) - 80, 10 + (banner_height // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        return frame
    
    def draw_circles(self, frame:np.ndarray, center_pts:tuple):
        frame = self._channel_checker(frame)
        cv2.drawMarker(frame, center_pts, markerType=cv2.MARKER_CROSS, thickness=2, color=(0, 0, 255))

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