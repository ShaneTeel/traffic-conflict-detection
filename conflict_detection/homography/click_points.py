import cv2
import numpy as np

from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class ClickPoints:
 
    def __init__(self, frame, window_name):
        self.original = frame
        self.canvas = self.original.copy()
        self.blank = np.zeros_like(self.canvas[:, :, 0])        
        self.window_name = window_name
        self.pts = []
        self.show()
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        logger.debug("Click points successfully initialized.")

    def show(self):
        cv2.imshow(self.window_name, self.canvas)

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.debug(f"User clicked image at pixel coordinates: {pt}")
            self.pts.append(pt)

        if self.pts and flags & cv2.EVENT_FLAG_LBUTTON:
            for img, color in zip([self.canvas, self.blank], ((0, 255, 0), 255)):
                cv2.drawMarker(img, pt, color, cv2.MARKER_CROSS)
            self.show()

    def draw(self):
        while True:
            key = cv2.waitKey()

            if key == 27:
                cv2.destroyWindow(self.window_name)
                break
            ret = cv2.inpaint(self.canvas, self.blank, 3, flags=cv2.INPAINT_TELEA)
            cv2.imshow(self.window_name, ret)
            if key == ord('r'):
                logger.debug("User reset image. Clearing pts list and image.")
                self.canvas = self.original.copy()
                self.blank = np.zeros_like(self.canvas[:, :, 0])
                self.show()

    def get_pts(self):
        logger
        return self.pts