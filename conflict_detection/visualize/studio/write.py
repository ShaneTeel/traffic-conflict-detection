import cv2
from .read import Reader

from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class Writer():
    def __init__(self, source:Reader):
        self.width = source.width
        self.height = source.height
        self.fps = source.fps
        self.writer = None

    def write_frame(self, frame):
        self.writer.write(frame)

    def _initialize_writer(self, file_out_name, fourcc:str):
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(file_out_name, fourcc, self.fps, (self.width, self.height))
        logger.debug("Successfully initialized writer object")