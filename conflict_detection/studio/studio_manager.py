# Type hints
from typing import Union, List
from numpy.typing import NDArray

# Sub-package imports
from .read import Reader
from .write import Writer
from .illustrate import Illustrator
from .render import Render
from .custodian import Custodian
from .control import Controller

# Package import
from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class StudioManager():
    
    def __init__(self, source:Union[str, int]):

        self.source = Reader(source)
        self.write = Writer(self.source)
        self.draw = Illustrator(stroke_color=(0, 0, 255))
        self.render = Render()
        self.playback = Controller(self.source)
        self.clean = Custodian(self.source, self.write)
        self.exit = False

        logger.debug("Initialized studio.")

    def return_frame(self):
        if self.source.source_type == 'image':
            return True, self.source.image
        
        if self.source.cap is None:
            return False, None

        ret, frame = self.source.cap.read()
        
        if ret:
            return True, frame
        else:
            return False, None
        
    def get_metadata(self):
        '''Returns fps, height, and width of media object'''
        if self.source.fps is None:
            return 1, self.source.height, self.source.width
        return self.source.fps, self.source.height, self.source.width
    
    def source_type(self):
        return self.source.source_type
    
    def get_name(self):
        return self.source.name
    
    def create_writer(self, file_out_name:str, fourcc:str):
        self.write._initialize_writer(file_out_name, fourcc)

    def write_frame(self, frame:NDArray):
        if self.write.writer is not None:
            self.write.write_frame(frame)
        else:
            raise RuntimeError("ERROR: Never created writer object")
    
    def writer_check(self):
        return True if self.write.writer is not None else False
    
    def print_menu(self):
        self.playback.print_playback_menu()
    
    def control_playback(self):
        return self.playback.playback_controls()
    
    def draw_tracked_objects(self, frame:NDArray, tracks:List[dict]):
        if len(tracks) != 0:
            for track in tracks:
                x1, y1, x2, y2 = map(int, track["bbox"])
                class_name = track["class_name"]
                conf = track["conf"]
                track_id = track["track_id"]
                frame = self.draw.draw_boxes(frame, (x1, y1), (x2, y2), class_name, conf, track_id)

    def release_all_resources(self):
        self.clean._clean_up()

    def release_writer(self):
        self.write.writer.release()

    def set_frame_idx(self, idx:int):
        self.source.set_frame_idx(idx)

    def draw_src_pts(self, frame:NDArray, coords:List[tuple]):
        for (x, y) in coords:
            self.draw.draw_circles(frame, (x, y))