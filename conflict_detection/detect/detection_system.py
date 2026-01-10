import numpy as np

from typing import Union
from numpy.typing import NDArray

from conflict_detection.studio import StudioManager
from conflict_detection.homography import ClickPoints, WorldProjector
from conflict_detection.objects import ObjectDetector, ObjectTracker
from conflict_detection.trajectory import TrajManager
from conflict_detection.safety import TimeToCollision
from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class DetectionSystem:

    def __init__(self, file_in:Union[str, int], world_pts:NDArray, model_path:str="./models/yolov8n.pt", model_conf:float=0.5, activation_thresh:float=0.25, lost_buffer:int=30, ttc_thresh:float=1.5, min_dist:float=0.5, use_wall_time:bool=False):

        self.studio = StudioManager(file_in)
        self.fps, _, _ = self.studio.get_metadata()
        self.detector = ObjectDetector(model_path=model_path, confidence=model_conf)
        self.tracker = ObjectTracker(fps=self.fps, activation_thresh=activation_thresh, lost_buffer=lost_buffer)
        self.projector = self._initialize_projector(world_pts)
        self.traj = TrajManager(self.projector, self.fps, use_wall_time=False)
        self.ttc = TimeToCollision(ttc_thresh, min_dist)
        
    def _initialize_projector(self, world_pts:NDArray):
        _, frame = self.studio.return_frame()

        click = ClickPoints(frame, "Image Space")
        click.draw()

        img_pts = np.array(click.get_pts(), dtype=np.float32)
        return  WorldProjector(img_pts, world_pts)
    
    def monitor_traffic(self, file_out:str=None):
        if file_out is not None:
            self.studio.create_writer(file_out, fourcc="mp4v")

        logger.info("Starting video processing.")

        frames_count = 0
        self.studio.set_frame_idx(0)

        while True:
            ret, frame = self.studio.return_frame()
            if not ret:
                logger.info(f"Finished processing {frames_count} frames.")
                if self.studio.writer_check():
                    logger.info(f"Output saved to: {file_out}")
                    self.studio.release_writer()
                break
            
            frames_count += 1
            if frames_count % 25 == 0:
                logger.info(f"Processing frame {frames_count}")

            results = self.detector.detect(frame)
            tracks = self.tracker.track(results)
            self.traj.collect_tracks(tracks)

            if self.studio.writer_check():
                self.studio.draw_tracked_objects(frame, tracks)
                self.studio.write_frame(frame)

            flag = self.studio.control_playback()
            if flag:
                logger.info(f"Finished processing {frames_count} frames.")
                if self.studio.writer_check():
                    logger.info(f"Output saved to: {file_out}")
                    self.studio.release_writer()
                break
    
        logger.info(f"Collected {len(self.traj.collector)} unique tracks.")
        self.traj.analyze_tracks()
    
    def detect_conflicts(self):
        all_analyzers = self.traj.get_analyzer()
        self.ttc.analyze_all_conflicts(all_analyzers)
        min_ttc = self.ttc.get_all_minimum_ttc()
        logger.info(f"Detected {len(min_ttc)}")
        return min_ttc