import numpy as np
import collections

from numpy.typing import NDArray

from conflict_detection.visualize import StudioManager
from conflict_detection.geometry import *
from conflict_detection.multi_object_tracking import ObjectDetector, ObjectTracker
from conflict_detection.trajectory import TrajManager
from .time_to_collision import TimeToCollision
from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class DetectionSystem:

    _OBJECT_INFO = collections.namedtuple("ObjectInfo", ["coords", "label", "lifecycle"])

    def __init__(self, file_in:str, file_out:str, world_pts:NDArray, img_pts:NDArray, model_path:str="./models/yolov8n.pt", model_conf:float=0.5, activation_thresh:float=0.25, lost_buffer:int=30, ttc_thresh:float=1.5, min_dist:float=0.5, use_wall_time:bool=False):

        self.studio_in = StudioManager(file_in)
        self.file_out = file_out
        self.fps, frame = self.studio_in._extract_init_data(file_out)
        self.temp_studio = None

        self.detector = ObjectDetector(model_path=model_path, confidence=model_conf)
        self.tracker = ObjectTracker(fps=self.fps, activation_thresh=activation_thresh, lost_buffer=lost_buffer)
        self.mapper = IPM(frame, world_pts, img_pts)
        self.traj = TrajManager(self.fps, use_wall_time=False)
        self.ttc = TimeToCollision(ttc_thresh, min_dist)
        self.conflicts = None

        logger.debug("Detection system initialized")
    
    def monitor_traffic(self):
        logger.info(f"Processing traffic cam footage.")

        analyzers = self._multi_object_tracking_with_traj_collection()
        logger.info(f"Collected {len(self.traj.collector)} unique tracks.")

        return self._detect_conflicts(analyzers=analyzers)

    def _multi_object_tracking_with_traj_collection(self):
        frames_count = 0
        self.studio_in.set_frame_idx(0)
        
        while True:
            ret, frame = self.studio_in.return_frame()
            if not ret:
                logger.info(f"Finished processing {frames_count} frames.")
                if self.studio_in.writer_check():
                    logger.info(f"Output saved to: {self.file_out}")
                    self.studio_in.release_writer()
                break
            
            frames_count += 1
            if frames_count % 25 == 0:
                logger.info(f"Processing frame {frames_count}")

            results = self.detector.detect(frame)
            tracks = self.tracker.track(results)
            self.traj.collect_tracks(tracks)

            self.studio_in.draw_tracked_objects(frame, tracks)
            self.studio_in.write_frame(frame)

        return self.traj.analyze_tracks()

    def _detect_conflicts(self, analyzers:dict[dict]):
        _ = self.ttc.analyze_all_conflicts(analyzers)
        conflicts = self.ttc.get_all_minimum_ttc()
        logger.info(f"Detected {len(conflicts)} conflicts (Time-to-Collision)")
        return conflicts

    def inspect_conflicts(self, conflicts:dict[dict], file_out:str):
        if conflicts is None:
            raise RuntimeError("Error. User must call `_detect_conflicts()` first before annotating")   
           
        coords = []
        for c in conflicts.values():
            pts_arr = np.array(c["collision_point"], np.float32)
            pts_arr
            coords.append(self.mapper.map_persepective(pts_arr))

        return self.mapper.generate_heatmap(coords, file_out)