import numpy as np
import cv2
import os

from typing import Union
from numpy.typing import NDArray

from conflict_detection.studio import StudioManager
from conflict_detection.space import ClickPoints, WorldProjector
from conflict_detection.objects import ObjectDetector, ObjectTracker
from conflict_detection.trajectory import TrajManager
from .time_to_collision import TimeToCollision
from conflict_detection.utils import get_logger, MinMaxScaler

logger = get_logger(__name__)

class DetectionSystem:

    def __init__(self, file_in:Union[str, int], world_pts:NDArray, model_path:str="./models/yolov8n.pt", model_conf:float=0.5, activation_thresh:float=0.25, lost_buffer:int=30, ttc_thresh:float=1.5, min_dist:float=0.5, use_wall_time:bool=False):

        self.studio = StudioManager(file_in)
        self.fps, _, _ = self.studio.get_metadata()
        self.detector = ObjectDetector(model_path=model_path, confidence=model_conf)
        self.tracker = ObjectTracker(fps=self.fps, activation_thresh=activation_thresh, lost_buffer=lost_buffer)
        self.projector = self._initialize_projector(world_pts)
        self.traj = TrajManager(self.fps, use_wall_time=False)
        self.ttc = TimeToCollision(ttc_thresh, min_dist)

        logger.debug("Detection system initialized")
        
    def _initialize_projector(self, world_pts:NDArray):
        _, frame = self.studio.return_frame()

        logger.info("Select four points that correspond to the four real-world points provided as the argument for `world_pts`. Press 'ESC' when complete")

        click = ClickPoints(frame, "Image Space")
        click.draw()

        img_pts = np.array(click.get_pts(), dtype=np.float32)

        logger.debug(f"User selected the following img_pts = \n{img_pts}")

        return WorldProjector(img_pts, world_pts)
    
    def monitor_traffic(self, file_out:str=None, view:bool=True):
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
    
        logger.info(f"Collected {len(self.traj.collector)} unique tracks.")

        if view:
            self._view_tracked_objects(file_out)

        self.conflicts = self._detect_conflicts()
        return self.conflicts 
    
    def _view_tracked_objects(self, file_out:str):
        if os.path.exists(file_out):
            logger.info("Playing back processed video...")
            studio = StudioManager(file_out)
            studio.print_menu()

            while True:
                ret, frame = studio.return_frame()
                if not ret:
                    break

                cv2.imshow("Processed Video", frame)
                flag = studio.control_playback()
                if flag:
                    break
        else:
            logger.warning("Cannot find video file associated with `file_out`.")

    def _detect_conflicts(self):
        analyzers = self.traj.analyze_tracks()
        _ = self.ttc.analyze_all_conflicts(analyzers)
        conflicts = self.ttc.get_all_minimum_ttc()
        logger.info(f"Detected {len(conflicts)} conflicts (Time-to-Collision)")
        return conflicts
    

    def format_conflicts(self):
        coords = []
        popups = []
        for k, c in self.conflicts.items():
            pts_arr = np.array(c["collision_point"], np.float32)
            coords.append(self.projector.project(pts_arr, "forward"))
            popup_info = f"""
<b><u>Object Pair</b></u>: {k}<br>
<b><u>Min. TTC</b></u>: {c["min_ttc"]}<br>
<b><u>Min. Distance</b></u>: {c["min_distance"]}<br>
"""
            popups.append(popup_info)

        return coords, popups