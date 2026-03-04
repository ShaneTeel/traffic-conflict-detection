import numpy as np
import cv2
import os
import tempfile
import atexit
import collections

from conflict_detection.visualization import StudioManager
from conflict_detection.geometry import *
from conflict_detection.multi_object_tracking import ObjectDetector, ObjectTracker
from conflict_detection.trajectory import TrajManager
from .time_to_collision import TimeToCollision
from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class DetectionSystem:

    _OBJECT_INFO = collections.namedtuple("ObjectInfo", ["coords", "label", "lifecycle"])

    def __init__(self, file_in:str, world_pts:np.ndarray, model_path:str="./models/yolov8n.pt", model_conf:float=0.5, activation_thresh:float=0.25, lost_buffer:int=30, ttc_thresh:float=1.5, min_dist:float=0.5, use_wall_time:bool=False):

        self.temp_file = self._create_temp_file()
        atexit.register(self._clean_up)

        self.studio_in = StudioManager(file_in)
        self.fps, frame = self.studio_in._extract_init_data(self.temp_file)
        self.temp_studio = None

        self.detector = ObjectDetector(model_path=model_path, confidence=model_conf)
        self.tracker = ObjectTracker(fps=self.fps, activation_thresh=activation_thresh, lost_buffer=lost_buffer)
        self.mapper = IPM(frame, world_pts)
        self.traj = TrajManager(self.fps, use_wall_time=False)
        self.ttc = TimeToCollision(ttc_thresh, min_dist)
        self.conflicts = None

        logger.debug("Detection system initialized")

    def _create_temp_file(self, suffix:str=".mp4"):
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        logger.info(f"Temp file created at {temp_path}")
        return temp_path
    
    def monitor_traffic(self, file_out:str=None):
        logger.info(f"Processing traffic cam footage.")

        analyzers = self._multi_object_tracking_with_traj_collection()
        logger.info(f"Collected {len(self.traj.collector)} unique tracks.")

        self.conflicts = self._detect_conflicts(analyzers=analyzers)
        self._annotate_conflicts(conflicts=self.conflicts, file_in=self.temp_file, file_out=file_out)

        return self.conflicts

    def _multi_object_tracking_with_traj_collection(self):
        frames_count = 0
        self.studio_in.set_frame_idx(0)
        
        while True:
            ret, frame = self.studio_in.return_frame()
            if not ret:
                logger.info(f"Finished processing {frames_count} frames.")
                if self.studio_in.writer_check():
                    logger.info(f"Output saved to: {self.temp_file}")
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
    
    def _annotate_conflicts(self, conflicts:dict[dict], file_in:str, file_out:str):
        if conflicts is None:
            raise RuntimeError("Error. User must call `_detect_conflicts()` first before annotating")

        self.temp_studio = StudioManager(file_in)
        self.temp_studio.create_writer(file_out, fourcc="mp4v")

        logger.info("Starting conflict annotation.")
        self.temp_studio.set_frame_idx(0)
        frames_count = 0
        active_objs = []

        while True:
            ret, frame = self.temp_studio.return_frame()
            if not ret:
                logger.info(f"Finished annotating {frames_count} frames.")
                if self.temp_studio.writer_check():
                    logger.info(f"Output saved to: {file_out}")
                    self.temp_studio.release_writer()
                break
            
            frames_count += 1
            frame_conflicts = {t: inner for t, inner in conflicts.items() if frames_count == inner["frame_idx"]}
            
            if frame_conflicts:
                for _, c in frame_conflicts.items():
                    coords = (int(c["collision_point"][0]), int(c["collision_point"][1]))
                    label = f"TTC: {c['min_ttc']}, Min Distance: {c['min_distance']}"
                    active_objs.append(self._OBJECT_INFO(coords, label, self.fps * 5))
                
            updated_objects = []
            for markers in active_objs:
                self.temp_studio.draw_conflicts(frame, markers.coords, label=markers.label)

                if markers.lifecycle > 1:
                    updated_objects.append(self._OBJECT_INFO(markers.coords, markers.label, markers.lifecycle - 1))

            active_objs = updated_objects

            self.temp_studio.write_frame(frame)

    def inspect_conflicts(self, conflicts:dict[dict]):
        if conflicts is None:
            raise RuntimeError("Error. User must call `_detect_conflicts()` first before annotating")   
           
        coords = []
        popups = []
        for k, c in conflicts.items():
            pts_arr = np.array(c["collision_point"], np.float32)
            coords.append(self.mapper.map_persepective(pts_arr))
            popup_info = f"""
<b><u>Object Pair</b></u>: {k}<br>
<b><u>Min. TTC</b></u>: {c["min_ttc"]}<br>
<b><u>Min. Distance</b></u>: {c["min_distance"]}<br>
<b><u>TTC Time</b></u>: {c["time_of_ttc"]}<br>
"""
            popups.append(popup_info)

        self.mapper.add_layer(coords, popups, name="Min TTC Points")
        return self.mapper.show_map()

    def _clean_up(self):

        if self.studio_in is not None:
            self.studio_in.release_all_resources()
        if self.temp_studio is not None:
            self.temp_studio.release_all_resources()

        if hasattr(self, "temp_file") and os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
                logger.info(f"Removed {self.temp_file}")
            except Exception as e:
                logger.error(e)