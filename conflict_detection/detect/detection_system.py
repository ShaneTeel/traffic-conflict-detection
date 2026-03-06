from numpy.typing import NDArray

from conflict_detection.visualize import StudioManager
from conflict_detection.geometry import *
from conflict_detection.multi_object_tracking import ObjectDetector, ObjectTracker
from conflict_detection.trajectory import TrajManager
from .time_to_collision import TimeToCollision
from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class DetectionSystem:
    '''
    Description
    -----------
    Pipeline for traffic conflict detection that combines multi-object tracking (YOLOv8 + Supervision) 
    with Inverse Perspective Mapping to project pixel-space conflict coordinates to real-world geographic space

    Parameters
    ----------
    file_in : str
        The source file that will be used to initialize a `cv2.videoCapture()` object

    file_out : str
        The file path for the .mp4 file that will be the output of the multi-object tracking event

    dst_pts : NDArray
        The real-world geographic coordinates, in Lat/Lon

    src_pts : NDArray
        The pixel coordinates that correspond to the real-world coordinates (arg passed for `dst_pts`). 
        The points must also be in the same order as the points passed to `dst_pts`.

    model_path : str, default = "./models/yolov8m.pt"
        Specifies the both the file path and the actual YOLOv8 model-size for detection.
        Reference [ultralytics](https://docs.ultralytics.com/usage/cfg/#modes).

    model_conf : float, default = 0.5
        Sets the minimum confidence threshold for detections. 
        Objects detected with confidence below this threshold will be disregarded. 
        Adjusting this value can help reduce false positives.
        Reference [ultralytics](https://docs.ultralytics.com/usage/cfg/#modes).
    
    activation_thresh : float, default = 0.25
        Detection confidence threshold for track activation. 
        Increasing track_activation_threshold improves accuracy and stability but might miss true detections. 
        Decreasing it increases completeness but risks introducing noise and instability.
        Reference [Supervision](https://supervision.roboflow.com/develop/trackers/#supervision.tracker.byte_tracker.core.ByteTrack).

    lost_buffer : int, default = 30
        Number of frames to buffer when a track is lost. 
        Increasing lost_track_buffer enhances occlusion handling, 
        significantly reducing the likelihood of track fragmentation or disappearance caused by brief detection gaps.
        Reference [Supervision](https://supervision.roboflow.com/develop/trackers/#supervision.tracker.byte_tracker.core.ByteTrack).
            
    min_dist : float, default = 0.5
        Threshold value used to determine if the distance between two objects at time 't' is considered a conflict or not.
        Value is in pixel-space. 

    Public Methods
    --------------
    `.monitor_traffic()`, `.inspect_conflicts()`    
    '''

    def __init__(self, file_in:str, file_out:str, dst_pts:NDArray, src_pts:NDArray, model_path:str="./models/yolov8m.pt", model_conf:float=0.5, activation_thresh:float=0.25, lost_buffer:int=30, min_dist:float=0.5):
        '''
        Description
        -----------
        
        Parameters
        ----------
        file_in : str
            The source file that will be used to initialize a `cv2.videoCapture()` object

        file_out : str
            The file path for the .mp4 file that will be the output of the multi-object tracking event

        dst_pts : NDArray
            The real-world geographic coordinates, in Lat/Lon

        src_pts : NDArray
            The pixel coordinates that correspond to the real-world coordinates (arg passed for `dst_pts`). 
            The points must also be in the same order as the points passed to `dst_pts`.

        model_path : str, default = "./models/yolov8m.pt"
            Specifies both the file path and the actual YOLOv8 model-size for detection.
            Reference [ultralytics](https://docs.ultralytics.com/usage/cfg/#modes).

        model_conf : float, default = 0.5
            Sets the minimum confidence threshold for detections. 
            Objects detected with confidence below this threshold will be disregarded. 
            Adjusting this value can help reduce false positives.
            Reference [ultralytics](https://docs.ultralytics.com/usage/cfg/#modes).
        
        activation_thresh : float, default = 0.25
            Detection confidence threshold for track activation. 
            Increasing track_activation_threshold improves accuracy and stability but might miss true detections. 
            Decreasing it increases completeness but risks introducing noise and instability.
            Reference [Supervision](https://supervision.roboflow.com/develop/trackers/#supervision.tracker.byte_tracker.core.ByteTrack).

        lost_buffer : int, default = 30
            Number of frames to buffer when a track is lost. 
            Increasing lost_track_buffer enhances occlusion handling, 
            significantly reducing the likelihood of track fragmentation or disappearance caused by brief detection gaps.
            Reference [Supervision](https://supervision.roboflow.com/develop/trackers/#supervision.tracker.byte_tracker.core.ByteTrack).
                
        min_dist : float, default = 0.5
            Threshold value used to determine if the distance between two objects at time 't' is considered a conflict or not.
            Value is in pixel-space. 
        '''
        self.studio_in = StudioManager(file_in)
        self.file_out = file_out
        self.fps, frame = self.studio_in._extract_init_data(file_out)
        self.studio_in.set_frame_idx(0)
        self.temp_studio = None

        self.detector = ObjectDetector(model_path=model_path, confidence=model_conf)
        self.tracker = ObjectTracker(fps=self.fps, activation_thresh=activation_thresh, lost_buffer=lost_buffer)
        self.mapper = IPM(frame, dst_pts, src_pts)
        self.traj = TrajManager(self.fps, use_wall_time=False)
        self.ttc = TimeToCollision(min_dist)
        self.conflicts = None

        logger.debug("Detection system initialized")
    
    def monitor_traffic(self):
        '''
        Description
        -----------
        Performs multi-object tracking, trajectory collection & analysis, and time-to-collision calculation for every tracked object detected.

        Returns
        -------
        conflicts : dict[dict]
            A nested dictionary representing the minimum time-to-collision for each object pair that had a recorded conflict
        '''
        logger.info(f"Processing traffic cam footage.")

        analyzers = self._multi_object_tracking_with_traj_collection()
        logger.info(f"Collected {len(self.traj.collector)} unique tracks.")

        return self._detect_conflicts(analyzers=analyzers)

    def _multi_object_tracking_with_traj_collection(self):
        frames_count = 0
        
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
        '''
        Description
        -----------
        Projects the pixel-space coordinates for each vehicle conflicts to real-world geography coordinates (Lat/Lon).
        The coordinates are then used to create a `folium.Map()` object with a `.HeatMap()` and `.CircleMaker()` 
        layer (for each conflict) added as feature groups.

        Parameters
        ----------
        conflicts : dict[dict]
            The output of `DetectionSystem.monitor_traffic()`
        
        file_out : str
            The file path that the user intends the `.html` map object to be saved at 
        
        Returns
        -------
        basemap : folium.Map()
            The folium map containing the HeatMap and the CircleMarkers for each conflict
        '''
        if conflicts is None:
            raise RuntimeError("Error. User must call `.monitor_traffic()` first before inspecting conflicts.")   
        popups = []
        coords = []
        for k, c in conflicts.items():
            coords.append(self.mapper.map_perspective(c["collision_point"]))
            popups.append(f"""
<b><u>Object Pair</b></u>: {k}<br>
<b><u>Time-to-Collision</b></u>: {c["min_ttc"]:.2f}<br>
<b><u>Timedelta (from start)</b></u>: {c["time_of_ttc"]:.2f}<br>
""")
        self.mapper.add_layer(coords, popups, "blue", "Conflict Events (Markers)")
        return self.mapper.generate_heatmap(coords, file_out, name="Conflict Events (HeatMap)")