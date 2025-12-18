import numpy as np
import supervision as sv
from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class TargetTracker:

    def __init__(self, activation_thresh:float=0.25, lost_buffer:int=30, fps:int=30):
        """
        Initialize ByteTrack tracker
        
        Parameters
        ----------
        fps : int
            Video FPS (for motion prediction)
        activation_thresh : float
            Minimum confidence to start tracking
        lost_buffer : int
            Frames to keep lost tracks alive
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=activation_thresh, 
            lost_track_buffer=lost_buffer, 
            frame_rate=fps
        )

        logger.debug(f"Initialied tracker (fps={fps}).")

    def track(self, detections:list):
        """
        Update tracker with new detections
        
        Parameters
        ----------
        detections : list of dicts
            Detection dicts from Detector.detect()
        
        Returns
        -------
        tracks : list of dict
            Detections with track_id added
        """
        sv_detections = self._detections_to_sv_detections(detections=detections)

        tracked = self.tracker.update_with_detections(sv_detections)
        
        return self._sv_detections_to_dict(tracked, detections)

    def _detections_to_sv_detections(self, detections:list):
        '''converts detection dict (output of Detector.detect()) to supervision format'''
        n_dims = len(detections)
        
        if n_dims == 0:
            logger.debug("Detections list contains no detections.")
            return sv.Detections.empty()
        
        xyxy = np.zeros((n_dims, 4), dtype=np.float32)
        conf = np.zeros(n_dims, dtype=np.float32)
        class_id = np.zeros(n_dims, dtype=np.int32)

        for i, det in enumerate(detections):
            xyxy[i] = det["bbox"]
            conf[i] = det["conf"]
            class_id[i] = det["class_id"]

        return sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=class_id
        )

    def _sv_detections_to_dict(self, sv_detections:sv.Detections, original_detections:list):
        '''Convert supervision detections back to dict format with 'track_id' added'''
        if len(sv_detections) == 0:
            logger.debug("No tracked found. Returning empty list.")
            return []
        
        tracks = []
        for i in range(len(sv_detections)):
            track_id = int(sv_detections.tracker_id[i]) if sv_detections.tracker_id[i] is not None else None
            x1, y1, x2, y2 = sv_detections.xyxy[i]

            track_dict = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(sv_detections.confidence[i]),
                "class_id": int(sv_detections.class_id[i]),
                "class_name": original_detections[i]["class_name"],
                "track_id": track_id
            }
            tracks.append(track_dict)
        logger.debug(f"Tracked {len(tracks)} objects.")
        return tracks