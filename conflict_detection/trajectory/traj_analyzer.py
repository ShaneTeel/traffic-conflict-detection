import numpy as np
from typing import List
from collections import Counter

from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class TrajAnalyzer:

    def __init__(self, track_id:int, positions:List[dict]):
        """
        Parameters
        ----------
        track_id : int
        positions : list of dict
            Each dict has: bbox, timestamp, frame_idx, class_name, conf
        """
        self.track_id = track_id
        self.positions = self._initialize_positions(positions)

        self._speed_cache = None
        self._path_length_cache = None
        self._segment_speeds = {}
        self._instant_positions = {}
        self._instant_velocity = {}
        
        logger.debug(f"Initialized trajectory analyzer for Track {self.track_id}.")

    def calculate_avg_speed(self):
        '''Get average speed a tracked object (cached operation)'''
        if self._speed_cache is None:
            if not self._sufficient_data():
                logger.warning(f"Track {self.track_id}: Need 2+ positions to compute path length.")
                return self._speed_cache
            
            self._speed_cache = self._compute_avg_speed()
        return self._speed_cache
    
    def calculate_instant_position(self, time):
        '''Get tracked object's position for a given time (cached operation).'''
        if time not in self._instant_positions.keys():
            if not self._sufficient_data():
                logger.warning(f"Track {self.track_id}: Need 2+ positions to compute instant position.")
                return None
            else:
                return self._compute_instant_position(time)
        else:
            return self._instant_positions.get(time)

    def calculate_segment_speed(self, time):
        '''Get tracked object's speed for a given time (cached operation).'''
        if time not in self._segment_speeds:
            if not self._sufficient_data():
                logger.warning(f"Track {self.track_id}: Need 2+ positions to compute instant speed.")
                return None
            else:
                return self._compute_segment_speed(time)
        else:
            return self._segment_speeds[time]
    
    def calculate_instant_velocity(self, time):
        '''Get tracked object's speed for a given time (cached operation).'''
        if time not in self._instant_velocity:
            if not self._sufficient_data():
                logger.warning(f"Track {self.track_id}: Need 2+ positions to compute instant speed.")
                return None
            else:
                return self._compute_instant_velocity(time)
        else:
            return self._instant_velocity[time]

    def calculate_path_length(self):
        '''Get total path length of a tracked object (cached operation).'''        
        if self._path_length_cache is None:
            if not self._sufficient_data():
                logger.warning(f"Track {self.track_id}: Need 2+ positions to compute path length.")
                return self._path_length_cache
            
            centers = self.get_centers()
            self._path_length_cache = self._compute_path_length(centers)
        return self._path_length_cache

    def get_stable_class(self):
        '''Return most common class across trajectory'''
        if len(self.positions) == 0:
            return None
        
        classes = self._get_value("class_name")
        return Counter(classes).most_common(1)[0][0]

    def get_centers(self):
        return np.array(self._get_value("center"))

    def _compute_avg_speed(self):
        '''compute speed where speed is a function of a tracked objects total distance 
        traveled divided by the total amount of time the tracked object persists across the
        camera's field of view.'''
        total_time = self.positions[-1]["timestamp"] - self.positions[0]["timestamp"]

        if total_time == 0:
            logger.warning(f"Track {self.track_id}: Zero time elapsed, cannot compute speed.")
            return None
        
        return self.calculate_path_length() / total_time

    def _compute_path_length(self, positions):
        '''compute total euclidean distance traveled in pixels'''
        deltas = np.diff(positions, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        return float(distances.sum())
    
    def _compute_segment_speed(self, time):
        '''
        computes instant speed given a specific time
        '''       
        timestamps = self._validate_time_arg(time)
        if timestamps is None:
            return None
    
        # Index info
        idx = np.searchsorted(timestamps, time)

        # Space info
        travel_range = np.array(self._get_value("center"))[[idx-1, idx]]
        travel_dist = self._compute_path_length(travel_range)
        
        # Time info
        ts1, ts2 = timestamps[[idx-1, idx]]
        time_delta = ts2 - ts1

        # Computation / Cache assignment
        if time_delta == 0:
            self._segment_speeds[time] = 0.0
            return 0.0
        
        speed = travel_dist / time_delta
        self._segment_speeds[time] = speed
        
        return speed

    def _compute_instant_position(self, time):
        '''
        computes instant position given a specific time
        '''       
        timestamps = self._validate_time_arg(time)
        if timestamps is None:
            return None
        
        centers = self.get_centers()

        if time in timestamps:
            idx = np.where(timestamps == time)[0][0]
            center = centers[idx]
            return (center[0].item(), center[1].item())
        
        idx = np.searchsorted(timestamps, time)
        ts1, ts2 = timestamps[[idx - 1, idx]]
        time_range = ts2 - ts1
        factor = (time - ts1) / time_range if time_range != 0 else 1

        x1, y1 = centers[idx - 1]
        x2, y2 = centers[idx]
        x = x1 + factor * (x2 - x1)
        y = y1 + factor * (y2 - y1)
        
        pos = (x.item(), y.item())
        
        self._instant_positions[time] = pos
        return pos
    
    def _compute_instant_velocity(self, time):
        '''
        Calculate velocity vector at given time
        '''
        timestamps = self._validate_time_arg(time)
        
        if timestamps is None:
            return None
        
        idx = np.searchsorted(timestamps, time)
        ts1, ts2 = timestamps[[idx - 1, idx]]

        centers = self.get_centers()
        x1, y1 = centers[idx - 1]
        x2, y2 = centers[idx]
        
        delta = ts2 - ts1
        if delta == 0:
            self._instant_velocity[time] = (0, 0)
            return (0, 0)

        vx = (x2 - x1) / delta
        vy = (y2 - y1) / delta
        
        velocity = (vx.item(), vy.item())
        self._instant_velocity[time] = velocity

        return velocity
        

    def _initialize_positions(self, positions:List[dict]):
        '''
        Sort by timestamp, 
        dedupe by frame_idx, 
        and convert bbox coords to center coords.
        '''
        if len(positions) == 0:
            logger.warning(f"Track {self.track_id} contains no positions.")

        # Sort by timestamp
        sorted_pos = sorted(positions, key=lambda p: p["timestamp"])

        # Dedupe by frame_idx        
        deduped = list({d["frame_idx"]: d for d in sorted_pos}.values())
        
        if len(deduped) == 0:
            return []

        bbox_pts = np.array([d["bbox"] for d in deduped])
        cx = bbox_pts[:, [0, 2]].mean(axis=1)
        cy = bbox_pts[:, [1, 3]].mean(axis=1)
        w = bbox_pts[:, 2] - bbox_pts[:, 0]
        h = bbox_pts[:, 3] - bbox_pts[:, 1]

        processed = []
        for i, pos in enumerate(deduped):
            processed.append({
                "center": (float(cx[i]), float(cy[i])),
                "size": (float(w[i]), float(h[i])),
                "timestamp": pos["timestamp"],
                "frame_idx": pos["frame_idx"],
                "class_name": pos["class_name"],
                "conf": pos["conf"]
            })

        return processed
    
    def _validate_time_arg(self, time):
        timestamps = np.array(self._get_value("timestamp"))
        
        if time < timestamps.min() or time > timestamps.max():
            logger.warning(f"Track {self.track_id}: Time argument is out-of-bounds. Must be between {timestamps[0]} and {timestamps[-1]}")
            return None
        
        else:
            return timestamps
    
    def _sufficient_data(self):
        '''Utility function to check if trajectory has sufficient data points to perform operations'''
        return len(self.positions) >= 2
    
    def _get_value(self, key:str):
        return [p[key] for p in self.positions]