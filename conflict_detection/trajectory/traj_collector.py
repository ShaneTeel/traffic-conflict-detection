import time

from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class TrajCollector:

    def __init__(self, fps:int=30, use_wall_time:bool = False):
        self.fps = fps
        self.frame_count = 0
        self.use_wall_time = use_wall_time
        if self.use_wall_time:
            self.start_time = time.time() 
        self.trajectories = {}
        
        logger.debug("Initialized TrajCollector.")


    def collect(self, tracks):
        self.frame_count += 1
        if self.use_wall_time:
            timestamp = time.time() - self.start_time
        else:
            timestamp = self.frame_count / self.fps

        for track in tracks:
            tid = track["track_id"]
            if tid is None:
                logger.debug(f"'track_id' for frame {self.frame_count} is None. Skipping.")
                continue

            if tid not in self.trajectories:
                self.trajectories[tid] = []

            self.trajectories[tid].append({
                "bbox": track["bbox"],
                "timestamp": timestamp,
                "frame_idx": self.frame_count,
                "class_name": track["class_name"],
                "conf": track["conf"]
            })
    
    def get_all_traj_data(self):
        self._runtime_check()

        return self.trajectories
    
    def get_specific_traj_data(self, track_id:int):
        self._runtime_check()
        return self.trajectories.get(track_id, [])
    
    def get_all_track_ids(self):
        self._runtime_check()
        return list(self.trajectories.keys())
        
    def __len__(self):
        self._runtime_check()
        return len(self.trajectories)

    def _runtime_check(self):
        if not self.trajectories:
            raise RuntimeError("No trajectory data collected. Coll .collect() first.")