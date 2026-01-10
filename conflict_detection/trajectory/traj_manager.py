from numpy.typing import NDArray
from typing import List

from .traj_collector import TrajCollector
from .traj_analyzer import TrajAnalyzer
from conflict_detection.homography import WorldProjector
from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class TrajManager:

    def __init__(self, projector:WorldProjector, fps:int=30, use_wall_time:bool = False):

        self.collector = TrajCollector(fps, use_wall_time)
        self.projector = projector
        self.analyzers = {}

        logger.debug(f"TrajManager successfully initialized.")

    def collect_tracks(self, tracks: List[dict]):
        self.collector.collect(tracks)
    
    def analyze_tracks(self):
        all_track_data = self.collector.get_all_traj_data()
        for track_id, track_data in all_track_data.items():
            traj = TrajAnalyzer(track_id, track_data)
            self.analyzers[track_id] = traj
        return self.analyzers

    def get_centers(self, track_id:int=None):
        all_centers = []
        if track_id is None:
            for track_id in self.analyzers:
                traj = self.analyzers[track_id]
                all_centers.append(traj.get_centers())
            return all_centers
        
        if track_id not in self.analyzers:
            raise KeyError(f"Error. {track_id} is not in analyzers. User must call `collect_tracks()` followed by `analyze_tracks()` first.")
        
        return self.analyzers[track_id].get_centers()
    
    def get_analyzer(self, track_id:int=None):
        if track_id is None:
            return self.analyzers
        
        if track_id not in self.analyzers:
            raise KeyError(f"Error. {track_id} is not in analyzers. User must call `collect_tracks()` followed by `analyze_tracks()` first.")
        
        return self.analyzers[track_id]