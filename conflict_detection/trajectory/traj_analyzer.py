from typing import List

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
        self.positions = positions
        

    def calculate_speed(self):
        pass
    
    def get_instant_position(self, time):
        pass

    def get_instant_speed(self, time):
        pass

    def calculate_path_length(self):
        pass

    def get_bbox(self):
        pass

    def export_to_dict(self):
        pass

    def get_stable_class(self):
        pass