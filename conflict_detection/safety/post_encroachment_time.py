import numpy as np

from typing import List

from conflict_detection.trajectory import TrajAnalyzer
from conflict_detection.utils import get_logger

logger = get_logger(__name__)

class PostEncroachmentTime:

    def __init__(self, ttc_thresh:float=1.5, pet_thresh:float=1.5, min_dist:float=0.5):
        
        self.ttc_thresh = ttc_thresh
        self.pet_thresh = pet_thresh
        self.min_dist = min_dist
        self.conflict_history = {}

        logger.debug("Conflict detector initialized.")

    def calculate_instant_ttc(self, traj_A:TrajAnalyzer, traj_B:TrajAnalyzer, time:float):
        '''
        Calcualte Time to Collision at specific time
        
        :param traj_A: Trajectory Analyzer object for a single tracked object
        :param traj_B: Trajectory Analyzer object for a single tracked object (different that traj_A)
        :param time: The specific time to calculate the two tracked objects time to collision
        
        :return: dict-like object containing the ttc info.
        :rtype: dict
        '''
        dummy = {
                "ttc": None,
                "collision_point": None,
                "min_distance": None,
                "time_checked": time,
                "track_A_id": traj_A.track_id,
                "track_B_id": traj_B.track_id,
                "conflict_detected": False
            }

        # Get pos / vel for both traj objects
        pos_A = traj_A.calculate_instant_position(time)
        vel_A = traj_A.calculate_instant_velocity(time)

        pos_B = traj_B.calculate_instant_position(time)
        vel_B = traj_B.calculate_instant_velocity(time)

        # Check if any pos / vel of traj_A / traj_B are None; return dummy dict if True
        if pos_A is None or vel_A is None or pos_B is None or vel_B is None:
            logger.warning("traj_A or traj_B position or velocity is None.")
            return dummy
        
        if vel_A == (0, 0) and vel_B == (0, 0):
            logger.warning("traj_A and traj_B velocity are both (0, 0). Two stationary objects have not ttc.")
            return dummy
        
        # Compute traj_B's relative position / velocity to traj_A
        rel_pos_x = pos_B[0] - pos_A[0]
        rel_pos_y = pos_B[1] - pos_A[1]

        rel_vel_x = vel_B[0] - vel_A[0]
        rel_vel_y = vel_B[1] - vel_A[1]

        rel_vel_sqrd = rel_vel_x**2 + rel_vel_y**2

        # Check if relative velocity is zero
        if rel_vel_sqrd == 0:
            logger.warning("Objects are moving at same velocity (parallel). No collision.")
            return dummy
        
        dot_product = rel_pos_x * rel_vel_x + rel_pos_y * rel_vel_y

        # Check if objects are moving apart; if so, no collision, return dummy
        if dot_product > 0:
            logger.info("Objects are moving apart; no collision.")
            return dummy
        
        logger.info(f"Objects at {time} are moving closer together; determining potential for collision.")
        
        # Determine when objects will be closest (in the past or in the future)
        t = -dot_product / rel_vel_sqrd

        # Check if closest point was in the past; if so, return dummy
        if t < 0:
            logger.info("Closest approach was in the past. No future collision detected.")
            return dummy
        
        logger.info(f"Objects will be closest in {t:.2f} seconds.")   

        # Compute future positions
        future_x_A = pos_A[0] + vel_A[0] * t
        future_y_A = pos_A[1] + vel_A[1] * t
        future_x_B = pos_B[0] + vel_B[0] * t
        future_y_B = pos_B[1] + vel_B[1] * t

        # Compute distance between projected positions at time t
        distance = ((future_x_B - future_x_A)**2 + (future_y_B - future_y_A)**2)**0.5

        if distance < self.min_dist:
            collision_x = pos_A[0] + vel_A[0] * t
            collision_y = pos_A[1] + vel_A[1] * t

            return {
                **dummy,
                "ttc": t, 
                "collision_point": (collision_x, collision_y),
                "min_distance": distance,
                "conflict_detected": True                
            }
        else:
            return dummy

    def _calculate_sweep_ttc(self, traj_A:TrajAnalyzer, traj_B:TrajAnalyzer, start:float=None, end:float=None, step:float=0.1):
        '''
        Calcualte TTC across multiple time points (sweep across time points)
        
        :param traj_A: Trajectory Analyzer object for a single tracked object
        :type traj_A: TrajAnalyzer
        :param traj_B: Trajectory Analyzer object for a single tracked object (different that traj_A)
        :type traj_B: TrajAnalyzer
        :param start: Beginning of time range
        :type start: float
        :param end: End of time range
        :type end: float
        :param step: Time step between start and end
        :type step: float, default 0.1s
        :return: A dict of dicts with time as the key for each time in the sweep.
        :rtype: dict[dict]
        '''
        results = {}

        if start is None or end is None:
            start, end = self._get_overlap_period(traj_A, traj_B)
            if start is None:
                return results

        num_steps = int((end - start) / step) + 1
        times = np.linspace(start, end, num_steps)
        
        for t in times:
            t_rounded = round(float(t), 2)
            results[t_rounded] = self.calculate_instant_ttc(traj_A, traj_B, t_rounded)
        return results

    def analyze_all_conflicts(self, analyzers:List[TrajAnalyzer], start:float=None, end:float=None, step:float=0.1):
        '''
        Calls `._calculate_sweep_ttc()` for every unique TrajAnalyzer() pair in contained 
        within the argument passed for `analyzers`. 
        If start / stop is None, the start and stop are auto calculated based on the overlap 
        period identified between the unique pair of TrajAnalyzer objects.
        
        :param analyzers: List of TrajAnalyzer() objects, each representing a single tracked object.
        :type analyzers: List[TrajAnalyzer]
        :param start: Beginning of time range
        :type start: float
        :param end: End of time range
        :type end: float
        :param step: Time step between start and end
        :type step: float, default 0.1s
        :return: A dict of dicts with time as the key for each time in the sweep.
        :rtype: dict[dict]
        '''
        if len(analyzers) < 2:
            logger.warning(f"The argument passed to analyzers must contain 2+ `TrajAnalyzer()` objects to perform TTC calculation.")
            return
        
        for i in range(len(analyzers) - 1):
            traj_A = analyzers[i]
            for traj_B in analyzers[i+1:]:
                pair_id = (traj_A.track_id, traj_B.track_id)
                self.conflict_history.update({
                    pair_id: self._calculate_sweep_ttc(traj_A, traj_B, start, end, step)
                })

        logger.info(f"Analyzed {len(self.conflict_history)} trajectory pairs.")
        return self.conflict_history

    def get_all_conflicts(self, conflicts_only:bool = True):
        '''
        Wrapper method to return the conflict history identified by earlier calls to `.analyze_all_conflicts()`.        
        '''
        if not conflicts_only:
            return self.conflict_history

        filtered = {}
        for pair_id, time_results in self.conflict_history.items():
            conflicts_at_times = {
                t: result for t, result in time_results.items() if result["conflict_detected"]
            }
            if conflicts_at_times:
                filtered[pair_id] = conflicts_at_times

        return filtered
    
    def get_minimum_ttc(self, target_pair:tuple=None):
        '''Get minimum TTC for specific pair'''
        if target_pair not in self.conflict_history:
            logger.debug("Invalid target pair provided.")
            raise KeyError(f"Pair {target_pair} not found. Run `analyze_all_conflicts()` first")
        
        time_results = self.conflict_history[target_pair]
        
        conflicts = {t: results for t, results in time_results.items() if results["conflict_detected"]}
        if not conflicts:
            logger.debug(f"Unalbe to determin minimum TTC for {target_pair} tracked object pair.")
            return {}
        
        min_time = min(conflicts.keys(), key=lambda t: conflicts[t]["ttc"])
        min_result = conflicts[min_time]
        
        return {
            "min_ttc": min_result["ttc"],
            "time_of_min": min_time,
            "collision_point": min_result["collision_point"],
            "min_distance": min_result["min_distance"]
        }       

    def get_all_minimum_ttc(self):
        results = {}
        for pair in self.conflict_history.keys():
            min_ttc = self.get_minimum_ttc(pair)
            if not min_ttc:
                continue
            
            results[pair] = min_ttc

        logger.info(f"Found minimum TTC for {len(results)} tracked object pairs.")

        return results

    def _get_overlap_period(self, traj_A: TrajAnalyzer, traj_B: TrajAnalyzer):

        times_A = np.array(traj_A._get_value("timestamp"))
        times_B = np.array(traj_B._get_value("timestamp"))

        start = max(times_A.min(), times_B.min())
        end = min(times_A.max(), times_B.max())

        if start > end:
            logger.warning(f"Tracks {traj_A.track_id} and {traj_B.track_id} don't overlap in time. ")
            return None, None
        
        logger.info(f"Analyzing tracks {traj_A.track_id} and {traj_B.track_id} from {start:.2f} to {end:.2f}.")
        return start, end