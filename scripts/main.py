import numpy as np
import matplotlib.pyplot as plt

from conflict_detection.detect import DetectionSystem
from conflict_detection.space import MapMaker
from conflict_detection.utils import get_logger, setup_logging

logger = get_logger(__name__)

setup_logging(
    log_level="INFO",
    log_to_file=True,
    log_dir="../logs/traffic",
    console_output=True
)

def main(file_in:str, file_out:str, model_path:str, dst_pts:np.ndarray):
 
    carto = MapMaker(dst_pts)

    system = DetectionSystem(file_in, dst_pts, model_path=model_path)

    _ = system.monitor_traffic(file_out=file_out, view=True)

    coords, popup = system.format_conflicts()

    carto.generate_overlay(coords, popup, "Min TTC Overlay")

    carto.add_layer_control()

    carto.m.show_in_browser()
    
if __name__ == "__main__":
    file_in = "./media/in/US_17_N_10th_Ave_20260107.mp4"
    file_out = "./media/out/US_17_N_10th_Ave_20260107-processed.mp4"

    model_path = "./models/yolov8m.pt"

    world_pts = np.array([[[33.713873, -78.899988],
                           [33.713982, -78.899616],
                           [33.713655, -78.899524],
                           [33.713522, -78.899829]]])

    main(file_in, file_out, model_path, world_pts)