import numpy as np

from conflict_detection.detect import DetectionSystem
from conflict_detection.space import MapMaker
from conflict_detection.studio import StudioManager
from conflict_detection.utils import get_logger, setup_logging

logger = get_logger(__name__)

setup_logging(
    log_level="INFO",
    log_to_file=True,
    log_dir="../logs/traffic",
    console_output=True
)

def main(file_in:str, file_out:str, model_path:str, dst_pts:np.ndarray):
 
    system = DetectionSystem(file_in, dst_pts, model_path=model_path)
    studio = StudioManager(file_out)
    carto = MapMaker(dst_pts)

    system.monitor_traffic(file_out=file_out)
    coords, popups = system.geocode_conflicts()
    
    carto.generate_overlay(coords, popups, "Min TTC Overlay")
    carto.add_layer_control()
    carto.m.show_in_browser()

    studio.play_video()
    
if __name__ == "__main__":
    file_in = "./media/in/US_17_N_10th_Ave_20260107.mp4"
    file_out = "./media/out/US_17_N_10th_Ave_20260107-processed.mp4"

    model_path = "./models/yolov8m.pt"

    world_pts = np.array([[[33.713873, -78.899988],
                           [33.713982, -78.899616],
                           [33.713655, -78.899524],
                           [33.713522, -78.899829]]])

    main(file_in, file_out, model_path, world_pts)