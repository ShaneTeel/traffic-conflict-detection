import numpy as np

from numpy.typing import NDArray

from conflict_detection.detect import DetectionSystem
from conflict_detection.visualize import *
from conflict_detection.utils import get_logger, setup_logging

logger = get_logger(__name__)

setup_logging(
    log_level="INFO",
    log_to_file=True,
    log_dir="../logs/traffic",
    console_output=True
)

def main(file_in:str, file_out:str, model_path:str, dst_pts:NDArray, src_pts:NDArray):
 
    system = DetectionSystem(file_in, file_out, dst_pts, src_pts, model_path=model_path)
    
    conflicts = system.monitor_traffic()
    
    system.inspect_conflicts(conflicts, file_out="./media/out/TTC-conflicts-heatmap.html")
    
if __name__ == "__main__":
    file_in = "./media/in/US_17_N_10th_Ave_20260107.mp4"
    file_out = "./media/out/US_17_N_10th_Ave_20260107-processed.mp4"

    model_path = "./models/yolov8m.pt"
    
    img_pts = np.array([[[795, 462],
                         [954, 396],
                         [1349, 412],
                         [1363, 481]]], dtype=np.float32)

    world_pts = np.array([[[33.713873, -78.899988],
                           [33.713982, -78.899616],
                           [33.713655, -78.899524],
                           [33.713522, -78.899829]]])

    main(file_in, file_out, model_path, world_pts, img_pts)