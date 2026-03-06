from numpy.typing import NDArray

from conflict_detection.detect import DetectionSystem
from conflict_detection.visualize import *
from conflict_detection.utils import get_logger, setup_logging

from conflict_detection.demo import SRC_PTS, DST_PTS

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

    main(file_in, file_out, model_path, DST_PTS, SRC_PTS)