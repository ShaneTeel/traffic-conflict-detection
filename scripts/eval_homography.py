from numpy.typing import NDArray

from conflict_detection.geometry import IPM
from conflict_detection.visualize import StudioManager
from conflict_detection.utils import get_logger, setup_logging

from conflict_detection.demo import SRC_PTS, DST_PTS, GEO_TEST, GEO_TRUE

logger = get_logger(__name__)

setup_logging(
    log_level="INFO",
    log_to_file=True,
    log_dir="../logs/traffic",
    console_output=True
)

def main(file_in:str, img_pts:NDArray, world_pts:NDArray, world_val_pts:NDArray, img_val_pts:NDArray):
 
    studio = StudioManager(file_in)

    ret, frame = studio.return_frame()

    if not ret: 
        studio.set_frame_idx(0)
        ret, frame = studio.return_frame()
    
    ipm = IPM(frame, world_pts, img_pts)
    
    Geo_pred, results = ipm.evaluate_projector(frame, world_val_pts, img_val_pts)
    ipm.inspect_results(world_val_pts, Geo_pred, results, file_out="./media/out/H-matrix-eval.html")

if __name__ == "__main__":
    file_in = "./media/in/US_17_N_10th_Ave_20260107.mp4"

    main(file_in, SRC_PTS, DST_PTS, GEO_TRUE, GEO_TEST)