import numpy as np

from conflict_detection.geometry import IPM
from conflict_detection.visualize import StudioManager
from conflict_detection.utils import get_logger, setup_logging

logger = get_logger(__name__)

setup_logging(
    log_level="INFO",
    log_to_file=True,
    log_dir="../logs/traffic",
    console_output=True
)

def main(file_in:str, img_pts:np.ndarray, world_pts:np.ndarray, world_val_pts:np.ndarray, img_val_pts:np.ndarray):
 
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

    img_pts = np.array([[[795, 462],
                         [954, 396],
                         [1349, 412],
                         [1363, 481]]], dtype=np.float32)

    world_pts = np.array([[[33.713863, -78.899981],
                           [33.713973, -78.899634],
                           [33.713650, -78.899528],
                           [33.713527, -78.899829]]])
    
    world_val_pts = np.array([[[33.713523, -78.900461],
                               [33.713400, -78.900377],
                               [33.713660, -78.899933]]])
    
    img_val_pts = np.array([[[595, 806],
                             [1134, 835],
                             [1093, 477]]], dtype=np.float32)

    main(file_in, img_pts, world_pts, world_val_pts, img_val_pts)