import numpy as np

from conflict_detection.geometry import *
from conflict_detection.visualization import *
from conflict_detection.utils import *

logger = get_logger(__name__)

setup_logging(
    log_level="INFO",
    log_to_file=True,
    log_dir="../logs/traffic",
    console_output=True
)

def main(file_in:str, dst_pts:np.ndarray, val_pts:np.ndarray):
 
    studio = StudioManager(file_in)
    carto = MapMaker(dst_pts)

    ret, frame = studio.return_frame()

    if not ret: 
        studio.set_frame_idx(0)
        ret, frame = studio.return_frame()
    
    ipm = IPM(frame, dst_pts, val_pts)
    
if __name__ == "__main__":
    file_in = "./media/in/US_17_N_10th_Ave_20260107.mp4"

    dst_pts = np.array([[[33.713863, -78.899981],
                         [33.713973, -78.899634],
                         [33.713650, -78.899528],
                         [33.713527, -78.899829]]])
    
    val_pts = np.array([[[33.713523, -78.900461],
                         [33.713400, -78.900377],
                         [33.713660, -78.899933]]])

    main(file_in, dst_pts, val_pts)