import cv2
import matplotlib.pyplot as plt
import numpy as np

from conflict_detection.detect import DetectionSystem
from conflict_detection.studio import StudioManager
from conflict_detection.utils import get_logger, setup_logging, path_checker

logger = get_logger(__name__)

setup_logging(
    log_level="INFO",
    log_to_file=True,
    log_dir="../logs/traffic",
    console_output=True
)

def main(file_in:str, file_out:str, dst_pts:np.ndarray):

    system = DetectionSystem(file_in, dst_pts)

    system.monitor_traffic(file_out=file_out)

    conflicts = system.detect_conflicts()

    if path_checker(file_out):
        logger.info("Playing back processed video...")
        studio = StudioManager(file_out)
        studio.print_menu()

        while True:
            ret, frame = studio.return_frame()
            if not ret:
                break
            cv2.imshow("Processed Video", frame)
            flag = studio.control_playback()
            if flag:
                break
    else:
        logger.warning("Cannot find video file assocaited with file_out.")

if __name__ == "__main__":
    # file_in = "./media/in/waco-traffic-circle.mp4"
    # file_out = "./media/out/waco-traffic-circle-processed.mp4"

    file_in = "./media/in/US_17_N_10th_Ave_20260107.mp4"
    file_out = "./media/out/US_17_N_10th_Ave_20260107-processed.mp4"
    world_pts = np.array([[[33.713863, 78.899982],
                           [33.713528, 78.899829],
                           [33.713651, 78.899529],
                           [33.713976, 78.899634]]])

    main(file_in, file_out, world_pts)