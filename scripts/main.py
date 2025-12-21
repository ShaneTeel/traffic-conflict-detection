import cv2
from conflict_detection.studio import StudioManager
from conflict_detection.target_dev import TargetDetector, TargetTracker
from conflict_detection.trajectory import TrajCollector, TrajAnalyzer
from conflict_detection.utils import get_logger, setup_logging, path_checker

logger = get_logger(__name__)

setup_logging(
    log_level="INFO",
    log_to_file=True,
    log_dir="../logs/traffic",
    console_output=True
)

def main(file_in:str, file_out:str):

    studio = StudioManager(file_in)
    fps, _, _ = studio.get_metadata()
    detector = TargetDetector()
    tracker = TargetTracker(fps=fps)
    collector = TrajCollector(fps=fps, use_wall_time=False)

    studio.create_writer(file_out, fourcc="mp4v")

    logger.info("Starting video processing.")

    frames_count = 0
    for _ in range(5):
        ret, frame = studio.return_frame()
        if not ret:
            break
        
        frames_count += 1
        if frames_count % 10 == 0:
            logger.info(f"Processing frame {frames_count}")

        results = detector.detect(frame)
        tracks = tracker.track(results)
        collector.collect(tracks)

        if len(tracks) != 0:
            for track in tracks:
                x1, y1, x2, y2 = map(int, track["bbox"])
                class_name = track["class_name"]
                conf = track["conf"]
                track_id = track["track_id"]
                frame = studio.draw_boxes(frame, (x1, y1), (x2, y2), class_name, conf, track_id)

        studio.write_frame(frame)
        
        flag = studio.control_playback()
        if flag:
            break

    all_data = collector.get_all_traj_data()
    for track_id, tracks in list(all_data.items())[:5]:
        traj = TrajAnalyzer(track_id, tracks)
        centers = traj._get_value("center")
        x = centers[:, 0]
        y = centers[:, 1]
        

    
    studio.release_resources()

    logger.info(f"Finished processing {frames_count} frames.")
    logger.info(f"Output saved to: {file_out}")
    logger.info(f"Collected {len(collector)} unique tracks.")

    if path_checker(file_out):
        logger.info("Playing back processed video...")
        studio2 = StudioManager(file_out)
        studio2.print_menu()

        while True:
            ret, frame = studio2.return_frame()
            if not ret:
                break
            cv2.imshow("Test", frame)
            flag = studio2.control_playback()
            if flag:
                break

if __name__ == "__main__":
    file_in = "./media/in/waco-traffic-circle.mp4"
    file_out = "./media/out/waco-traffic-circle-processed.mp4"

    main(file_in, file_out)