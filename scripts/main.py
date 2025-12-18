from conflict_detection.utils import get_logger, setup_logging

logger = get_logger(__name__)

setup_logging(
    log_level="DEBUG",
    log_to_file=True,
    log_dir="../../logs/traffic",
    console_output=True
)

def main():

    pass

if __name__ == "__main__":

    logger.debug("Test")