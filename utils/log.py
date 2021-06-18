import logging
import sys
from pathlib import Path


logger = logging.getLogger("alphalens")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(levelname)s :%(asctime)s %(name)s %(pathname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S"
)
log_path = Path(__file__).parent.parent.joinpath("alphalens.log").resolve()

file_handler = logging.FileHandler(log_path, encoding='utf-8')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
