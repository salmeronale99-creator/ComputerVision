"""
Configuration file for the AprilTag vision system.
Change settings here instead of throughout the code.
"""

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Available resolutions to test:
# 640 x 480
# 1280 x 720 <- using at the moment
# 1920 x 1080
# 4056 x 3040 (full resolution)


TAG_FAMILY = "tag36h11"

# Physical size of the BLACK square (meters)
TAG_SIZE = 0.1695


SHOW_FPS = True
SHOW_DISTANCE = True
SHOW_RESOLUTION = True
SHOW_TAG_ID = True
DRAW_BOX = True


SAVE_LOG = True
LOG_FILE = "logs/experiment.csv"