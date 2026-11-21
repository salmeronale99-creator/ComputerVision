

"""
Configuration file for the AprilTag vision system.
Change settings here instead of throughout the code.
"""

CAMERA_WIDTH = 1456
CAMERA_HEIGHT = 1088

# Available resolutions to test:
# 640 x 480
# 1280 x 720 
# 1920 x 1080
# 4056 x 3040 (full resolution)
# 1456 x 1088 <- using this one!!!


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

# MAVLINK & ROVER CONTROL SETTINGS
MAVLINK_PORT = "tcp:192.168.2.1:5760" 
MAVLINK_BAUD = 115200

PWM_NEUTRAL = 1500
PWM_MAX_DELTA = 300             # Caps speed to 1200-1800 us for safe testing

# Static Offset from Upward Camera Lens to Male Docking Cone (meters)
CAM_TO_CONE_X = 0.00            # Measure physical X offset on rover
CAM_TO_CONE_Y = 0.08            # Measure physical Y offset on rover

# PD Control Gains (Tune Kp first, then add Kd to damp overshoot)
KP_XY = 1.2                     # Proportional gain for lateral/longitudinal chase
KD_XY = 0.15                    # Derivative gain (tracks plane ground velocity)
KP_YAW = 0.8                    # Proportional gain for heading alignment

# Docking Tolerances & Safety
ALIGN_TOLERANCE_M = 0.025       # 2.5 cm capture cone radius
TRIGGER_ALTITUDE_M = 0.35       # Trigger elevation/lock when plane is within 35 cm
WATCHDOG_TIMEOUT_S = 0.25       # Neutralize motors if tag is lost > 250 ms