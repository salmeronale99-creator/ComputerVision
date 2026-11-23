import math
import time
import threading
from pymavlink import mavutil
import config

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

class RoverController:
    def __init__(self):
        print(f"Connecting to Flight Controller on {config.MAVLINK_PORT}...")
        self.master = mavutil.mavlink_connection(
            config.MAVLINK_PORT, baud=config.MAVLINK_BAUD
        )
        self.master.wait_heartbeat()
        print("Flight Controller heartbeat received!")

        # Thread-safe shared RC commands (1500 = neutral)
        self.lock = threading.Lock()
        self.steer_pwm = config.PWM_NEUTRAL     # channel 1: steering (yaw)
        self.throttle_pwm = config.PWM_NEUTRAL  # channel 3: throttle (forward/backward)
        self.lateral_pwm = config.PWM_NEUTRAL   # channel 4: strafe (left/right)
        self.last_tag_time = 0.0

        # Previous errors for derivative (D) term
        self.prev_ex = 0.0
        self.prev_ey = 0.0
        self.prev_time = time.time()

        self.running = True
        self.tx_thread = threading.Thread(target=self._mavlink_loop, daemon=True)
        self.tx_thread.start()

    def arm(self):
        print("Arming Rover...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1, 0, 0, 0, 0, 0, 0
        )
        time.sleep(2.0)

    def update_from_tag(self, x_cam, y_cam, z_cam, yaw_rad=0.0):
        """
        Maps AprilTag camera pose to ArduRover Omni3 RC channels:
          - ex (X offset) -> channel 4 (lateral strafe: > 1500 right, < 1500 left)
          - ey (Y offset) -> channel 3 (throttle: > 1500 forward, < 1500 reverse)
          - yaw (twist)   -> channel 1 (steering: > 1500 CW, < 1500 CCW)
        """
        now = time.time()
        dt = max(now - self.prev_time, 1e-3)

        # 1. Apply camera-to-cone physical offset
        ex = x_cam - config.CAM_TO_CONE_X
        ey = y_cam - config.CAM_TO_CONE_Y

        # 2. Derivative (tracks plane ground speed)
        dex = (ex - self.prev_ex) / dt
        dey = (ey - self.prev_ey) / dt

        # 3. PD Control Law -> Normalized commands [-1.0, +1.0]
        # Note: Flip the sign on throttle_cmd (-config.KP_XY * ey) if camera +Y points backward
        lateral_cmd = clamp(config.KP_XY * ex + config.KD_XY * dex, -1.0, 1.0)
        throttle_cmd = clamp(-config.KP_XY * ey - config.KD_XY * dey, -1.0, 1.0)
        steer_cmd = clamp(config.KP_YAW * yaw_rad, -0.6, 0.6)

        # 4. Convert to PWM values
        with self.lock:
            self.steer_pwm = int(config.PWM_NEUTRAL + steer_cmd * config.PWM_MAX_DELTA)
            self.throttle_pwm = int(config.PWM_NEUTRAL + throttle_cmd * config.PWM_MAX_DELTA)
            self.lateral_pwm = int(config.PWM_NEUTRAL + lateral_cmd * config.PWM_MAX_DELTA)
            self.last_tag_time = now

        self.prev_ex, self.prev_ey, self.prev_time = ex, ey, now

        # check capture -> fully theoretical right now
        radial_err = math.hypot(ex, ey)
        if z_cam <= config.TRIGGER_ALTITUDE_M and radial_err <= config.ALIGN_TOLERANCE_M:
            return "READY_TO_LOCK"
        return "TRACKING"

    def _mavlink_loop(self):
        while self.running:
            with self.lock:
                if time.time() - self.last_tag_time > config.WATCHDOG_TIMEOUT_S:
                    steer = throttle = lateral = config.PWM_NEUTRAL
                else:
                    steer = self.steer_pwm
                    throttle = self.throttle_pwm
                    lateral = self.lateral_pwm

            self.master.mav.rc_channels_override_send(
                self.master.target_system, self.master.target_component,
                steer,     # channel 1: steering (rotation)
                65535,     # channel 2: pitch (unused)
                throttle,  # channel 3: throttle (forward / reverse)
                lateral,   # channel 4: lateral strafe (left / right)
                65535, 65535, 65535, 65535  # channels 5-8: unused
            )
            time.sleep(0.05)  # 20 Hz update rate

    def shutdown(self):
        self.running = False
        time.sleep(0.1)
        # Neutralize channels
        self.master.mav.rc_channels_override_send(
            self.master.target_system, self.master.target_component,
            config.PWM_NEUTRAL, 65535, config.PWM_NEUTRAL, config.PWM_NEUTRAL,
            65535, 65535, 65535, 65535
        )
        time.sleep(0.1)
        # Release overrides (send 0) and disarm
        self.master.mav.rc_channels_override_send(
            self.master.target_system, self.master.target_component,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            0, 0, 0, 0, 0, 0, 0
        )