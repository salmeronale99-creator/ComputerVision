from flask import Flask, Response, render_template
from picamera2 import Picamera2
from pupil_apriltags import Detector

import cv2
import numpy as np
import math
import time
import threading
import os
import csv

import config
from rover_controller import RoverController

# ============================================================
# PROJECT PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")
CALIBRATION_FOLDER = os.path.join(BASE_DIR, "calibration")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_FOLDER, exist_ok=True) # creates log folder

CSV_FILE = os.path.join(LOG_FOLDER, "apriltag_measurements.csv")
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "FPS", "Distance_m", "X_m", "Y_m", "Tag_ID", "Status"])

app = Flask(__name__, template_folder=TEMPLATE_FOLDER)

# ============================================================
# LOAD CAMERA CALIBRATION & START CAMERA
# ============================================================

#### Get Fisheye Calibration

from scipy.io import loadmat

calibration_path = os.path.join(
    CALIBRATION_FOLDER,
    "opencv_rectification.mat"
)

calibration = loadmat(calibration_path)

mapX = calibration["mapX_cv"].astype(np.float32)
mapY = calibration["mapY_cv"].astype(np.float32)

camera_matrix = calibration["K"].astype(np.float64)

dist_coeffs = np.zeros((4,1), dtype = np.float64) ## make sure to rectify each camera frame before detecting Apriltags and calculating 3D poses

print("MATLAB fisheye calibration loaded.")
print("Camera matrix:")
print(camera_matrix)

##########

#camera_matrix = np.load(os.path.join(CALIBRATION_FOLDER, "cameraMatrix.npy"))
#dist_coeffs = np.load(os.path.join(CALIBRATION_FOLDER, "distCoeffs.npy"))
#print("Camera calibration loaded.")

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)}
)
picam2.configure(camera_config)
picam2.start()
print("Camera started.")

detector = Detector(
    families=config.TAG_FAMILY,
    nthreads=4,
    quad_decimate=1.5,  # Slight speed boost for higher control Hz on Pi 5
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25
)

half_size = config.TAG_SIZE / 2
object_points = np.array([
    [-half_size, -half_size, 0],
    [ half_size, -half_size, 0],
    [ half_size,  half_size, 0],
    [-half_size,  half_size, 0]
], dtype=np.float32)

# ============================================================
# INITIALIZE ROVER CONTROLLER (MAVLink to MicoAir 743)
# ============================================================
rover = RoverController()
rover.arm()

# Shared frame buffer for Flask streaming
frame_lock = threading.Lock()
latest_jpeg = None
vision_running = True

# ============================================================
# BACKGROUND VISION + CONTROL LOOP
# ============================================================
def vision_control_loop():
    global latest_jpeg
    previous_time = time.time()
    last_log_time = time.time()

    while vision_running:
        # Capture frame
        frame = picam2.capture_array() 

        #Apply fisheye rectification
        frame = cv2.remap(
            frame,
            mapX,
            mapY,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        #Convert to grayscale

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect AprilTags
        tags = detector.detect(gray)

        x, y, z = 0.0, 0.0, 0.0
        rover_status = "SEARCHING (MOTORS STOPPED)"

        for tag in tags:
            corners = tag.corners.astype(int)
            for i in range(4):
                cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)

            center = tuple(tag.center.astype(int))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                tag.corners.astype(np.float32),
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if success:
                x = float(tvec[0, 0])
                y = float(tvec[1, 0])
                z = float(tvec[2, 0])

                # Extract tag yaw angle in radians from rotation matrix
                rmat, _ = cv2.Rodrigues(rvec)
                tag_yaw = math.atan2(rmat[1, 0], rmat[0, 0])

                # Send pose to RoverController -> updates MicoAir 743 PWM
                rover_status = rover.update_from_tag(x, y, z, tag_yaw)

                # Draw 3D coordinate axes
                axis_length = config.TAG_SIZE / 2
                axis_points = np.float32([
                    [axis_length, 0, 0],
                    [0, axis_length, 0],
                    [0, 0, -axis_length]
                ])

                # Fix coordinates axis

                axis_length = config.TAG_SIZE / 2

                axis_points = np.float32([
                    [0, 0, 0],
                    [axis_length, 0, 0],
                    [0,axis_length, 0],
                    [0, 0, -axis_length]
                ])

                img_axes, _ = cv2.projectPoints(
                    axis_points,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs
                )

                img_axes = img_axes.reshape(-1, 2).astype(int)

                origin = tuple(img_axes[0]
                )

                # X axis - Red
                cv2.line(frame, origin, tuple[img_axes[1], (0, 0, 0 255), 3])

                # Y axis - Green
                cv2.line(frame, origin, tuple(img_axes[2]), (0, 255, 0), 3)

                # Z axis - Blue

                cv2.line(frame, origin, tuple(img_axes[3]), (255, 0, 0), 3)


                # Coordinate axes will be drawn from the center of the AprilTag

                

                

                # Log to CSV once per second
                now = time.time()
                if now - last_log_time >= 1.0:
                    with open(CSV_FILE, "a", newline="") as file:
                        csv.writer(file).writerow([
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            round(1.0 / max(now - previous_time, 1e-3), 1),
                            round(z, 3), round(x, 3), round(y, 3),
                            tag.tag_id, rover_status
                        ])
                    last_log_time = now

            cv2.putText(frame, f"ID: {tag.tag_id}", (center[0] + 10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # FPS & Telemetry Overlay
        current_time = time.time()
        fps = 1.0 / max(current_time - previous_time, 1e-3)
        previous_time = current_time

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Status: {rover_status}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Z (Alt): {z:.2f} m", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"X: {x:.2f} m | Y: {y:.2f} m", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Encode JPEG for optional browser viewing
        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            with frame_lock:
                latest_jpeg = buffer.tobytes()

# ============================================================
# FLASK ROUTES (Read-Only Live View)
# ============================================================
def generate_frames():
    while True:
        with frame_lock:
            frame_bytes = latest_jpeg
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route("/")
def index():
    return render_template("apriltag.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    vision_thread = threading.Thread(target=vision_control_loop, daemon=True)
    vision_thread.start()
    print("Vision & Autonomous Chase Loop Started! Press Ctrl+C to stop safely.")

    try:
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        vision_running = False
        rover.shutdown()
        picam2.stop()
        print("Motors disarmed and camera stopped safely.")