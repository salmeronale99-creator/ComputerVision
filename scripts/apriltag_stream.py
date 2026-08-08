from flask import Flask, Response, render_template
from picamera2 import Picamera2
from pupil_apriltags import Detector

import cv2
import numpy as np
import time
import os
import csv

import config

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMERA = "module3" # Change to camera name that is being used!!
# ai_camera, module3, module3Wide


TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")
CALIBRATION_FOLDER = os.path.join(BASE_DIR, "calibration", CAMERA)
LOG_FOLDER = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_FOLDER, exist_ok=True)

# create CSV FILE

CSV_FILE = os.path.join(
    LOG_FOLDER,
    "apriltag_measurements.csv"
)

if not os.path.exists(CSV_FILE):

    with open(CSV_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Timestamp",
            "FPS",
            "Distance_m",
            "X_m",
            "Y_m",
            "Tag_ID"
        ])



# FLASK APP

app = Flask(
    __name__,
    template_folder=TEMPLATE_FOLDER
)

# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

camera_matrix = np.load(
    os.path.join(
        CALIBRATION_FOLDER, 
        "cameraMatrix.npy"
    )
)

dist_coeffs = np.load(
    os.path.join(
        CALIBRATION_FOLDER,
        "distCoeffs.npy"
    )
)

print("Camera calibration loaded.")

# ============================================================
# START CAMERA
# ============================================================

picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={
        "size": (
            config.CAMERA_WIDTH,
            config.CAMERA_HEIGHT
        )
    }
)

picam2.configure(camera_config)
picam2.start()

print("Camera started.")

# ============================================================
# APRILTAG DETECTOR
# ============================================================

detector = Detector(
    families=config.TAG_FAMILY,
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25
)

print("AprilTag detector initialized.")


# APRILTAG 3D MODEL

half_size = config.TAG_SIZE / 2

object_points = np.array([
    [-half_size, -half_size, 0],
    [ half_size, -half_size, 0],
    [ half_size,  half_size, 0],
    [-half_size,  half_size, 0]
], dtype=np.float32)
# FPS VARIABLES


previous_time = time.time()
fps = 0.0

last_log_time = time.time()





# GENERATE VIDEO FRAMES

def generate_frames():

    global previous_time
    global fps
    global last_log_time

    while True:

        # ----------------------------------------------------
        # Capture Frame
        # ----------------------------------------------------
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ----------------------------------------------------
        # Detect AprilTags
        # ----------------------------------------------------
        tags = detector.detect(gray)

        # Default values
        x = 0.0
        y = 0.0
        z = 0.0

        # ----------------------------------------------------
        # Process each detected tag
        # ----------------------------------------------------
        for tag in tags:

            corners = tag.corners.astype(int)

            # Draw bounding box
            for i in range(4):

                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i + 1) % 4])

                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # Draw center
            center = tuple(tag.center.astype(int))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # ---------------- Pose Estimation ----------------

            image_points = tag.corners.astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if success:

                x = float(tvec[0, 0])
                y = float(tvec[1, 0])
                z = float(tvec[2, 0])

                axis_length = config.TAG_SIZE / 2

                axis_points = np.float32([
                    [0, 0, 0],
                    [axis_length, 0, 0],
                    [0, axis_length, 0],
                    [0, 0, -axis_length]
                ])

                image_axis_points, _ = cv2.projectPoints(
                    axis_points,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs
                )

                image_axis_points = image_axis_points.reshape(-1, 2).astype(int)

                origin = tuple(image_axis_points[0])
                x_axis = tuple(image_axis_points[1])
                y_axis = tuple(image_axis_points[2])
                z_axis = tuple(image_axis_points[3])

                # Draw axes
                cv2.line(frame, origin, x_axis, (0, 0, 255), 3)
                cv2.line(frame, origin, y_axis, (0, 255, 0), 3)
                cv2.line(frame, origin, z_axis, (255, 0, 0), 3)

                # Save one measurement every second
                current_log_time = time.time()

                if current_log_time - last_log_time >= 1.0:

                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                    with open(CSV_FILE, "a", newline="") as file:

                        writer = csv.writer(file)

                        writer.writerow([
                            timestamp,
                            round(fps, 1),
                            round(z, 3),
                            round(x, 3),
                            round(y, 3),
                            tag.tag_id
                        ])

                    last_log_time = current_log_time

            # Draw Tag ID
            cv2.putText(
                frame,
                f"ID: {tag.tag_id}",
                (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------
        current_time = time.time()
        fps = 1.0 / (current_time - previous_time)
        previous_time = current_time

        # ----------------------------------------------------
        # Display Information
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"{config.CAMERA_WIDTH} x {config.CAMERA_HEIGHT}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Distance: {z:.2f} m",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"X: {x:.2f} m",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Y: {y:.2f} m",
            (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Tags: {len(tags)}",
            (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Encode JPEG
        # ----------------------------------------------------

        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buffer.tobytes() +
            b'\r\n'
        )

@app.route("/")
def index():
    return render_template("apriltag.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# MAIN 

if __name__ == "__main__":

    print("---------------------------------------")
    print("AprilTag Web Server Started")
    print("---------------------------------------")
    print("Open your browser and go to:")
    print("http://<raspberry_pi_ip>:5000")
    print("---------------------------------------")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )