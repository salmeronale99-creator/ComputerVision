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
CALIBRATION_FOLDER = os.path.join(BASE_DIR, "calibration")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_FOLDER, exist_ok=True)

# ============================================================
# FLASK APPLICATION
# ============================================================

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


# FPS VARIABLES


previous_time = time.time()
fps = 0.0



# GENERATE VIDEO FRAMES


def generate_frames():

    global previous_time
    global fps

    while True:

        # ----------------------------------------------------
        # Capture Frame
        # ----------------------------------------------------
        frame = picam2.capture_array()

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ----------------------------------------------------
        # Detect AprilTags + Pose Estimation
        # ----------------------------------------------------
        tags = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(
                camera_matrix[0, 0],   # fx
                camera_matrix[1, 1],   # fy
                camera_matrix[0, 2],   # cx
                camera_matrix[1, 2],   # cy
            ),
            tag_size=config.TAG_SIZE
        )

        # ----------------------------------------------------
        # Draw Every Detected Tag
        # ----------------------------------------------------
        for tag in tags:

            corners = tag.corners.astype(int)

            # Draw green box
            for i in range(4):

                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i + 1) % 4])

                cv2.line(
                    frame,
                    pt1,
                    pt2,
                    (0, 255, 0),
                    2
                )

            # Draw center
            center = tuple(tag.center.astype(int))

            cv2.circle(
                frame,
                center,
                5,
                (0, 0, 255),
                -1
            )

            # ----------------------------
            # Pose
            # ----------------------------

            translation = tag.pose_t

            print("-------------------")
            print("Translation:")
            print(tag.pose_t)

            x = translation[0][0]
            y = translation[1][0]
            z = translation[2][0]

            # ----------------------------
            # Draw Tag ID
            # ----------------------------

            cv2.putText(
                frame,
                f"ID: {tag.tag_id}",
                (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            # ----------------------------
            # Draw Distance
            # ----------------------------

            cv2.putText(
                frame,
                f"Distance: {z:.2f} m",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # ----------------------------
            # Draw X
            # ----------------------------

            cv2.putText(
                frame,
                f"X: {x:.2f} m",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # ----------------------------
            # Draw Y
            # ----------------------------

            cv2.putText(
                frame,
                f"Y: {y:.2f} m",
                (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # ----------------------------------------------------
        # Draw Number of Tags
        # ----------------------------------------------------

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
        # FPS
        # ----------------------------------------------------

        current_time = time.time()

        fps = 1 / (current_time - previous_time)

        previous_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Resolution
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"{config.CAMERA_WIDTH} x {config.CAMERA_HEIGHT}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
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


        
# FLASK ROUTES


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

