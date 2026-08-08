import numpy as np
import os
from flask import Flask, Response
from picamera2 import Picamera2
from pupil_apriltags import Detector
import cv2
import time
import threading


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CALIBRATION_FOLDER = os.path.join(
    BASE_DIR,
    "calibration"
)


# ============================================================
# LOAD IMX296 CAMERA CALIBRATION
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

print("IMX296 camera calibration loaded.")

print()
print("Camera Matrix:")
print(camera_matrix)

print()
print("Distortion Coefficients:")
print(dist_coeffs)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# CAMERA
# ============================================================

CAMERA_WIDTH = 1456
CAMERA_HEIGHT = 1088  ## testing full resolution 

picam2 = Picamera2()

camera_config = picam2.create_video_configuration(
    main={
        "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
        "format": "RGB888"
    },
    controls={
        "FrameRate": 30
    }
)

picam2.configure(camera_config)

picam2.start()

time.sleep(1)

print("IMX296 Global Shutter Camera started.")


# ============================================================
# APRILTAG DETECTOR
# ============================================================

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25
)

print("AprilTag detector initialized.")


# ============================================================
# APRILTAG SIZE
# ============================================================

# Physical width of detected AprilTag square:
# 3 inches = 0.0762 meters

TAG_SIZE = 0.0762 

half_size = TAG_SIZE / 2.0

object_points = np.array(
    [
        [-half_size, -half_size, 0],
        [ half_size, -half_size, 0],
        [ half_size,  half_size, 0],
        [-half_size,  half_size, 0]
    ],
    dtype=np.float32
)


# ============================================================
# SHARED FRAME
# ============================================================

latest_frame = None
frame_lock = threading.Lock()


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_loop():

    global latest_frame

    while True:

        frame = picam2.capture_array()

        with frame_lock:
            latest_frame = frame.copy()


camera_thread = threading.Thread(
    target=camera_loop,
    daemon=True
)

camera_thread.start()


# ============================================================
# VIDEO + APRILTAG PROCESSING
# ============================================================

def generate_frames():

    previous_time = time.time()

    while True:

        # ----------------------------------------------------
        # GET LATEST CAMERA FRAME
        # ----------------------------------------------------

        with frame_lock:

            if latest_frame is None:
                continue

            frame = latest_frame.copy()


        # ----------------------------------------------------
        # CONVERT TO GRAYSCALE
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )


        # ----------------------------------------------------
        # DETECT APRILTAGS
        # ----------------------------------------------------

        tags = detector.detect(gray)


        # ----------------------------------------------------
        # PROCESS EVERY DETECTED APRILTAG
        # ----------------------------------------------------

        for tag in tags:

            corners = tag.corners.astype(int)


            # =================================================
            # MEASURE APRILTAG WIDTH IN PIXELS
            # =================================================

            top_width = np.linalg.norm(
                tag.corners[1] - tag.corners[0]
            )

            bottom_width = np.linalg.norm(
                tag.corners[2] - tag.corners[3]
            )

            tag_width_pixels = (
                top_width + bottom_width
            ) / 2.0

            print(
                f"Tag width: "
                f"{tag_width_pixels:.2f} pixels"
            )


            # =================================================
            # DRAW APRILTAG BOUNDING BOX
            # =================================================

            for i in range(4):

                pt1 = tuple(corners[i])

                pt2 = tuple(
                    corners[(i + 1) % 4]
                )

                cv2.line(
                    frame,
                    pt1,
                    pt2,
                    (0, 255, 0),
                    2
                )


            # =================================================
            # DRAW TAG CENTER
            # =================================================

            center = tuple(
                tag.center.astype(int)
            )

            cv2.circle(
                frame,
                center,
                5,
                (0, 0, 255),
                -1
            )


            # =================================================
            # DRAW TAG ID
            # =================================================

            cv2.putText(
                frame,
                f"ID: {tag.tag_id}",
                (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )


            # =================================================
            # POSE ESTIMATION
            # =================================================

            # Raw AprilTag corner coordinates
            image_points = tag.corners.astype(
                np.float32
            ).reshape(
                -1,
                1,
                2
            )


            # -------------------------------------------------
            # UNDISTORT APRILTAG CORNERS
            # -------------------------------------------------

            undistorted_points = cv2.undistortPoints(
                image_points,
                camera_matrix,
                dist_coeffs,
                P=camera_matrix
            )

            undistorted_points = (
                undistorted_points
                .reshape(-1, 2)
                .astype(np.float32)
            )


            # -------------------------------------------------
            # SOLVE POSE
            # -------------------------------------------------
            #
            # The image points have already been undistorted,
            # so distortion coefficients are set to None.
            # -------------------------------------------------

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                undistorted_points,
                camera_matrix,
                None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )


            # =================================================
            # PROCESS POSE RESULT
            # =================================================

            if success:

                x = float(tvec[0, 0])
                y = float(tvec[1, 0])
                z = float(tvec[2, 0])

                # Full 3D Euclidean distance
                distance = np.sqrt(
                    x**2 +
                    y**2 +
                    z**2
                )


                # ---------------------------------------------
                # TERMINAL OUTPUT
                # ---------------------------------------------

                print(
                    f"ID {tag.tag_id} | "
                    f"X={x:.3f} m | "
                    f"Y={y:.3f} m | "
                    f"Z={z:.3f} m | "
                    f"Distance={distance:.3f} m"
                )


                # ---------------------------------------------
                # DISPLAY X
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    f"X: {x:.2f} m",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )


                # ---------------------------------------------
                # DISPLAY Y
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    f"Y: {y:.2f} m",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )


                # ---------------------------------------------
                # DISPLAY Z
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    f"Z: {z:.2f} m",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )


                # ---------------------------------------------
                # DISPLAY DISTANCE
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    f"Distance: {distance:.2f} m",
                    (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )


        # ====================================================
        # FPS
        # ====================================================

        current_time = time.time()

        time_difference = (
            current_time - previous_time
        )

        if time_difference > 0:

            fps = 1.0 / time_difference

        else:

            fps = 0.0

        previous_time = current_time


        # ====================================================
        # DISPLAY FPS
        # ====================================================

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )


        # ====================================================
        # DISPLAY TAG COUNT
        # ====================================================

        cv2.putText(
            frame,
            f"Tags: {len(tags)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )


        # ====================================================
        # DISPLAY CAMERA RESOLUTION
        # ====================================================

        cv2.putText(
            frame,
            f"{CAMERA_WIDTH} x {CAMERA_HEIGHT}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )


        # ====================================================
        # BROWSER PREVIEW
        # ====================================================

        preview = cv2.resize(
            frame,
            (640, 480)
        )

        success, buffer = cv2.imencode(
            ".jpg",
            preview,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                65
            ]
        )

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return """
    <html>

    <head>
        <title>IMX296 AprilTag Detection</title>
    </head>

    <body>

        <h1>
            IMX296 Global Shutter - AprilTag Detection
        </h1>

        <img src="/video_feed">

    </body>

    </html>
    """


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("----------------------------------------")
    print("IMX296 APRILTAG SERVER")
    print("----------------------------------------")
    print("Camera resolution: 1280 x 720")
    print(f"AprilTag size: {TAG_SIZE:.5f} m")
    print("Open browser:")
    print("http://<raspberry_pi_ip>:5000")
    print("----------------------------------------")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )