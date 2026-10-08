from flask import Flask, Response, redirect
from picamera2 import Picamera2
import cv2
import os
import threading
import time


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SAVE_FOLDER = os.path.join(
    BASE_DIR,
    "calibration",
    "images"
)

os.makedirs(SAVE_FOLDER, exist_ok=True)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# CAMERA
# ============================================================

picam2 = Picamera2()

camera_config = picam2.create_video_configuration(
    main={
        "size": (1280, 720),
        "format": "RGB888"
    },
    controls={
        "FrameRate": 30
    }
)

picam2.configure(camera_config)
picam2.start()

time.sleep(1)


# ============================================================
# SHARED FRAME
# ============================================================

latest_frame = None
frame_lock = threading.Lock()


# ============================================================
# IMAGE COUNTER
# ============================================================

existing_images = [
    f for f in os.listdir(SAVE_FOLDER)
    if f.lower().endswith(".jpg")
]

image_count = len(existing_images)


# ============================================================
# CAMERA LOOP
# Only THIS function captures from the camera
# ============================================================

def camera_loop():

    global latest_frame

    while True:

        frame = picam2.capture_array()

        with frame_lock:
            latest_frame = frame.copy()


# Start camera capture in background
camera_thread = threading.Thread(
    target=camera_loop,
    daemon=True
)

camera_thread.start()


# ============================================================
# BROWSER STREAM
# ============================================================

def generate_frames():

    while True:

        with frame_lock:

            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        # Smaller image ONLY for browser preview
        preview = cv2.resize(
            frame,
            (640, 480)
        )

        success, buffer = cv2.imencode(
            ".jpg",
            preview,
            [cv2.IMWRITE_JPEG_QUALITY, 65]
        )

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

        # Browser does not need 30 FPS for calibration
        time.sleep(0.05)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return """
    <html>

    <head>
        <title>IMX296 Calibration</title>
    </head>

    <body>

        <h1>IMX296 Calibration</h1>

        <img src="/video_feed">

        <br><br>

        <form action="/capture" method="post">

            <button
                type="submit"
                style="font-size:30px;">

                Capture Calibration Image

            </button>

        </form>

    </body>

    </html>
    """


# ============================================================
# VIDEO
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# CAPTURE
# ============================================================

@app.route("/capture", methods=["POST"])
def capture():

    global image_count

    with frame_lock:

        if latest_frame is None:
            return redirect("/")

        # Copy latest FULL resolution processing frame
        frame = latest_frame.copy()

    filename = os.path.join(
        SAVE_FOLDER,
        f"image_{image_count:02d}.jpg"
    )

    cv2.imwrite(filename, frame)

    print(f"Saved: {filename}")

    image_count += 1

    return redirect("/")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("--------------------------------")
    print("IMX296 Calibration Capture")
    print("--------------------------------")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )