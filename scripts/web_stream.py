from flask import Flask, Response, render_template, redirect
from picamera2 import Picamera2
import cv2
import os

# -----------------------------
# Create Flask application
# -----------------------------


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMERA = "module3" ## Specify camera that is being used


app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

# -----------------------------
# Initialize Raspberry Pi Camera
# -----------------------------
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (1280, 720)}
)

picam2.configure(config)
picam2.start()

SAVE_FOLDER = os.path.join(BASE_DIR, "calibration", CAMERA, "images")

os.makedirs(SAVE_FOLDER, exist_ok=True)

image_count = 0


# -----------------------------
# Generate camera frames
# -----------------------------
def generate_frames():

    while True:

        # Capture frame from camera
        frame = picam2.capture_array()

        # Convert BGR image into JPEG
        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        # Send frame to browser
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )


# -----------------------------
# Home page
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------
# Video stream
# -----------------------------
@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/capture", methods=["POST"])
def capture():

    global image_count

    frame = picam2.capture_array()

    filename = os.path.join(
        SAVE_FOLDER,
        f"image_{image_count:02d}.jpg"
    )

    cv2.imwrite(filename, frame)

    print(f"Saved {filename}")

    image_count += 1

    return redirect("/")

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )