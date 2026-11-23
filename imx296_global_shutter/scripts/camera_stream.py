from flask import Flask, Response
from picamera2 import Picamera2
import cv2

## FOR GLOBAR SHUTTER 

# --------------------------------
# Create Flask application
# --------------------------------

app = Flask(__name__)


# --------------------------------
# Initialize IMX296 camera
# --------------------------------

picam2 = Picamera2()

camera_config = picam2.create_video_configuration(
    main={
        "size": (1280, 720), ## ONly processing stream resolution 
        "format": "RGB888"
    },
    controls={
        "FrameRate": 30
    }
)

picam2.configure(camera_config)
picam2.start()

print("IMX296 Global Shutter Camera started.")


# --------------------------------
# Generate camera frames
# --------------------------------

def generate_frames():

    while True:

        frame = picam2.capture_array()

        # Compress JPEG slightly to reduce streaming load
        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

# --------------------------------
# Video stream
# --------------------------------

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# --------------------------------
# Simple home page
# --------------------------------

@app.route("/")
def index():

    return """
    <html>
        <head>
            <title>IMX296 Global Shutter Camera</title>
        </head>

        <body>
            <h1>IMX296 Global Shutter Camera</h1>

            <img src="/video_feed">
        </body>
    </html>
    """


# --------------------------------
# Run server
# --------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )