from picamera2 import Picamera2
import cv2

# -----------------------------
# Initialize the camera
# -----------------------------
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (1280, 720)}
)

picam2.configure(config)
picam2.start()

print("Camera started!")
print("Press Ctrl+C to stop.")

try:
    while True:
        frame = picam2.capture_array()

        # Save the latest frame
        cv2.imwrite("images/live_frame.jpg", frame)

except KeyboardInterrupt:
    print("\nStopping camera...")

finally:
    picam2.stop()
    print("Camera stopped.")
    