from picamera2 import Picamera2
import cv2

# ------------------------------------
# Initialize the Raspberry Pi camera
# ------------------------------------
picam2 = Picamera2()

# Configure the camera for preview
config = picam2.create_preview_configuration(
    main={"size": (1280, 720)}
)

picam2.configure(config)

# Start the camera
picam2.start()

print("Camera started successfully!")
print("Press 'q' to quit.")

# ------------------------------------
# Main loop
# ------------------------------------
while True:

    # Capture one frame
    frame = picam2.capture_array()

    # Display the frame
    cv2.imshow("Raspberry Pi Camera", frame)

    # Exit when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------------
# Cleanup
# ------------------------------------
picam2.stop()
cv2.destroyAllWindows()

print("Camera closed.")
