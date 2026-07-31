from picamera2 import Picamera2
from pupil_apriltags import Detector
import cv2
import time

# -----------------------------
# Camera Resolution
# Change this for your tests
# -----------------------------
WIDTH = 1280
HEIGHT = 720

picam2 = Picamera2()

config = picam2.create_still_configuration(
    main={"size": (WIDTH, HEIGHT)}
)

picam2.configure(config)
picam2.start()

# Give camera time to adjust
time.sleep(2)

frame = picam2.capture_array()

picam2.stop()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

detector = Detector(families="tag36h11")

tags = detector.detect(gray)

print("--------------------------------")
print(f"Resolution: {WIDTH} x {HEIGHT}")
print(f"Tags Found: {len(tags)}")
print("--------------------------------")

for tag in tags:

    print(f"Tag ID: {tag.tag_id}")
    print(f"Center: {tag.center}")

    corners = tag.corners.astype(int)

    print("Corners:")

    for i, c in enumerate(corners):
        print(f" Corner {i}: {tuple(c)}")

    cv2.polylines(
        frame,
        [corners],
        True,
        (0,255,0),
        3
    )

    cv2.putText(
        frame,
        f"ID {tag.tag_id}",
        tuple(corners[0]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255),
        2
    )

cv2.imwrite("result.jpg", frame)

print("\nAnnotated image saved as result.jpg")
