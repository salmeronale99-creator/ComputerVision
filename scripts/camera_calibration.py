import cv2
import numpy as np
import glob
import os

# -----------------------------------
# Checkerboard Settings
# -----------------------------------
CHECKERBOARD = (5, 7)

# Size of one square on the checkerboard (meters)
# The checkerboard used has squares of 32.5 mm.
SQUARE_SIZE = 0.0325

# -----------------------------------
# Project Paths ( be careful here)
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMERA ="module3"  ## <- Change to camera you are using!!!!

calibration_folder = os.path.join(BASE_DIR, "calibration", CAMERA)

image_folder = os.path.join(
    calibration_folder,
    "images",
    "*.jpg"
)

# Debug information
print("--------------------------------")
print("Camera:", CAMERA)
print("BASE_DIR:", BASE_DIR)
print("Calibration folder:", calibration_folder)
print("Image folder:", image_folder)
print("--------------------------------")


# -----------------------------------
# Prepare Object Points
# -----------------------------------
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

objp[:, :2] = (
    np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]]
    .T
    .reshape(-1, 2)
)

objp *= SQUARE_SIZE

object_points = []
image_points = []

# -----------------------------------
# Load Images
# -----------------------------------
images = glob.glob(image_folder)

print(f"Found {len(images)} images.")

successful = 0

for filename in images:

    image = cv2.imread(filename)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        None
    )

    if found:

        successful += 1

        object_points.append(objp)

        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            ),
        )

        image_points.append(refined_corners)

        print(f"✓ {os.path.basename(filename)}")

    else:

        print(f"✗ {os.path.basename(filename)}")

# -----------------------------------
# Make sure at least one image worked
# -----------------------------------
if successful == 0:
    print("No checkerboards were detected.")
    exit()

# -----------------------------------
# Camera Calibration
# -----------------------------------
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points,
    image_points,
    gray.shape[::-1],
    None,
    None,
)

# -----------------------------------
# Save Calibration
# -----------------------------------
np.save(
    os.path.join(calibration_folder, "cameraMatrix.npy"),
    cameraMatrix,
)

np.save(
    os.path.join(calibration_folder, "distCoeffs.npy"),
    distCoeffs,
)

print("\n--------------------------------")
print("Calibration Complete")
print("--------------------------------")
print(f"Images Found: {len(images)}")
print(f"Successful: {successful}")
print(f"Failed: {len(images) - successful}")

print("\nCamera Matrix:")
print(cameraMatrix)

print("\nDistortion Coefficients:")
print(distCoeffs)

print("\nSaved:")
print("cameraMatrix.npy")
print("distCoeffs.npy")