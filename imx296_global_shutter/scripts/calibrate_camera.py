import cv2
import numpy as np
import glob
import os


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

IMAGE_DIR = os.path.join(PROJECT_DIR, "calibration", "images")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "calibration")

# Save separately so we do NOT overwrite the current calibration
CAMERA_MATRIX_FILE = os.path.join(
    OUTPUT_DIR,
    "cameraMatrix_test.npy"
)

DIST_COEFFS_FILE = os.path.join(
    OUTPUT_DIR,
    "distCoeffs_test.npy"
)


# ============================================================
# CHECKERBOARD SETTINGS
# ============================================================

# Number of INNER corners
CHECKERBOARD = (5, 7)

# Actual measured square size:
# 25.5 cm / 8 squares = 3.1875 cm
# 3.1875 cm = 0.031875 m
SQUARE_SIZE = 0.031875


# ============================================================
# CREATE CHECKERBOARD 3D POINTS
# ============================================================

object_points_template = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    dtype=np.float32
)

object_points_template[:, :2] = (
    np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1]
    ].T.reshape(-1, 2)
)

object_points_template *= SQUARE_SIZE


# ============================================================
# STORAGE
# ============================================================

object_points = []
image_points = []
good_image_names = []


# ============================================================
# LOAD CALIBRATION IMAGES
# ============================================================

images = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
)

print()
print("==========================================")
print("IMX296 CAMERA CALIBRATION - TEST")
print("==========================================")
print()

print(f"Image directory: {IMAGE_DIR}")
print(f"Images found: {len(images)}")
print(f"Checkerboard inner corners: {CHECKERBOARD}")
print(f"Square size: {SQUARE_SIZE} m")
print()


if len(images) == 0:
    raise RuntimeError("No JPG calibration images were found.")


# ============================================================
# DETECT CHECKERBOARD
# ============================================================

image_size = None

for filename in images:

    image = cv2.imread(filename)

    if image is None:
        print(f"Could not read: {filename}")
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    current_size = (
        gray.shape[1],
        gray.shape[0]
    )

    # Make sure all images have the same resolution
    if image_size is None:
        image_size = current_size

        print(
            f"Calibration image resolution: "
            f"{image_size[0]} x {image_size[1]}"
        )
        print()

    elif current_size != image_size:
        print(
            f"Skipping {os.path.basename(filename)} "
            f"because resolution is {current_size}, "
            f"expected {image_size}"
        )
        continue


    # --------------------------------------------------------
    # Detect checkerboard using the more robust SB detector
    # --------------------------------------------------------

    found, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD,
        flags=(
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
    )


    if found:

        object_points.append(
            object_points_template.copy()
        )

        image_points.append(
            corners.astype(np.float32)
        )

        good_image_names.append(
            os.path.basename(filename)
        )

        print(
            f"[GOOD] {os.path.basename(filename)}"
        )

    else:

        print(
            f"[FAIL] {os.path.basename(filename)}"
        )


# ============================================================
# CHECK NUMBER OF GOOD IMAGES
# ============================================================

print()
print("------------------------------------------")
print(
    f"Good calibration images: "
    f"{len(object_points)} / {len(images)}"
)
print("------------------------------------------")
print()


if len(object_points) < 10:
    raise RuntimeError(
        "Not enough good checkerboard images. "
        "At least 10 are recommended."
    )


# ============================================================
# INITIAL CAMERA MATRIX
# ============================================================

width, height = image_size

# Start at image center
cx = width / 2.0
cy = height / 2.0

# Diagnostic initial focal-length estimate.
#
# IMPORTANT:
# We are NOT forcing calibration to stay at 775.
# This is only the optimizer's starting point.
initial_focal_length = 775.0


camera_matrix_initial = np.array(
    [
        [initial_focal_length, 0.0, cx],
        [0.0, initial_focal_length, cy],
        [0.0, 0.0, 1.0]
    ],
    dtype=np.float64
)


print("Initial camera matrix:")
print(camera_matrix_initial)

print()
print(
    f"Initial focal length: "
    f"{initial_focal_length:.2f} px"
)

print(
    f"Fixed principal point: "
    f"({cx:.2f}, {cy:.2f})"
)

print()


# ============================================================
# CALIBRATION FLAGS
# ============================================================

# USE_INTRINSIC_GUESS:
#     Start from our initial camera matrix.
#
# FIX_ASPECT_RATIO:
#     Keep fx = fy.
#
# FIX_PRINCIPAL_POINT:
#     Keep principal point at image center.
#
# ZERO_TANGENT_DIST:
#     Diagnostic assumption that tangential distortion is small.
#
# RATIONAL_MODEL:
#     Uses additional radial distortion coefficients, useful for
#     stronger wide-angle/barrel distortion.

calibration_flags = (
    cv2.CALIB_USE_INTRINSIC_GUESS
    | cv2.CALIB_FIX_ASPECT_RATIO
    | cv2.CALIB_FIX_PRINCIPAL_POINT
    | cv2.CALIB_ZERO_TANGENT_DIST
    | cv2.CALIB_RATIONAL_MODEL
)


# ============================================================
# RUN CALIBRATION
# ============================================================

print("Running calibration...")
print()


rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
    cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        camera_matrix_initial,
        None,
        flags=calibration_flags
    )
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("==========================================")
print("CALIBRATION RESULTS")
print("==========================================")
print()

print(f"OpenCV RMS error: {rms:.6f} pixels")

print()

print("Camera matrix:")
print(camera_matrix)

print()

print("Distortion coefficients:")
print(dist_coeffs.ravel())

print()


fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx_result = camera_matrix[0, 2]
cy_result = camera_matrix[1, 2]


print(f"fx = {fx:.3f} px")
print(f"fy = {fy:.3f} px")
print(f"cx = {cx_result:.3f} px")
print(f"cy = {cy_result:.3f} px")

print()


# ============================================================
# CALCULATE FIELD OF VIEW
# ============================================================

horizontal_fov = (
    2.0
    * np.arctan(width / (2.0 * fx))
    * 180.0
    / np.pi
)

vertical_fov = (
    2.0
    * np.arctan(height / (2.0 * fy))
    * 180.0
    / np.pi
)


print(
    f"Approximate horizontal FOV: "
    f"{horizontal_fov:.2f} degrees"
)

print(
    f"Approximate vertical FOV: "
    f"{vertical_fov:.2f} degrees"
)

print()


# ============================================================
# PER-IMAGE REPROJECTION ERROR
# ============================================================

print("==========================================")
print("PER-IMAGE REPROJECTION RMS")
print("==========================================")
print()


errors = []

for i in range(len(object_points)):

    projected_points, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    observed = image_points[i].reshape(-1, 2)
    projected = projected_points.reshape(-1, 2)

    difference = observed - projected

    error = np.sqrt(
        np.mean(
            np.sum(
                difference ** 2,
                axis=1
            )
        )
    )

    errors.append(error)

    print(
        f"{good_image_names[i]} : "
        f"{error:.4f} px"
    )


# ============================================================
# ERROR STATISTICS
# ============================================================

errors = np.array(errors)

print()
print("------------------------------------------")

print(
    f"Mean per-image RMS: "
    f"{np.mean(errors):.4f} px"
)

print(
    f"Median per-image RMS: "
    f"{np.median(errors):.4f} px"
)

print(
    f"Best image RMS: "
    f"{np.min(errors):.4f} px"
)

print(
    f"Worst image RMS: "
    f"{np.max(errors):.4f} px"
)

print("------------------------------------------")
print()


# ============================================================
# SHOW WORST IMAGES
# ============================================================

sorted_indices = np.argsort(errors)[::-1]

print("Worst calibration images:")
print()

number_to_show = min(10, len(errors))

for rank in range(number_to_show):

    index = sorted_indices[rank]

    print(
        f"{rank + 1}. "
        f"{good_image_names[index]} "
        f"-> {errors[index]:.4f} px"
    )


# ============================================================
# SAVE TEST CALIBRATION
# ============================================================

np.save(
    CAMERA_MATRIX_FILE,
    camera_matrix
)

np.save(
    DIST_COEFFS_FILE,
    dist_coeffs
)


print()
print("==========================================")
print("TEST CALIBRATION SAVED")
print("==========================================")
print()

print(CAMERA_MATRIX_FILE)
print(DIST_COEFFS_FILE)

print()
print(
    "Your existing cameraMatrix.npy and "
    "distCoeffs.npy were NOT overwritten."
)