import cv2
import numpy as np
import os
import glob


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

IMAGE_FOLDER = os.path.join(
    BASE_DIR,
    "calibration",
    "images"
)

CALIBRATION_FOLDER = os.path.join(
    BASE_DIR,
    "calibration"
)


# ============================================================
# CHECKERBOARD SETTINGS
# ============================================================

# Number of INNER corners
CHECKERBOARD = (5, 7)

# Physical size of one checkerboard square in meters
SQUARE_SIZE = 0.0325


# ============================================================
# PREPARE CHECKERBOARD OBJECT POINTS
# ============================================================

# Fisheye calibration expects object points shaped (1, N, 3)
object_points_template = np.zeros(
    (1, CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    dtype=np.float64
)

object_points_template[0, :, :2] = (
    np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1]
    ].T.reshape(-1, 2)
)

object_points_template *= SQUARE_SIZE


# Lists for calibration data
object_points = []
image_points = []
valid_image_names = []


# ============================================================
# LOAD IMAGES
# ============================================================

images = sorted(
    glob.glob(
        os.path.join(IMAGE_FOLDER, "*.jpg")
    )
)

print("----------------------------------------")
print("IMX296 FISHEYE CAMERA CALIBRATION")
print("----------------------------------------")
print(f"Images found: {len(images)}")
print()


# ============================================================
# FIND CHECKERBOARD CORNERS
# ============================================================

image_size = None

for image_path in images:

    image = cv2.imread(image_path)

    if image is None:
        print(
            os.path.basename(image_path),
            "-> Could not read image"
        )
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    current_image_size = gray.shape[::-1]

    # Save resolution from first valid image
    if image_size is None:
        image_size = current_image_size

        print(
            f"Image resolution: "
            f"{image_size[0]} x {image_size[1]}"
        )
        print()

    # Reject images with different resolution
    if current_image_size != image_size:

        print(
            os.path.basename(image_path),
            "-> Wrong image size ✗"
        )

        continue


    # ========================================================
    # DETECT CHECKERBOARD
    # ========================================================

    found, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:

        # Fisheye calibration expects (1, N, 2)
        corners = corners.reshape(
            1,
            -1,
            2
        ).astype(np.float64)

        object_points.append(
            object_points_template.copy()
        )

        image_points.append(
            corners
        )

        valid_image_names.append(
            os.path.basename(image_path)
        )

        print(
            os.path.basename(image_path),
            "-> corners detected ✓"
        )

    else:

        print(
            os.path.basename(image_path),
            "-> NOT detected ✗"
        )


# ============================================================
# NUMBER OF VALID IMAGES
# ============================================================

good_images = len(object_points)

print()
print("----------------------------------------")
print(
    f"Good images: "
    f"{good_images} / {len(images)}"
)
print("----------------------------------------")


# ============================================================
# CHECK BEFORE CALIBRATION
# ============================================================

if good_images < 5:

    print()
    print("Not enough valid calibration images.")
    print("Need at least 5.")

    raise SystemExit


# ============================================================
# INITIAL CAMERA MATRIX AND DISTORTION
# ============================================================

camera_matrix = np.zeros(
    (3, 3),
    dtype=np.float64
)

dist_coeffs = np.zeros(
    (4, 1),
    dtype=np.float64
)


# Rotation and translation vectors
rvecs = [
    np.zeros(
        (1, 1, 3),
        dtype=np.float64
    )
    for _ in range(good_images)
]

tvecs = [
    np.zeros(
        (1, 1, 3),
        dtype=np.float64
    )
    for _ in range(good_images)
]


# ============================================================
# FISHEYE CALIBRATION SETTINGS
# ============================================================

calibration_flags = (
    cv2.CALIB_RECOMPUTE_EXTRINSIC
    | cv2.CALIB_CHECK_COND
    | cv2.CALIB_FIX_SKEW
)

criteria = (
    cv2.TERM_CRITERIA_EPS
    + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1e-6
)


# ============================================================
# RUN FISHEYE CALIBRATION
# ============================================================

try:

    rms, _, _, _, _ = cv2.fisheye.calibrate(
        object_points,
        image_points,
        image_size,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
        calibration_flags,
        criteria
    )

except cv2.error as error:

    print()
    print("----------------------------------------")
    print("FISHEYE CALIBRATION FAILED")
    print("----------------------------------------")
    print()
    print(error)
    print()
    print(
        "Do NOT delete or retake the calibration "
        "images yet."
    )

    raise SystemExit


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("----------------------------------------")
print("FISHEYE CALIBRATION SUCCESSFUL")
print("----------------------------------------")

print()
print(
    f"RMS reprojection error: "
    f"{rms:.4f} px"
)

print()
print("Camera Matrix:")
print(camera_matrix)

print()
print("Focal lengths:")
print(
    f"fx = "
    f"{camera_matrix[0, 0]:.2f} px"
)
print(
    f"fy = "
    f"{camera_matrix[1, 1]:.2f} px"
)

print()
print("Principal point:")
print(
    f"cx = "
    f"{camera_matrix[0, 2]:.2f} px"
)
print(
    f"cy = "
    f"{camera_matrix[1, 2]:.2f} px"
)

print()
print("Fisheye Distortion Coefficients:")
print(dist_coeffs)


# ============================================================
# PER-IMAGE RMS REPROJECTION ERROR
# ============================================================

print()
print("----------------------------------------")
print("PER-IMAGE RMS REPROJECTION ERROR")
print("----------------------------------------")

errors = []

for i in range(good_images):

    projected_points, _ = (
        cv2.fisheye.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )
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
        f"{valid_image_names[i]} -> "
        f"{error:.4f} px"
    )


mean_error = np.mean(errors)

print("----------------------------------------")

print(
    f"Mean per-image RMS error: "
    f"{mean_error:.4f} px"
)


# ============================================================
# SAVE FISHEYE CALIBRATION
# ============================================================

# IMPORTANT:
# Use different filenames so we don't overwrite
# the normal calibration.

np.save(
    os.path.join(
        CALIBRATION_FOLDER,
        "fisheye_cameraMatrix.npy"
    ),
    camera_matrix
)

np.save(
    os.path.join(
        CALIBRATION_FOLDER,
        "fisheye_distCoeffs.npy"
    ),
    dist_coeffs
)


print()
print("----------------------------------------")
print("Fisheye calibration saved:")
print("fisheye_cameraMatrix.npy")
print("fisheye_distCoeffs.npy")
print("----------------------------------------")