import numpy as np
import cv2
import glob

# 1. SETTINGS
# Number of internal corners (cols, rows)
CHECKERBOARD = (7, 10) 
# Side length of a square in your preferred unit (e.g., mm or meters)
SQUARE_SIZE = 25.0 

# Termination criteria for refining corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 2. DATA STORAGE
# 3D points in real world space
objpoints = [] 
# 2D points in image plane
imgpoints = [] 

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# 3. PROCESSING IMAGES
images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret == True:
        objpoints.append(objp)

        # Refine corner locations for better accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Optional: Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Calibration', img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

# 4. CALIBRATION
# ret: RMS error, mtx: Camera Matrix, dist: Distortion Coefficients
# rvecs: Rotation vectors, tvecs: Translation vectors
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix:\n", mtx)
print("\nDistortion coefficients:\n", dist)