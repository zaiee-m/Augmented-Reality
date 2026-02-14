AR Tag Detection and Overlay Engine

This project implements an Augmented Reality (AR) engine capable of detecting custom AR tags, estimating their pose, and overlaying either 2D images or 3D OBJ models onto them in real-time. It utilizes a custom C++ extension module for high-performance image processing operations.

Prerequisites

Python 3.x

C++ Compiler (GCC/Clang/MSVC)

Python Libraries:

numpy

opencv-python

Project Structure

main.py: The entry point for the application.

utils.py: Contains core logic (Tag detection, Homography, 3D Rendering, OBJ loading).

customCV.so / customCV.pyd: The compiled C++ extension module.

capture_calibration.py: Script to capture checkerboard images.

calibrate_camera.py: Script to calculate camera intrinsics.

assets/: Directory for calibration images, 3D models, and overlay templates.

Step 1: Compiling the Custom Module

Before running the Python scripts, you must compile the customCV C++ extension. This module handles heavy image processing tasks like Gaussian blurring and contour finding.

Ensure you have a setup.py file configured for the customCV module.

Open your terminal/command prompt in the project root.

Run the build command:

python setup.py build_ext --inplace


If successful, this will generate a .so (Linux/Mac) or .pyd (Windows) file in your directory.

Step 2: Camera Calibration

For accurate AR overlays, the system needs to know your camera's intrinsic parameters (focal length, optical center, distortion).

A. Capture Calibration Images

Run the capture script to take photos of a checkerboard pattern using your webcam.

python capture_calibration.py


Usage:

Hold a printed checkerboard visible to the camera.

Press 's' to save a snapshot (aim for 10-20 images at different angles).

Press 'q' to quit.

Images are saved to assets/calibration_images/.

B. Generate Intrinsics

Run the calibration script to process the saved images and generate the camera_calibration.npz file.

python calibrate_camera.py


Note: Ensure your checkerboard dimensions in the script match your physical board (default code assumes 7x10 internal corners).

Output: camera_calibration.npz containing the camera matrix and distortion coefficients.

Step 3: Running the AR Application

Once compiled and calibrated, you can run the main application.

python main.py


By default, the main.py provided utilizes the overlay_object function to render a 3D model.

Usage Guide: AR Functions

You can modify the loop in main.py to use different visualization modes provided in utils.py.

1. Tag Detection & ID Visualization

To simply detect tags, draw their boundaries, and display their decoded binary ID:

# In main.py
from utils import detect_tags_in_image

# Inside the video loop:
output_frame = detect_tags_in_image(frame, resizing_factor=1)
cv2.imshow("AR Output", output_frame)


2. 2D Image Overlay

To warp a flat 2D image (like a logo or poster) onto the perspective of the detected tag:

# In main.py
from utils import overlay_image

# Ensure you have an image at the specified path
template_path = "assets/iitd_logo_template.jpg"

# Inside the video loop:
output_frame = overlay_image(frame, template_path)
cv2.imshow("AR Output", output_frame)


3. 3D Object Overlay

To render a 3D .obj file on top of the tag. This function handles 3D projection, scaling, and orientation correction based on the tag's rotation.

# In main.py
from utils import overlay_object

# Ensure you have a valid .obj file
model_path = "assets/model1.obj"

# Inside the video loop:
output_frame = overlay_object(frame, model_path)
cv2.imshow("AR Output", output_frame)
