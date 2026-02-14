import cv2
import os

# 1. SETTINGS
# Create a folder to save images if it doesn't exist
folder = os.path.join('assets', 'calibration_images')
if not os.path.exists(folder):
    os.makedirs(folder)

# Initialize the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)
count = 0

print("Controls: \n's' - Save Image \n'q' - Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the live feed
    cv2.imshow('Webcam Calibration Capture', frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Press 's' to save the current frame
    if key == ord('s'):
        img_name = os.path.join(folder, f'calib_{count:02d}.jpg')
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        count += 1
    
    # Press 'q' to quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()