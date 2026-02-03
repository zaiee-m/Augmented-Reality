import cv2 # OpenCV - Only for video capture and display
import argparse
from utils import * # Define custom CV functions in utils.py
import sys

def main():
    parser = argparse.ArgumentParser(description="AR Tag Detection and Overlay")
    parser.add_argument("--video", type=str, help="Path to video file. If not provided, webcam (0) is used.", default=None)
    parser.add_argument("--template", type=str, help="Path to template image for overlay.", default=None)
    parser.add_argument("--model", type=str, help="Path to .obj model for 3D projection.", default=None)
    
    args = parser.parse_args()
    
    video_source = args.video if args.video else 0
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error opening source: {video_source}")
        return
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Frame", extract_and_draw_final(frame))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# def main():
#     parser = argparse.ArgumentParser(description="AR Tag Detection on Single Image")
    
#     # 1. Update Argument to take an image path
#     parser.add_argument("--image", type=str, required=True, help="Path to the input image file.")
    
#     args = parser.parse_args()
    
#     # 2. Load the Image
#     frame = cv2.imread(args.image)
    
#     if frame is None:
#         print(f"Error: Could not load image from '{args.image}'. Check the path.")
#         sys.exit(1)
        
#     print(f"Processing image: {args.image} | Size: {frame.shape}")

#     # 3. Process (Preprocess -> Extract -> Filter -> Draw)
#     # This uses your consolidated 'extract_and_draw_final' function
#     result_image = extract_and_draw_final(frame)
    
#     # 4. Display
#     cv2.imshow("Detected AR Tag", result_image)
    
#     # 5. Wait Indefinitely (0) until a key is pressed
#     print("Processing complete. Press any key to close the window...")
#     cv2.waitKey(0)
    
#     cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# python3 main.py --video "assets/videos_and_intrinsics/multipleTags.mp4"
# python3 main.py --image "assets/test.png"