import cv2
import os
import subprocess
import argparse

def capture_photo(fruit):
    # Open the webcam
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Press Space to Capture")

    if not cam.isOpened():
        print("Error: Could not access the webcam.")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Define square dimensions (80% of the smaller frame dimension)
        side_length = int(min(width, height) * 0.8)
        x1 = int((width - side_length) / 2)
        y1 = int((height - side_length) / 2)
        x2 = x1 + side_length
        y2 = y1 + side_length
        
        # Draw square on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display the frame with the square
        cv2.imshow("Press Space to Capture", frame)

        # Wait for key press
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed, exit without capturing
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed, capture the image
            # Crop the frame to the square region
            cropped_frame = frame[y1:y2, x1:x2]
            # Resize the cropped image to 150x150
            resized_frame = cv2.resize(cropped_frame, (150, 150))
            
            img_name = f"{fruit}/cam.png"
            os.makedirs(os.path.dirname(img_name), exist_ok=True)
            cv2.imwrite(img_name, resized_frame)
            print(f"{img_name} saved!")
            break

    # Release the webcam and close the window
    cam.release()
    cv2.destroyAllWindows()

def run_predict(fruit):
    command = ["python3", "predict.py", f"splitted/datasets/{fruit}/training/{fruit}", f"{fruit}/cam.png"]
    try:
        print(f'Execute command: {" ".join(command)}')
        #result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #print("Prediction Output:\n", result.stdout.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print("Error during prediction execution:\n", e.stderr.decode('utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='store_true', help='Use Apples model')
    parser.add_argument('-g', action='store_true', help='Use Grapes model')
    args = parser.parse_args()

    # Set the fruit based on the command-line argument; default is 'Grapes'
    if args.a:
        fruit = 'Apples'
    elif args.g:
        fruit = 'Grapes'
    else:
        fruit = 'Grapes'

    capture_photo(fruit)
    run_predict(fruit)
