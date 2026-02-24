import cv2
import mediapipe as mp

# Import the tasks API for gesture recognition
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions

import webbrowser
import time
import pyautogui

# Path to the gesture recognition model
GESTURE_MODEL = "gesture_recognizer.task"  # Make sure this path matches where you have the model files saved on your computer!

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=GESTURE_MODEL),
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(options)

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        # and convert the BGR image to RGB.
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a Mediapipe Image object for the gesture recognizer
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Perform gesture recognition on the image
        result = gesture_recognizer.recognize(mp_image)

        # Draw the gesture recognition results on the image
        if result.gestures:
            recognized_gesture = result.gestures[0][0].category_name
            confidence = result.gestures[0][0].score

            # Example of taking browser action based on recognized gesture
            if recognized_gesture == "Open_Palm":
                webbrowser.open('http://google.com', new=2)
                time.sleep(1)
                pyautogui.write("Hello world!", interval=0.25)
                # Make sure to allow for time between recognized gestures so only one window is opened
                time.sleep(5)

            # Display recognized gesture and confidence
            cv2.putText(image, f"Gesture: {recognized_gesture} ({confidence:.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting image
        cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
