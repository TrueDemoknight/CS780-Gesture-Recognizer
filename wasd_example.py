import cv2
import mediapipe as mp

# Import the tasks API for gesture recognition
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions

import pyautogui

# Path to the gesture recognition model
GESTURE_MODEL = "gesture_recognizer.task"  # Update this to the correct path where the model is saved, if not in current directory

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

        # Flip the image horizontally and convert the BGR image to RGB.
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

            # Example of pressing keys with pyautogui based on recognized gesture
            if recognized_gesture == "Thumb_Up":
                pyautogui.press("w")
            elif recognized_gesture == "Thumb_Down":
                pyautogui.press("s")
            elif recognized_gesture == "Open_Palm":
                pyautogui.press("a")
            elif recognized_gesture == "Closed_Fist":
                pyautogui.press("d")
            elif recognized_gesture == "Victory":
                pyautogui.press("space")

            # Display recognized gesture and confidence 
            cv2.putText(image, f"Gesture: {recognized_gesture} ({confidence:.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting image (can comment this out for better performance later on)
        cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
