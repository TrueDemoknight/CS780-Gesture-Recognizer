import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------
# Model paths
# -------------------------
HAND_MODEL = "hand_landmarker.task"
GESTURE_MODEL = "gesture_recognizer.task"

# -------------------------
# Create HandLandmarker
# -------------------------
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = vision.RunningMode

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# -------------------------
# Create GestureRecognizer
# -------------------------
gesture_options = vision.GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=GESTURE_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

gesture_recognizer = vision.GestureRecognizer.create_from_options(
    gesture_options
)

# -------------------------
# Custom Gesture Logic
# -------------------------
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def recognize_ok(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    return distance(thumb_tip, index_tip) < 0.05

def recognize_palm(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    extended = []
    for tip, pip in zip(finger_tips, finger_pips):
        extended.append(landmarks[tip][1] < landmarks[pip][1])

    return all(extended)

# -------------------------
# OpenCV loop
# -------------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    timestamp = int(time.time() * 1000)

    # Run both models
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)
    gesture_result = gesture_recognizer.recognize_for_video(mp_image, timestamp)

    detected_text = "None"

    # -------------------------
    # 1. Check canned gestures
    # -------------------------
    if gesture_result.gestures:
        top_gesture = gesture_result.gestures[0][0]
        if top_gesture.score > 0.8:
            detected_text = f"CANNED: {top_gesture.category_name}"

    # -------------------------
    # 2. Check custom gestures
    # -------------------------
    if hand_result.hand_landmarks:
        lm = hand_result.hand_landmarks[0]
        landmarks = [(p.x, p.y) for p in lm]

        # Draw landmarks
        for x, y in landmarks:
            px = int(x * frame.shape[1])
            py = int(y * frame.shape[0])
            cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

        # If no canned gesture has been recognized, look for customs
        if detected_text == "None":
            if recognize_ok(landmarks):
                detected_text = "CUSTOM: OK"
            elif recognize_palm(landmarks):
                detected_text = "CUSTOM: PALM"

    cv2.putText(frame, detected_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Gesture System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()