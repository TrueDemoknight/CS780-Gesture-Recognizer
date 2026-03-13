import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui

HAND_MODEL = "hand_landmarker.task"
GESTURE_MODEL = "gesture_recognizer.task"

def distance(p1, p2):
  return np.linalg.norm(np.array(p1) - np.array(p2))

def recognize_ok(landmarks):
  thumb_tip = landmarks[4]
  index_tip = landmarks[8]
  return distance(thumb_tip, index_tip) < 0.05

def recognize_palm(landmarks):
  finger_tips = [8, 12, 16, 20]
  finger_pips = [6, 10, 14, 18]

  for tip, pip in zip(finger_tips, finger_pips):
    if landmarks[tip][1] < landmarks[pip][1]:
      return False

  return True

def pointing(landmarks):
  finger_tips = [6, 7, 8]

  for i in range(2):
    dist = abs(distance(landmarks[5], landmarks[17])) / 5
    if abs(distance(landmarks[finger_tips[i]], landmarks[finger_tips[i + 1]])) > dist:
      return False
  return True

def palm_flat(landmarks):
  finger_tips = [8, 12, 16, 20]
  finger_pips = [5, 9, 13, 17]

  dist = abs(distance(landmarks[5], landmarks[0])) / 5
  for i in range(3):
    if abs(landmarks[finger_tips[i]][1] - landmarks[finger_tips[i + 1]][1]) > dist or abs(landmarks[finger_pips[i]][1] - landmarks[finger_pips[i + 1]][1]) > dist:
      return False
  return True

def palm_left(landmarks):
  finger_tips = [8, 12, 16, 20]
  finger_sips = [7, 11, 15, 19]
  finger_mips = [6, 10, 14, 18]
  finger_pips = [5, 9, 13, 17]

  for tip, pip, sip, mip in zip(finger_tips, finger_pips, finger_sips, finger_mips):
    if landmarks[tip][0] > landmarks[sip][0] or landmarks[sip][0] > landmarks[mip][0] or landmarks[mip][0] > landmarks[pip][0]:
      return False
  return True

def palm_right(landmarks):
  finger_tips = [8, 12, 16, 20]
  finger_sips = [7, 11, 15, 19]
  finger_mips = [6, 10, 14, 18]
  finger_pips = [5, 9, 13, 17]

  for tip, pip, sip, mip in zip(finger_tips, finger_pips, finger_sips, finger_mips):
    if landmarks[tip][0] < landmarks[sip][0] or landmarks[sip][0] < landmarks[mip][0] or landmarks[mip][0] < landmarks[pip][0]:
      return False
  return True

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = vision.RunningMode

hand_options = vision.HandLandmarkerOptions(
  base_options=BaseOptions(model_asset_path=HAND_MODEL),
  running_mode=VisionRunningMode.VIDEO,
  num_hands=2
)

hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

gesture_options = vision.GestureRecognizerOptions(
  base_options=BaseOptions(model_asset_path=GESTURE_MODEL),
  running_mode=VisionRunningMode.VIDEO,
  num_hands=2
)

gesture_recognizer = vision.GestureRecognizer.create_from_options(
  gesture_options
)

cap = cv2.VideoCapture(0)

is_pointing = False

SCROLL_MAGNITUDE = 75

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

  if gesture_result.gestures and hand_result.hand_landmarks:
    lm = hand_result.hand_landmarks[0]

    landmarks = [(p.x, p.y) for p in lm]

    # Draw landmarks
    for x, y in landmarks:
      px = int(x * frame.shape[1])
      py = int(y * frame.shape[0])
      cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

    top_gesture = gesture_result.gestures[0][0]
    if top_gesture.category_name != "None":
      detected_text = f"CANNED: {top_gesture.category_name}"

      if top_gesture.category_name == "Thumb_Up":
        pyautogui.scroll(SCROLL_MAGNITUDE)
      elif top_gesture.category_name == "Thumb_Down":
        pyautogui.scroll(-SCROLL_MAGNITUDE)
      elif top_gesture.category_name == "Open_Palm":
        ...
      elif top_gesture.category_name == "Closed_Fist":
        ...
      elif top_gesture.category_name == "Victory":
        ...
      elif top_gesture.category_name == "Love":
        ...
    elif pointing(landmarks):
      detected_text = "Pointing"
      if not is_pointing:
        pyautogui.click()
      is_pointing = True
    elif palm_flat:
      if palm_left(landmarks):
        detected_text = "Palm Left"
        with pyautogui.hold("ctrl"):
          with pyautogui.hold("alt"):
            pyautogui.press("up")
        time.sleep(.5)
      elif palm_right(landmarks):
        detected_text = "Palm Right"
        with pyautogui.hold("ctrl"):
          with pyautogui.hold("alt"):
            pyautogui.press("down")
        time.sleep(.5)

    if not pointing(landmarks):
      is_pointing = False


  cv2.putText(frame, detected_text, (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2)

  cv2.imshow("Gesture System", frame)

  if cv2.waitKey(1) & 0xFF == 27:
    break

cap.release()
cv2.destroyAllWindows()
 