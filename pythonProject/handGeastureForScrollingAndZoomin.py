import cv2
import mediapipe as mp
import pyautogui
import math
from collections import deque

# Camera and Mediapipe setup
cam = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

# Cursor smoothing
history_len = 5
pos_history_x = deque(maxlen=history_len)
pos_history_y = deque(maxlen=history_len)

# Zoom smoothing
prev_distance = None
zoom_cooldown = 0  # small cooldown to prevent spam

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror effect
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    frame_h, frame_w, _ = frame.shape

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_label in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = hand_label.classification[0].label  # 'Left' or 'Right'

            # ---------------- Right Hand: Cursor + Scroll ----------------
            if label == "Right":
                index = hand_landmarks.landmark[8]   # index tip
                middle = hand_landmarks.landmark[12] # middle tip

                ix, iy = int(index.x * frame_w), int(index.y * frame_h)
                mx, my = int(middle.x * frame_w), int(middle.y * frame_h)

                cv2.circle(frame, (ix, iy), 8, (255, 0, 0), -1)
                cv2.circle(frame, (mx, my), 8, (0, 0, 255), -1)

                # Cursor smoothing
                pos_history_x.append(ix)
                pos_history_y.append(iy)
                smooth_x = int(sum(pos_history_x) / len(pos_history_x))
                smooth_y = int(sum(pos_history_y) / len(pos_history_y))

                screen_x = screen_w / frame_w * smooth_x
                screen_y = screen_h / frame_h * smooth_y
                pyautogui.moveTo(screen_x, screen_y)

                # Scroll gesture → vertical gap between index & middle
                vertical_gap = abs(index.y - middle.y)
                horizontal_gap = abs(index.x - middle.x)

                if vertical_gap < 0.05 and horizontal_gap < 0.05:  # fingers close
                    if iy < frame_h // 2:
                        pyautogui.scroll(50)    # scroll up
                    else:
                        pyautogui.scroll(-50)   # scroll down

            # ---------------- Left Hand: Zoom ----------------
            if label == "Left":
                index = hand_landmarks.landmark[8]  # index tip
                thumb = hand_landmarks.landmark[4]  # thumb tip

                ix, iy = int(index.x * frame_w), int(index.y * frame_h)
                tx, ty = int(thumb.x * frame_w), int(thumb.y * frame_h)
                cv2.circle(frame, (ix, iy), 8, (255, 0, 0), -1)
                cv2.circle(frame, (tx, ty), 8, (0, 255, 0), -1)

                # Distance between index and thumb
                distance = math.hypot(index.x - thumb.x, index.y - thumb.y)

                if prev_distance is not None and zoom_cooldown == 0:
                    if distance - prev_distance > 0.05:   # fingers moving apart
                        pyautogui.hotkey("ctrl", "+")     # zoom in
                        zoom_cooldown = 5
                    elif prev_distance - distance > 0.05: # fingers moving closer
                        pyautogui.hotkey("ctrl", "-")     # zoom out
                        zoom_cooldown = 5

                prev_distance = distance

        # Decrease cooldown each frame
        if zoom_cooldown > 0:
            zoom_cooldown -= 1

            # Draw skeleton
        drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cam.release()
cv2.destroyAllWindows()
