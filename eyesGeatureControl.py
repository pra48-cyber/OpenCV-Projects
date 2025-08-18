import cv2
import mediapipe as mp
import pyautogui
from collections import deque

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Smoothing buffer
history_len = 5
pos_history_x = deque(maxlen=history_len)
pos_history_y = deque(maxlen=history_len)

while True:
    _, frame = cam.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w = frame.shape[:2]

    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), cv2.FILLED)

            if id == 1:  # use one point to control mouse
                pos_history_x.append(x)
                pos_history_y.append(y)

                # Smooth by averaging last few points
                smooth_x = int(sum(pos_history_x) / len(pos_history_x))
                smooth_y = int(sum(pos_history_y) / len(pos_history_y))

                screen_x = screen_w / frame_w * smooth_x
                screen_y = screen_h / frame_h * smooth_y
                pyautogui.moveTo(screen_x, screen_y)

        # Eye blink for click
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), cv2.FILLED)

        if (left[0].y - left[1].y) < 0.004:  # corrected blink condition
            pyautogui.click()
            pyautogui.sleep(1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cam.release()
cv2.destroyAllWindows()
