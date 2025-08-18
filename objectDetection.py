import cv2
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the output layer names from the network
output_layer_names = net.getUnconnectedOutLayersNames()

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0)  # 0 is the default camera
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break  # Exit if the frame can't be read

    height, width, channels = frame.shape

    # --- Pre-processing ---
    # Create a 'blob' from the image for the network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # --- Forward Pass (Detection) ---
    outs = net.forward(output_layer_names)

    # --- Process Results ---
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # --- Draw Bounding Boxes ---
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence_label = int(confidences[i] * 100)
            color = (0, 255, 0)  # Green box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence_label}%', (x, y - 10), font, 2, color, 2)

    # Calculate and display FPS
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 30), font, 3, (0, 0, 0), 3)
    frame_id += 1

    # Display the resulting frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()