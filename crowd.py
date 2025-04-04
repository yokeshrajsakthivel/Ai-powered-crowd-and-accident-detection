import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque

# ✅ Load YOLO Model
yolo_model = YOLO("yolov8n.pt")

# ✅ Parameters for Anomaly Detection
people_threshold = 30
movement_history = deque(maxlen=10)
speed_threshold = 15
movement_variance_threshold = 50

def process_crowd_feed(detect=True):  # ✅ Add a parameter to control AI detection
    """Processes video feed for crowd detection & anomaly behavior detection."""
    cap = cv2.VideoCapture(0)  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if detect:  # ✅ AI detection is enabled only if detect=True
            results = yolo_model(frame_rgb)
            count = 0
            people_positions = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])  

                    if class_id == 0:  # Class 0 = 'person' in COCO dataset
                        count += 1
                        person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        people_positions.append(person_center)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # ✅ Detect Abnormal Crowd Density
            anomaly_detected = False
            if count > people_threshold:
                anomaly_detected = True
                cv2.putText(frame, "CROWD ANOMALY DETECTED!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # ✅ Detect Unusual Movements (Erratic movement detection)
            movement_history.append(people_positions)
            if len(movement_history) == movement_history.maxlen:
                movement_speeds = []
                
                for i in range(len(movement_history) - 1):
                    prev_positions = movement_history[i]
                    curr_positions = movement_history[i + 1]

                    if len(prev_positions) == len(curr_positions):  
                        speeds = [np.linalg.norm(np.array(curr) - np.array(prev)) 
                                for curr, prev in zip(curr_positions, prev_positions)]
                        movement_speeds.extend(speeds)

                if movement_speeds:
                    avg_speed = np.mean(movement_speeds)
                    movement_variance = np.var(movement_speeds)

                    if avg_speed > speed_threshold or movement_variance > movement_variance_threshold:
                        anomaly_detected = True
                        cv2.putText(frame, "MOVEMENT ANOMALY DETECTED!", (20, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # ✅ If any anomaly is detected, show alert
            if anomaly_detected:
                cv2.putText(frame, "ALERT: UNUSUAL BEHAVIOR DETECTED!", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ✅ Display People Count
            cv2.putText(frame, f"People Count: {count}", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
