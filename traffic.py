import cv2
import torch
from ultralytics import YOLO

# ✅ Load YOLO Model
yolo_model = YOLO("yolov5mu.pt")

# ✅ Fixed Accident-Prone Zone
accident_zone_ratio = 0.65  # 80% of frame height (adjustable if needed)
vehicle_positions = {}  # Dictionary to track vehicle positions for stationary detection
stationary_threshold = 3  # Number of frames a vehicle must remain stationary

def process_traffic_feed(video_path, detect=True):
    """Processes the video and detects vehicles while streaming frames for the dashboard."""
    global vehicle_positions
    cap = cv2.VideoCapture(video_path)
    frame_width, frame_height = 640, 480  # Resizing for consistency
    accident_zone_line = int(frame_height * accident_zone_ratio)  # Fixed accident-prone zone

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))

        if detect:
            prone_zone_count, non_prone_zone_count, stationary_vehicles = count_vehicles(frame, accident_zone_line)

            # ✅ Draw Fixed Accident-Prone Zone Line
            cv2.line(frame, (0, accident_zone_line), (frame_width, accident_zone_line), (0, 0, 255), 2)
            cv2.putText(frame, "Accident-Prone Zone", (5, accident_zone_line - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ✅ Display Vehicle Counts
            cv2.putText(frame, f"Vehicles in Safe Zone: {non_prone_zone_count}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Vehicles in Prone Zone: {prone_zone_count}", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Stationary Vehicles: {stationary_vehicles}", (20, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # ✅ Accident Alert Condition
            if prone_zone_count > 20 or stationary_vehicles > 0:
                cv2.putText(frame, "ACCIDENT ALERT!", (20, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ✅ Normal feed (no detection)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

def count_vehicles(frame, accident_zone_line):
    """Counts vehicles and detects stationary ones."""
    results = yolo_model(frame, imgsz=320)
    prone_zone_count, non_prone_zone_count, stationary_vehicles = 0, 0, 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])  
            object_name = yolo_model.names[class_id]

            if object_name in ["car", "truck", "bus", "motorcycle"]:
                vehicle_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                if y2 > accident_zone_line:
                    prone_zone_count += 1
                    color = (0, 0, 255)

                    if object_name not in vehicle_positions:
                        vehicle_positions[object_name] = []
                    vehicle_positions[object_name].append(vehicle_center)

                    if len(vehicle_positions[object_name]) > stationary_threshold:
                        if all(vehicle_positions[object_name][-1] == prev 
                               for prev in vehicle_positions[object_name][-stationary_threshold:]):
                            stationary_vehicles += 1

                else:
                    non_prone_zone_count += 1
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{object_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return prone_zone_count, non_prone_zone_count, stationary_vehicles
