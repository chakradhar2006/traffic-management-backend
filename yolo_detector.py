import cv2
from ultralytics import YOLO
import numpy as np
import os

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")

# Vehicle classes in COCO: 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

class YOLODetector:
    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        # Using YOLOv8 nano for speed
        if os.path.exists(MODEL_PATH):
            self.model = YOLO(MODEL_PATH)
        else:
            self.model = YOLO("yolov8n.pt")
            self.model.save(MODEL_PATH)

    def process_frame(self, frame):
        """
        Process a single frame: detect vehicles, draw bounding boxes, and count.
        """
        results = self.model(frame, classes=VEHICLE_CLASSES, verbose=False)[0]
        
        vehicle_count = 0
        emergency_detected = False
        bboxes = []
        processed_frame = frame.copy()
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = self.model.names[cls_id]
            
            vehicle_count += 1
            bboxes.append([x1, y1, x2, y2, class_name])
            
            # Simple simulation for emergency vehicle logic based on class and a dummy condition.
            # Apply heuristic for emergency vehicle
            is_emergency = self._is_emergency(frame, x1, y1, x2, y2, class_name)
            if is_emergency:
                emergency_detected = True
                color = (0, 0, 255) # Red for emergency
                label = f"EMERGENCY {class_name}"
            else:
                color = (0, 255, 0) # Green for normal
                label = class_name
                
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(processed_frame, f"{label} {conf:.2f}", (x1, max(y1-10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
        return processed_frame, vehicle_count, emergency_detected, bboxes

    def _is_emergency(self, frame, x1, y1, x2, y2, class_name):
        """
        Heuristic to detect emergency vehicles using HSV color masking on the bounding box crop.
        - Fire trucks: Large vehicles ('truck', 'bus') with significant RED colors.
        - Police cars: 'car' with significant BLUE coloring/lights.
        """
        if class_name not in ['truck', 'bus', 'car']:
            return False
            
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return False
            
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Red mask
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Blue mask
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        total_pixels = crop.shape[0] * crop.shape[1] + 1
        red_ratio = cv2.countNonZero(mask_red) / total_pixels
        blue_ratio = cv2.countNonZero(mask_blue) / total_pixels
        
        if class_name in ['truck', 'bus'] and red_ratio > 0.15:
            return True # Fire truck
        if class_name == 'car' and blue_ratio > 0.15:
            return True # Police car
            
        return False
