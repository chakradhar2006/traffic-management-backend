import cv2
import os
import time


class ZebraCrossingMonitor:
    def __init__(self, line_y_ratio=0.75):
        # 75% down the screen
        self.line_y_ratio = line_y_ratio
        # Prevent logging the same vehicle multiple times
        self.last_violation_time = 0 
        
    def check_violation(self, frame, bboxes, signal_state, lane_number):
        """
        Draw ROI line. If vehicle crosses during RED, take a snapshot.
        Returns a (filename, image_bytes) tuple on violation, or None.
        """
        h, w = frame.shape[:2]
        line_y = int(h * self.line_y_ratio)
        
        color = (0, 0, 255) if signal_state == "RED" else (0, 255, 0)
        cv2.line(frame, (0, line_y), (w, line_y), color, 2)
        cv2.putText(frame, "ZEBRA CROSSING", (10, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if signal_state != "RED":
            return None
            
        current_time = time.time()
        # Cooldown of 3 seconds per lane violation to prevent spam
        if current_time - self.last_violation_time < 3:
            return None
            
        violation_detected = False
        
        for box in bboxes:
            x1, y1, x2, y2, cls_name = box
            # If the bounding box intersects the line
            if y1 < line_y < y2:
                violation_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3) # Orange box
                break
                
        if violation_detected:
            self.last_violation_time = current_time
            timestamp = int(current_time)
            filename = f"lane_{lane_number}_{timestamp}.jpg"
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                return None
            return (filename, buffer.tobytes())
            
        return None
