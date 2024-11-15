import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import logging
from collections import defaultdict
import time

class LineCounter:
    def __init__(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point
        self.in_count = 0
        self.out_count = 0
        self.last_positions = {}
        
    def update(self, track_id, center_point, direction='vertical'):
        if track_id not in self.last_positions:
            self.last_positions[track_id] = center_point
            return
            
        last_pos = self.last_positions[track_id]
        current_pos = center_point
        
        # Line equation
        if direction == 'vertical':
            line_y = self.start_point[1]  # y-coordinate of horizontal line
            
            # Check if object crossed the line
            if (last_pos[1] < line_y and current_pos[1] >= line_y):
                self.in_count += 1
            elif (last_pos[1] >= line_y and current_pos[1] < line_y):
                self.out_count += 1
                
        self.last_positions[track_id] = current_pos

class SpeedEstimator:
    def __init__(self, fps: float, pixel_per_meter: float):
        self.fps = fps
        self.pixel_per_meter = pixel_per_meter
        self.vehicle_positions = {}
        self.speeds = defaultdict(list)
        self.speed_limit = 40  # Changed to 40 km/h
        self.high_speed_vehicles = set()  # Track high speed vehicles
        
    def estimate_speed(self, track_id: int, bbox: np.ndarray, timestamp: float) -> tuple:
        center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
        
        if track_id not in self.vehicle_positions:
            self.vehicle_positions[track_id] = [(center, timestamp)]
            return 0.0, False
            
        positions = self.vehicle_positions[track_id]
        if len(positions) > 5:
            positions.pop(0)
            
        prev_pos, prev_time = positions[-1]
        distance = np.sqrt((center[0] - prev_pos[0])**2 + (center[1] - prev_pos[1])**2)
        time_diff = timestamp - prev_time
        
        if time_diff > 0:
            speed = (distance / self.pixel_per_meter) / time_diff * 3.6
            self.speeds[track_id].append(speed)
            
            avg_speed = np.mean(self.speeds[track_id][-3:])
            
            # Mark vehicle as high speed if it exceeds limit
            if avg_speed > self.speed_limit:
                self.high_speed_vehicles.add(track_id)
            
            # Return speed and whether it's a high speed vehicle
            return avg_speed, track_id in self.high_speed_vehicles
            
        return 0.0, False

class VehicleClassifier:
    def __init__(self):
        self.size_thresholds = {
            'car': (0, float('inf'))  # Classify all as cars for now
        }
    
    def classify_by_size(self, bbox: np.ndarray) -> str:
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return 'car'  # Simplified classification

class TrafficAnalytics:
    def __init__(self):
        self.hourly_counts = defaultdict(int)
        self.vehicle_types = defaultdict(int)
        self.speeds = []
        self.violations = 0
        self.total_tracked = 0
        self.peak_flow = 0
        self.total_high_speed_detections = 0
        self.current_high_speed_count = 0
        self.tracked_vehicles = set()  # Keep track of unique vehicles
        self.high_speed_vehicles = set()  # Keep track of unique high-speed vehicles
        
    def update(self, timestamp: datetime, vehicle_type: str, speed: float, is_high_speed: bool, track_id: int):
        hour = timestamp.strftime('%H:00')
        
        # Only count if this is a new vehicle
        if track_id not in self.tracked_vehicles:
            self.tracked_vehicles.add(track_id)
            self.total_tracked += 1
            self.vehicle_types[vehicle_type] += 1
            self.hourly_counts[hour] += 1
        
        # Update speeds for all detections
        self.speeds.append(speed)
        
        # Count high speed vehicles only once
        if is_high_speed and track_id not in self.high_speed_vehicles:
            self.high_speed_vehicles.add(track_id)
            self.total_high_speed_detections += 1
            self.violations += 1
    
    def get_statistics(self) -> dict:
        return {
            'total_vehicles': len(self.tracked_vehicles),  # Use unique vehicle count
            'average_speed': np.mean(self.speeds) if self.speeds else 0,
            'max_speed': max(self.speeds) if self.speeds else 0,
            'violations': len(self.high_speed_vehicles),  # Use unique violations count
            'total_high_speed': len(self.high_speed_vehicles),
            'peak_hour': max(self.hourly_counts.items(), key=lambda x: x[1])[0] if self.hourly_counts else "N/A",
            'peak_flow': max(self.hourly_counts.values()) if self.hourly_counts else 0
        }

class VehicleTrackingSystem:
    def __init__(self, source_path: str, target_path: str):
        self.source_path = source_path
        self.target_path = target_path
        
        # Initialize video capture to get dimensions
        cap = cv2.VideoCapture(source_path)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        # Initialize YOLO model
        self.model = YOLO("yolov8x.pt")
        
        # Initialize line counter
        line_y = int(self.frame_height * 0.5)
        self.line_counter = LineCounter(
            start_point=(0, line_y),
            end_point=(self.frame_width, line_y)
        )
        
        # Vehicle tracking
        self.tracks = {}
        self.next_track_id = 0
        
        # Initialize new components
        self.speed_estimator = SpeedEstimator(
            fps=self.fps,
            pixel_per_meter=7.8
        )
        self.vehicle_classifier = VehicleClassifier()
        self.analytics = TrafficAnalytics()
        
        self.blink_state = False  # For blinking effect
        self.blink_counter = 0    # To control blink timing
        self.high_speed_count = 0 # Count of high speed vehicles
        
    def assign_tracks(self, detections, iou_threshold=0.3):
        if not self.tracks:  # First frame
            for det in detections:
                self.tracks[self.next_track_id] = {
                    'bbox': det['bbox'],
                    'time': time.time(),
                    'lost': 0
                }
                self.next_track_id += 1
            return
            
        # Calculate IoU between current detections and existing tracks
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]['bbox'] for tid in track_ids]
        
        matched_tracks = []
        matched_detections = []
        
        for i, det in enumerate(detections):
            best_iou = 0
            best_track = None
            
            for j, track_id in enumerate(track_ids):
                if track_id in matched_tracks:
                    continue
                    
                iou = self.calculate_iou(det['bbox'], track_boxes[j])
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track = track_id
            
            if best_track is not None:
                matched_tracks.append(best_track)
                matched_detections.append(i)
                self.tracks[best_track]['bbox'] = det['bbox']
                self.tracks[best_track]['time'] = time.time()
                self.tracks[best_track]['lost'] = 0
        
        # Add new tracks for unmatched detections
        for i in range(len(detections)):
            if i not in matched_detections:
                self.tracks[self.next_track_id] = {
                    'bbox': detections[i]['bbox'],
                    'time': time.time(),
                    'lost': 0
                }
                self.next_track_id += 1
        
        # Update lost counts and remove stale tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]['lost'] += 1
                if self.tracks[track_id]['lost'] > 30:  # Remove after 30 frames
                    del self.tracks[track_id]
    
    def calculate_iou(self, box1, box2):
        # Calculate intersection over union between two boxes
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection)
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        try:
            # Run YOLO detection
            results = self.model(frame, verbose=False)[0]
            
            # Process detections
            detections = []
            for box in results.boxes:
                if int(box.cls[0]) in [2, 3, 5, 7]:  # Vehicle classes
                    det = {
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'conf': float(box.conf[0]),
                        'cls': int(box.cls[0])
                    }
                    detections.append(det)
            
            # Update tracking
            self.assign_tracks(detections)
            
            # Create annotated frame
            annotated_frame = frame.copy()
            
            # Process each track
            timestamp = time.time()
            high_speed_detected = False
            current_high_speed_count = 0
            
            for track_id, track in self.tracks.items():
                bbox = track['bbox']
                
                # Estimate speed and get high speed status
                speed, is_high_speed = self.speed_estimator.estimate_speed(
                    track_id=track_id,
                    bbox=bbox,
                    timestamp=timestamp
                )
                
                # Update analytics with track_id
                self.analytics.update(
                    timestamp=datetime.now(),
                    vehicle_type='car',
                    speed=speed,
                    is_high_speed=is_high_speed,
                    track_id=track_id
                )
                
                # Update high speed detection flag and count
                if is_high_speed:
                    high_speed_detected = True
                    current_high_speed_count += 1
                
                # Draw visualizations
                x1, y1, x2, y2 = map(int, bbox)
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                
                # Update line counter
                self.line_counter.update(track_id, center)
                
                # Use red for high speed vehicles
                if is_high_speed:
                    color = (0, 0, 255)  # Red
                    # Add warning box with blinking effect
                    warning_margin = 5
                    if self.blink_state:  # Blink the box
                        cv2.rectangle(annotated_frame, 
                                    (x1-warning_margin, y1-warning_margin),
                                    (x2+warning_margin, y2+warning_margin),
                                    color, 
                                    4)
                    
                    # Add "HIGH SPEED DETECTED" label to vehicle
                    warning_label = "HIGH SPEED DETECTED"
                    label_size = cv2.getTextSize(warning_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw warning label background
                    cv2.rectangle(annotated_frame,
                                (x1, y2 + 5),
                                (x1 + label_size[0] + 10, y2 + 35),
                                (0, 0, 0),
                                -1)
                    
                    # Draw warning label
                    cv2.putText(annotated_frame,
                              warning_label,
                              (x1 + 5, y2 + 25),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.6,
                              (0, 0, 255),
                              2)
                    
                    # Add speed label
                    speed_label = f"{speed:.1f} km/h"
                    cv2.putText(annotated_frame,
                              speed_label,
                              (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.8,
                              color,
                              2)
                else:
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add normal speed label
                    cv2.putText(annotated_frame,
                              f"{speed:.1f} km/h",
                              (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.6,
                              color,
                              2)
            
            # Add global high speed warning with only total count
            if high_speed_detected:
                stats = self.analytics.get_statistics()
                warning_text = f"HIGH SPEED DETECTED! ({stats['total_high_speed']})"
                
                # Make text larger and more visible
                font_scale = 2.0  # Increased font size
                thickness = 4     # Increased thickness
                
                # Get text size for centering
                text_size = cv2.getTextSize(
                    warning_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    thickness
                )[0]
                
                # Position at top center of frame with more space from top
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = 100  # Moved down slightly for better visibility
                
                # Larger background for better visibility
                padding = 20
                if self.blink_state:
                    # Draw warning background
                    cv2.rectangle(
                        annotated_frame,
                        (text_x - padding, text_y - text_size[1] - padding),
                        (text_x + text_size[0] + padding, text_y + padding),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Draw warning text
                    cv2.putText(
                        annotated_frame,
                        warning_text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 255),  # Red color
                        thickness,
                        cv2.LINE_AA
                    )
            
            # Update blink state every 15 frames
            self.blink_counter += 1
            if self.blink_counter >= 15:
                self.blink_state = not self.blink_state
                self.blink_counter = 0
            
            # Draw statistics overlay
            stats = self.analytics.get_statistics()
            self.draw_statistics(annotated_frame, stats)
            
            return annotated_frame
            
        except Exception as e:
            logging.error(f"Error processing frame {frame_number}: {str(e)}")
            return frame
            
    def draw_statistics(self, frame, stats):
        # Create a larger statistics panel
        panel_height = 300  # Increased height
        panel_width = 500   # Increased width
        margin = 20
        
        # Create semi-transparent background with larger area
        overlay = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        cv2.rectangle(frame, 
                     (margin, margin), 
                     (margin + panel_width, margin + panel_height),
                     (0, 0, 0), 
                     -1)
        cv2.rectangle(frame, 
                     (margin, margin), 
                     (margin + panel_width, margin + panel_height),
                     (0, 255, 255), 
                     3)  # Thicker yellow border
        
        # Add title with better positioning
        cv2.putText(frame,
                    "Traffic Statistics",
                    (margin + 20, margin + 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,  # Larger font
                    (0, 255, 255),
                    2)
        
        # Add statistics with better spacing
        y_offset = margin + 90
        line_spacing = 40  # Increased spacing
        
        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Statistics with better formatting
        stats_text = [
            (f"Total Vehicles: {self.line_counter.in_count + self.line_counter.out_count} units", (255, 255, 255)),
            (f"Average Speed: {stats['average_speed']:.1f} km/h", (255, 255, 255)),
            (f"Maximum Speed: {stats['max_speed']:.1f} km/h", (255, 255, 255)),
            (f"Total High Speed: {stats['total_high_speed']}", (0, 0, 255)),  # Only show total high speed
            (f"Current Flow: {len(self.tracks)} vehicles/frame", (255, 255, 255)),
            (f"Time: {current_time}", (0, 255, 255))  # Added time with yellow color
        ]
        
        # Draw each statistic
        for text, color in stats_text:
            # Get text size for background
            (text_width, text_height), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                2
            )
            
            # Draw text background
            cv2.rectangle(frame,
                         (margin + 20, y_offset - text_height - 5),
                         (margin + 20 + text_width + 10, y_offset + 5),
                         (0, 0, 0),
                         -1)
            
            # Draw text
            cv2.putText(frame,
                        text,
                        (margin + 25, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2)
            y_offset += line_spacing

def main():
    try:
        # Initialize the tracking system
        tracker = VehicleTrackingSystem(
            source_path='/Users/mustafaumutozbek/Documents/finance_analysis/factory_analysis/ml-depth-pro/data/parking_test.mp4',
            target_path='/Users/mustafaumutozbek/Documents/finance_analysis/factory_analysis/ml-depth-pro/data/output_tracking.mp4'
        )
        
        print("Starting vehicle tracking...")
        tracker.process_video()
        print("\nProcessing completed!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Check vehicle_tracking.log for details")

if __name__ == "__main__":
    main()