import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2
import numpy as np
import time
import torch
from PIL import Image
from ultralytics import YOLO
import logging
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import traceback

from models import ResNet50, LSTMPyTorch
from utils import pth_processing, display_EMO_PRED, display_FPS

class EmotionProcessor:
    def __init__(self, backbone_model_path, lstm_model_name):
        self.DICT_EMO = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}
        
        # Initialize models
        self.pth_backbone_model = ResNet50(7, channels=3)
        self.pth_backbone_model.load_state_dict(torch.load(backbone_model_path))
        self.pth_backbone_model.eval()

        self.pth_LSTM_model = LSTMPyTorch()
        self.pth_LSTM_model.load_state_dict(torch.load(f'weights/FER_dinamic_LSTM_{lstm_model_name}.pt'))
        self.pth_LSTM_model.eval()
        
        # Initialize YOLO model for face detection with all verbose output disabled
        self.face_detector = YOLO('weights/yolov8l-face.pt', verbose=False)
        
        # Dictionary to store LSTM features for each face
        self.face_features = {}

        # Create visualization directory
        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')

        # Initialize logging
        self.setup_logging()

    def create_visualizations(self):
        """Create visualizations of the emotion analysis results"""
        try:
            # Get emotion data from statistics
            emotions = list(self.stats['emotions'].keys())
            values = list(self.stats['emotions'].values())
            
            # Validate data before visualization
            if not emotions or not values or any(np.isnan(v) for v in values):
                self.logger.warning("Invalid emotion data for visualization")
                return
            
            # Normalize values to ensure they sum to 1
            total = sum(values)
            if total == 0:
                self.logger.warning("All emotion values are zero")
                return
            
            values = [v/total for v in values]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Pie chart
            colors = plt.colormaps['Set3'](np.linspace(0, 1, len(emotions)))
            ax1.pie(values, labels=emotions, colors=colors, autopct='%1.1f%%')
            ax1.set_title('Emotion Distribution')
            
            # Bar chart
            ax2.bar(emotions, values, color=colors)
            ax2.set_title('Emotion Scores')
            ax2.set_xticklabels(emotions, rotation=45, ha='right')
            ax2.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig('emotion_analysis.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            self.logger.debug(traceback.format_exc())

    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # Setup file handler for detailed logs
        self.logger = logging.getLogger('EmotionProcessor')
        self.logger.setLevel(logging.INFO)
        
        # Create a file handler for the current session
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(f'logs/emotion_detection_{current_time}.log')
        file_handler.setLevel(logging.INFO)
        
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        self.logger.addHandler(file_handler)

        # Initialize statistics
        self.stats = {
            'total_frames': 0,
            'frames_with_faces': 0,
            'total_faces_detected': 0,
            'emotions': {emotion: 0 for emotion in self.DICT_EMO.values()},
            'processing_times': []
        }
        
        # Initialize frame-by-frame emotion tracking
        self.frame_emotions = []

    def update_statistics(self, num_faces, emotions, processing_time):
        """Update statistics with new detection results"""
        self.stats['total_frames'] += 1
        if num_faces > 0:
            self.stats['frames_with_faces'] += 1
            self.stats['total_faces_detected'] += num_faces
            for emotion in emotions:
                self.stats['emotions'][emotion] += 1
        self.stats['processing_times'].append(processing_time)
        self.frame_emotions.append(emotions)

    def save_statistics(self):
        """Save statistics to a JSON file"""
        stats_file = 'logs/emotion_statistics.json'
        
        # Calculate average processing time
        avg_processing_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        # Prepare statistics for saving
        stats_to_save = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_frames': self.stats['total_frames'],
            'frames_with_faces': self.stats['frames_with_faces'],
            'total_faces_detected': self.stats['total_faces_detected'],
            'face_detection_rate': self.stats['frames_with_faces'] / self.stats['total_frames'] if self.stats['total_frames'] > 0 else 0,
            'average_faces_per_frame': self.stats['total_faces_detected'] / self.stats['frames_with_faces'] if self.stats['frames_with_faces'] > 0 else 0,
            'emotion_distribution': self.stats['emotions'], # Store raw counts
            'average_processing_time_ms': avg_processing_time,
            'frame_by_frame_emotions': self.frame_emotions,
            'has_data': self.stats['total_faces_detected'] > 0
        }

        # Load existing statistics if file exists
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                try:
                    existing_stats = json.load(f)
                    if not isinstance(existing_stats, list):
                        existing_stats = [existing_stats]
                except json.JSONDecodeError:
                    existing_stats = []
        else:
            existing_stats = []

        # Add new statistics
        existing_stats.append(stats_to_save)

        # Save updated statistics
        with open(stats_file, 'w') as f:
            json.dump(existing_stats, f, indent=4)

    def get_display_size(self, frame_width, frame_height):
        """Calculate display size based on frame dimensions"""
        # Common screen resolutions and their corresponding display sizes
        resolutions = {
            (1920, 1080): (1280, 720),  # Full HD
            (2560, 1440): (1600, 900),  # 2K
            (3840, 2160): (1920, 1080), # 4K
            (1366, 768):  (1024, 576),  # HD
            (1440, 900):  (1152, 720),  # WXGA+
            (1600, 900):  (1280, 720),  # HD+
        }
        
        # Default size if no matching resolution
        default_width = 1280
        default_height = 720
        
        # Find the closest resolution
        min_diff = float('inf')
        best_size = (default_width, default_height)
        
        for (res_w, res_h), (disp_w, disp_h) in resolutions.items():
            diff = abs(res_w - frame_width) + abs(res_h - frame_height)
            if diff < min_diff:
                min_diff = diff
                best_size = (disp_w, disp_h)
        
        return best_size

    def resize_frame(self, frame, target_width, target_height):
        """Resize frame maintaining aspect ratio"""
        h, w = frame.shape[:2]
        aspect = w / h
        
        if w > h:
            new_width = target_width
            new_height = int(new_width / aspect)
        else:
            new_height = target_height
            new_width = int(new_height * aspect)
            
        return cv2.resize(frame, (new_width, new_height))

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        # box format: (x1, y1, x2, y2)
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def merge_overlapping_boxes(self, boxes, iou_threshold=0.3):
        """Merge overlapping bounding boxes"""
        if not boxes:
            return []

        # Sort boxes by area (largest first)
        boxes = sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
        merged_boxes = []
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue

            current_box = boxes[i]
            used[i] = True

            # Find all boxes that overlap with current box
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue

                if self.calculate_iou(current_box, boxes[j]) > iou_threshold:
                    # Merge boxes
                    x1 = min(current_box[0], boxes[j][0])
                    y1 = min(current_box[1], boxes[j][1])
                    x2 = max(current_box[2], boxes[j][2])
                    y2 = max(current_box[3], boxes[j][3])
                    current_box = (x1, y1, x2, y2)
                    used[j] = True

            merged_boxes.append(current_box)

        return merged_boxes

    def process_video(self, input_video_path, output_video_path):
        cap = cv2.VideoCapture(input_video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = np.round(cap.get(cv2.CAP_PROP_FPS))

        # Log video information
        self.logger.info(f"Processing video: {input_video_path}")
        self.logger.info(f"Video resolution: {w}x{h}, FPS: {fps}")

        # Create window with initial size
        cv2.namedWindow('Video Processing', cv2.WINDOW_NORMAL)
        
        # Get target size based on frame dimensions
        target_width, target_height = self.get_display_size(w, h)
        
        # Set initial window size
        cv2.resizeWindow('Video Processing', target_width, target_height)

        vid_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        faces_found = False  # Новый флаг
        try:
            while cap.isOpened():
                t1 = time.time()
                success, frame = cap.read()
                if frame is None:
                    break

                # Detect faces using YOLO with all verbose output disabled
                results = self.face_detector(frame, conf=0.5, verbose=False, show=False)[0]
                
                frame_emotions = []
                merged_boxes = []
                if len(results.boxes) > 0:
                    faces_found = True
                    # Collect all bounding boxes
                    boxes = []
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        boxes.append((x1, y1, x2, y2))

                    # Merge overlapping boxes
                    merged_boxes = self.merge_overlapping_boxes(boxes)

                    # Process each merged box
                    for face_idx, (startX, startY, endX, endY) in enumerate(merged_boxes):
                        cur_face = frame[startY:endY, startX:endX]
                        
                        # Skip if face region is too small
                        if cur_face.size == 0 or cur_face.shape[0] < 20 or cur_face.shape[1] < 20:
                            continue
                        
                        cur_face = pth_processing(Image.fromarray(cur_face))
                        features = torch.nn.functional.relu(self.pth_backbone_model.extract_features(cur_face)).detach().numpy()

                        # Initialize or update features for this face
                        if face_idx not in self.face_features:
                            self.face_features[face_idx] = [features] * 10
                        else:
                            self.face_features[face_idx] = self.face_features[face_idx][1:] + [features]

                        lstm_f = torch.from_numpy(np.vstack(self.face_features[face_idx]))
                        lstm_f = torch.unsqueeze(lstm_f, 0)
                        output = self.pth_LSTM_model(lstm_f).detach().numpy()
                
                        cl = np.argmax(output)
                        label = self.DICT_EMO[cl]
                        frame_emotions.append(label)
                        frame = display_EMO_PRED(frame, (startX, startY, endX, endY), 
                                               f'Face {face_idx + 1}: {label} {output[0][cl]:.1%}', 
                                               line_width=3)

                t2 = time.time()
                processing_time = (t2 - t1) * 1000  # Convert to milliseconds
                frame = display_FPS(frame, 'FPS: {0:.1f}'.format(1 / (t2 - t1)), box_scale=.5)
                
                # Обновляем статистику только если есть лица
                if len(merged_boxes) > 0:
                    self.update_statistics(len(merged_boxes), frame_emotions, processing_time)
                    # Логируем только кадры с лицами
                    self.logger.info(f"Frame {self.stats['total_frames']}: {len(frame_emotions)} faces detected, "
                                   f"Emotions: {frame_emotions}, Processing time: {processing_time:.2f}ms")
                
                # Resize frame for display while maintaining aspect ratio
                display_frame = self.resize_frame(frame, target_width, target_height)
                
                # Draw bounding box and text for each face
                for (x1, y1, x2, y2), emotion in zip(merged_boxes, frame_emotions):
                    w = x2 - x1
                    h = y2 - y1
                    # Bounding Box Color based on emotion
                    bbox_color = (0, 255, 0) # Green default (B, G, R)
                    if emotion == 'Anger':
                        bbox_color = (0, 0, 255) # Red
                    elif emotion == 'Happiness':
                        bbox_color = (0, 255, 0) # Green
                    elif emotion == 'Sadness':
                        bbox_color = (255, 0, 0) # Blue
                    elif emotion == 'Neutral':
                        bbox_color = (200, 200, 200) # Light Gray
                    elif emotion == 'Surprise':
                        bbox_color = (0, 255, 255) # Yellow
                    elif emotion == 'Fear':
                        bbox_color = (128, 0, 128) # Purple
                    elif emotion == 'Disgust':
                        bbox_color = (0, 128, 128) # Teal
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                    
                    # Text properties
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5 # Reduced font size
                    font_thickness = 1 # Adjusted thickness
                    text_color = (255, 255, 255) # White text
                    padding = 5 # Padding around text

                    # Get text size
                    (text_w, text_h), baseline = cv2.getTextSize(emotion, font, font_scale, font_thickness)
                    
                    # Calculate text position (top-left of the text string)
                    text_x = x1
                    text_y_baseline = y1 - padding # Try to place text baseline 5 pixels above bbox top

                    # Adjust text_y if it goes off screen at the top
                    if (text_y_baseline - text_h - baseline) < 0: # If top of text goes off screen
                        text_y_baseline = y1 + text_h + padding # Place text baseline below bbox top, inside

                    # Create an overlay for semi-transparent background
                    overlay = frame.copy()
                    
                    rect_x1 = text_x
                    rect_y1 = text_y_baseline - text_h - baseline
                    rect_x2 = text_x + text_w + padding
                    rect_y2 = text_y_baseline + baseline
                    
                    # Ensure background rectangle coordinates are within frame bounds
                    rect_x1 = max(0, rect_x1)
                    rect_y1 = max(0, rect_y1)
                    rect_x2 = min(frame.shape[1], rect_x2)
                    rect_y2 = min(frame.shape[0], rect_y2)

                    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1) # Black background
                    
                    alpha = 0.6 # Transparency factor
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                    # Draw text after background
                    cv2.putText(frame, emotion, (text_x, text_y_baseline), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                
                # Write original frame to video
                vid_writer.write(frame)
                
                # Show resized frame
                cv2.imshow('Video Processing', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Save final statistics and create visualizations
            self.save_statistics()
            self.create_visualizations()
            
            # Log completion
            self.logger.info("Video processing completed")
            self.logger.info(f"Total frames processed: {self.stats['total_frames']}")
            self.logger.info(f"Frames with faces: {self.stats['frames_with_faces']}")
            self.logger.info(f"Total faces detected: {self.stats['total_faces_detected']}")
            self.logger.info(f"Emotion distribution: {self.stats['emotions']}")

            vid_writer.release()
            cap.release()
            cv2.destroyAllWindows()

            # Если ни одного лица не найдено — возвращаем специальный результат
            if not faces_found or self.stats['frames_with_faces'] == 0:
                return {
                    'emotions_data': [],
                    'top_emotions': {k: 0 for k in self.DICT_EMO.values()},
                    'total_frames': 0,
                    'frames_with_faces': 0,
                    'total_faces_detected': 0,
                    'average_processing_time_ms': 0,
                    'message': 'На видео не найдено ни одного лица.'
                }
            # Иначе — стандартный результат
            return {
                'emotions_data': self.frame_emotions,
                'top_emotions': self.stats['emotions'], # Return raw counts
                'total_frames': self.stats['total_frames'],
                'frames_with_faces': self.stats['frames_with_faces'],
                'total_faces_detected': self.stats['total_faces_detected'],
                'average_processing_time_ms': np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
            } 