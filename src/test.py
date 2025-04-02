# refactor.py


import cv2
import numpy as np
import random
import math
import pandas as pd
import json
import os
from scipy.optimize import linear_sum_assignment
from PyQt5.QtWidgets import QMessageBox
import variables as var


#correct lens distortion in a frame using the camera calibration data
# Example usage:
# corrected_frame = undistort_frame(frame, CALIBRATION_FILE)
def undistort_frame(frame, CALIBRATION_FILE):

    if os.path.exists(var.CALIBRATION_FILE):
        try:
            data = np.load(var.CALIBRATION_FILE)
            cameraMatrix = data['cameraMatrix']
            distCoeffs = data['distCoeffs']
            print("Calibration data loaded.")
        except Exception as e:
            print("Error loading calibration data:", e)
            return frame
        if cameraMatrix is not None and distCoeffs is not None:
            corrected_frame = cv2.undistort(frame, cameraMatrix, distCoeffs)
            return corrected_frame
    print("Calibration file not found or corrupted.")
    return frame

class ObjectDetector:
    def __init__(self, 
                 calibration_file=var.CALIBRATION_FILE, 
                 blur_kernel=var.BLUR_KERNEL, 
                 canny_low=var.CANNY_LOW, 
                 canny_high=var.CANNY_HIGH,
                 confirmation_callback=None,
                 parent=None):
        self.calibration_file = calibration_file
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.confirmation_callback = confirmation_callback
        self.parent = parent

    def objectDetection(self, frame):
        """
        Process the frame: apply lens distortion correction, detect all candidate working areas
        using minAreaRect, and return the first confirmed one as overlay image, binary mask,
        and conversion factor. If no candidate is confirmed, return None.
        """
        print("[DEBUG] Starting objectDetection...")

        import variables as var

        # Step 1: Correct lens distortion
        corrected_frame = undistort_frame(frame, self.calibration_file)
        print("[DEBUG] Undistortion completed")
        
        # add corrected_frame scaling

        # Step 2: Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        print("[DEBUG] Grayscale + blur completed")


        # Step 3: Apply Canny edge detection
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        print("[DEBUG] Canny edge detection completed")


        # Step 4: Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"[DEBUG] {len(contours)} contours found")

        return contours, corrected_frame, gray, blur, edges
    
    def process_frame(self, frameUndist, frame_counter, mask=None, mode="test"):
        self.load_config()
        edges, objects_vis, rects, contours = process_frame_canny(
            frameUndist, self.BLUR_KERNEL, self.CANNY_LOW, self.CANNY_HIGH, self.MIN_AREA, self.MAX_AREA, mask
        )
        # Safety check: if no valid contours/rectangles found, return defaults
        if rects is None or len(rects) == 0:
            objects_overlayed = frameUndist.copy()
            return edges, objects_overlayed, []

        detections = []
        processed = []
        for idx, rect in enumerate(rects):
            contour = contours[idx]
            center = get_center(rect)
            if any(cv2.pointPolygonTest(d['contour'], center, False) >= 0 for d in processed):
                continue
            features = compute_shape_features(contour, rect)
            avg_color = compute_average_color(frameUndist, contour)
            hu_norm = np.linalg.norm(np.array(features["hu_moments"]))
            avg_color_val = np.mean(avg_color)
            width, height = rect[1]
            angle = rect[2]
            area = width * height
            aspect_ratio = width / height if height != 0 else 0
            features_record = {
                "aspect_ratio": aspect_ratio,
                "area": area,
                "perimeter": features["perimeter"],
                "extent": features["extent"],
                "hu_moments_norm": hu_norm,
                "circularity": features["circularity"],
                "convexity_defects_count": features["convexity_defects_count"],
                "avg_defect_depth": features["avg_defect_depth"],
                "avg_color_val": avg_color_val
            }
            predicted_category = recognize_object(features_record, self.IDENTIFICATION_CONFIG, self.features_list)
            if predicted_category is None:
                predicted_category = "unknown"
            detection = {
                'center': center,
                'width': width,
                'height': height,
                'angle': angle,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'perimeter': features["perimeter"],
                'extent': features["extent"],
                'hu_moments': features["hu_moments"],
                'circularity': features["circularity"],
                'is_convex': features["is_convex"],
                'convexity_defects_count': features["convexity_defects_count"],
                'avg_defect_depth': features["avg_defect_depth"],
                'avg_color': avg_color,
                'predicted_category': predicted_category,
                'contour': contour
            }
            detections.append(detection)
            processed.append(detection)

        # Tracking algorithm
        if not self.tracks:
            for det in detections:
                self.tracks.append({
                    'id': self.next_track_id,
                    'last_center': det['center'],
                    'lost_frames': 0,
                    'object_name': det['predicted_category'] if det['predicted_category'] != "unknown" else ""
                })
                det['track_id'] = self.next_track_id
                self.next_track_id += 1
        else:
            track_centers = [t['last_center'] for t in self.tracks]
            detection_centers = [d['center'] for d in detections]
            if track_centers and detection_centers:
                distances = np.linalg.norm(np.array(track_centers)[:, np.newaxis] - np.array(detection_centers), axis=2)
                row_ind, col_ind = linear_sum_assignment(distances)
                assigned = set()
                for r, c in zip(row_ind, col_ind):
                    if distances[r, c] < MAX_DISTANCE:
                        self.tracks[r]['last_center'] = detections[c]['center']
                        if detections[c]['predicted_category'] not in [None, "", "unknown"]:
                            self.tracks[r]['object_name'] = detections[c]['predicted_category']
                        detections[c]['track_id'] = self.tracks[r]['id']
                        assigned.add(c)
                for i, det in enumerate(detections):
                    if i not in assigned:
                        self.tracks.append({
                            'id': self.next_track_id,
                            'last_center': det['center'],
                            'lost_frames': 0,
                            'object_name': det['predicted_category'] if det['predicted_category'] not in [None, "", "unknown"] else ""
                        })
                        det['track_id'] = self.next_track_id
                        self.next_track_id += 1
                for track in self.tracks:
                    if not any(d.get('track_id') == track['id'] for d in detections):
                        track['lost_frames'] = track.get('lost_frames', 0) + 1
                    else:
                        track['lost_frames'] = 0
                self.tracks = [t for t in self.tracks if t['lost_frames'] < MAX_LOST_FRAMES]
                for det in detections:
                    if det.get('predicted_category') in [None, "unknown"]:
                        for t in self.tracks:
                            if t['id'] == det.get('track_id') and t.get('object_name') not in [None, "", "unknown"]:
                                det['predicted_category'] = t['object_name']
                                break

        # Create overlayed frame with annotations
        # Define conversion factor: adjust this value to convert pixels to mm as needed
        conversion_factor = 1.0  # Например, 1.0, если 1 пиксель = 1 мм


 
        # Create overlayed frame with ID, category, size (in mm)
        objects_overlayed = frameUndist.copy()
        for det in detections:
            cx, cy = det['center']
            width_mm = det['width'] * CONVERSION_FACTOR
            height_mm = det['height'] * CONVERSION_FACTOR
            label = f"ID {det.get('track_id', '?')} | {det['predicted_category']} | {width_mm:.1f}x{height_mm:.1f} mm"
            cv2.putText(objects_overlayed, label, (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(objects_overlayed, (cx, cy), 4, (0, 255, 0), -1)
            cv2.drawContours(objects_overlayed, [det['contour']], -1, (0, 255, 0), 2)
        
        return edges, objects_overlayed, detections


    def send_robot_data(detections, conversion_factor, robot_comm, category_mapping=None):
        if category_mapping is None:
            category_mapping = {
                "circular_white": 1,
                "circular_pink": 2,
                "circular_black": 3,
                "rhombus_white": 4,
                "rhombus_pink": 5,
                "rhombus_black": 6,
            }
        for det in detections:
            x_mm = det['center'][0] * conversion_factor
            y_mm = det['center'][1] * conversion_factor
            width_mm = det['width'] * conversion_factor
            height_mm = det['height'] * conversion_factor
            angle = det['angle']
            category_code = category_mapping.get(det['predicted_category'], 0)
            robot_comm.send_data(det.get('track_id', 0), x_mm, y_mm, width_mm, height_mm, angle, category_code)

    def get_center(rect):
        center = rect[0]
        return (int(center[0]), int(center[1]))

    def recognize_object(features_record, IDENTIFICATION_CONFIG, features_list):
        predicted_category = None
        for category, conf in IDENTIFICATION_CONFIG.items():
            params = conf.get("parameters", {})
            match = True
            for param in features_list:
                if param not in features_record or param not in params:
                    continue
                if not (params[param]["min"] <= features_record[param] <= params[param]["max"]):
                    match = False
                    break
            if match:
                predicted_category = category
                break
        return predicted_category

    def compute_average_color(frameUndist, contour):
        mask = np.zeros(frameUndist.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_color = cv2.mean(frameUndist, mask=mask)
        return (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))