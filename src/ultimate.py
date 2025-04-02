# ultimate.py

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


def undistort_frame(frame, calibration_file):
    if os.path.exists(calibration_file):
        try:
            data = np.load(calibration_file)
            cameraMatrix = data['cameraMatrix']
            distCoeffs = data['distCoeffs']
            print("Calibration data loaded.")
        except Exception as e:
            print("Error loading calibration data:", e)
            return frame
        if cameraMatrix is not None and distCoeffs is not None:
            return cv2.undistort(frame, cameraMatrix, distCoeffs)
    print("Calibration file not found or corrupted.")
    return frame


class WorkingArea:
    def __init__(self, 
                 detection_params,
                 calibration_file=var.CALIBRATION_FILE, 
                 confirmation_callback=None,
                 parent=None):
        self.detection_params = detection_params
        self.calibration_file = calibration_file
        self.confirmation_callback = confirmation_callback
        self.parent = parent

    def objectDetection(self, frame):
        print("[DEBUG] Starting objectDetection...")
        corrected_frame = undistort_frame(frame, self.calibration_file)
        print("[DEBUG] Undistortion completed")
        # add
        gray = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2GRAY)
        blur_kernel = self.detection_params.get("BLUR_KERNEL", 5)
        blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        print("[DEBUG] Grayscale + blur completed")

        canny_low = self.detection_params.get("CANNY_LOW", 50)
        canny_high = self.detection_params.get("CANNY_HIGH", 150)
        edges = cv2.Canny(blur, canny_low, canny_high)
        print("[DEBUG] Canny edge detection completed")

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"[DEBUG] {len(contours)} contours found")

        frame_height, frame_width = corrected_frame.shape[:2]
        min_ratio = var.WORKING_AREA_MIN_SIZE_RATIO
        max_ratio = var.WORKING_AREA_MAX_SIZE_RATIO

        for cnt in contours:
            if cv2.contourArea(cnt) < 1:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (width, height), angle = rect
            short_side, long_side = sorted([width, height])
            short_ratio = short_side / min(frame_width, frame_height)
            long_ratio = long_side / max(frame_width, frame_height)
            print(f"[DEBUG] Found: short={short_side:.1f} ({short_ratio:.2f}), long={long_side:.1f} ({long_ratio:.2f})")

            if (min_ratio <= short_ratio <= max_ratio and min_ratio <= long_ratio <= max_ratio):
                print("[DEBUG] ✅ Candidate passed ratio check")
                box = cv2.boxPoints(rect).astype(np.int32)
                workArea_fr = corrected_frame.copy()
                cv2.drawContours(workArea_fr, [box], 0, (0, 0, 255), 2)

                user_confirmed = self.confirmation_callback(workArea_fr) if self.confirmation_callback else True
                if user_confirmed:
                    working_mask = np.zeros(corrected_frame.shape[:2], dtype="uint8")
                    cv2.drawContours(working_mask, [box], 0, 255, -1)
                    conversion_factor = 210.0 / short_side
                    self.detection_params["CONVERSION_FACTOR"] = conversion_factor
                    print("[DEBUG] ✅ Returning confirmed working area")
                    return workArea_fr, working_mask, conversion_factor
                else:
                    print("[DEBUG] User rejected candidate; trying next...")
        print("No candidate working area found within area constraints.")
        return None


class ObjectDetector:
    def __init__(self, identification_config_path=None, detection_params=None, confirmation_callback=None, parent=None):
        self.confirmation_callback = confirmation_callback
        self.parent = parent
        self.records = []
        self.next_track_id = 1
        self.tracks = []
        self.IDENTIFICATION_CONFIG = {}
        self.features_list = []
        self.mask = None
        self.detection_params = detection_params or {}
        self.calibration_file = var.CALIBRATION_FILE
        self.MAX_TRACK_LOST_FRAMES = 5
        self.load_detection_params(identification_config_path)

    def load_detection_params(self, config_path=None):
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, "r") as f:
                    data = json.load(f)
                    self.IDENTIFICATION_CONFIG = data.get("categories", {})
                    self.features_list = data.get("features", [])
                    self.mask = np.array(data["working_area_mask"], dtype=np.uint8) if "working_area_mask" in data else None
                    print(f"[CONFIG] Loaded identification config from: {config_path}")
        except Exception as e:
            print("[ERROR] Failed to load identification config:", e)

    def WorkingDetect(self, frame):
        print("[DEBUG] Starting objectDetection...")
        corrected_frame = undistort_frame(frame, self.calibration_file)
        print("[DEBUG] Undistortion completed")
        gray = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2GRAY)
        blur_kernel = self.detection_params.get("BLUR_KERNEL", 5)
        blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        print("[DEBUG] Grayscale + blur completed")
        canny_low = self.detection_params.get("CANNY_LOW", 50)
        canny_high = self.detection_params.get("CANNY_HIGH", 150)
        edges = cv2.Canny(blur, canny_low, canny_high)
        print("[DEBUG] Canny edge detection completed")
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"[DEBUG] {len(contours)} contours found")
        return contours, corrected_frame, gray, blur, edges

    def process_frame(self, corrected_frame, frame_counter, mask=None, mode="test"):
        mask_to_use = mask if mask is not None else self.mask
        if mask_to_use is None:
            print("[WARNING] No mask provided for detection!")
            mask_to_use = np.ones(corrected_frame.shape[:2], dtype=np.uint8)

        gray = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2GRAY)
        blur_kernel = self.detection_params.get("BLUR_KERNEL", 5)
        blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        canny_low = self.detection_params.get("CANNY_LOW", 50)
        canny_high = self.detection_params.get("CANNY_HIGH", 150)
        edges = cv2.Canny(blur, canny_low, canny_high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.detection_params.get("MIN_AREA", 100)
        max_area = self.detection_params.get("MAX_AREA", 10000)
        filtered = [(cnt, cv2.minAreaRect(cnt)) for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]
        if not filtered:
            return edges, corrected_frame.copy(), []

        detections, processed = [], []
        conv = self.detection_params.get("CONVERSION_FACTOR", 1.0)
        for contour, rect in filtered:
            center = self.get_center(rect)
            if not (0 <= center[1] < mask_to_use.shape[0] and 0 <= center[0] < mask_to_use.shape[1]):
                continue
            if any(cv2.pointPolygonTest(d['contour'], center, False) >= 0 for d in processed):
                continue

            features = self.compute_shape_features(contour, rect)
            avg_color = self.compute_average_color(corrected_frame, contour)
            hu_norm = np.linalg.norm(np.array(features["hu_moments"]))
            avg_color_val = np.mean(avg_color)
            width, height = rect[1]
            angle = rect[2]
            area = width * height
            aspect_ratio = width / height if height != 0 else 0

            center_mm = (center[0] * conv, center[1] * conv)
            width_mm = width * conv
            height_mm = height * conv
            area_mm2 = area * (conv ** 2)
            perimeter_mm = features["perimeter"] * conv
            avg_defect_depth_mm = features["avg_defect_depth"] * conv

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

            predicted_category = self.recognize_object(features_record, self.IDENTIFICATION_CONFIG, self.features_list)
            if predicted_category is None:
                predicted_category = "unknown"

            detection = {
                'center': center_mm,
                'width': width_mm,
                'height': height_mm,
                'angle': angle,
                'area': area_mm2,
                'aspect_ratio': aspect_ratio,
                'perimeter': perimeter_mm,
                'extent': features["extent"],
                'hu_moments': features["hu_moments"],
                'circularity': features["circularity"],
                'convexity_defects_count': features["convexity_defects_count"],
                'avg_defect_depth': avg_defect_depth_mm,
                'avg_color': avg_color,
                'predicted_category': predicted_category,
                'contour': contour
            }
            detections.append(detection)
            processed.append(detection)

            self.records.append({
                'frame': frame_counter,
                'center_x_mm': center_mm[0],
                'center_y_mm': center_mm[1],
                'width_mm': width_mm,
                'height_mm': height_mm,
                'angle_deg': angle,
                'area_mm2': area_mm2,
                'aspect_ratio': aspect_ratio,
                'perimeter_mm': perimeter_mm,
                'extent': features["extent"],
                'hu_norm': hu_norm,
                'circularity': features["circularity"],
                'defects_count': features["convexity_defects_count"],
                'defect_depth_mm': avg_defect_depth_mm,
                'avg_color': avg_color,
                'category': predicted_category
            })

        self.export_csv(os.path.join(var.APP_DIR, "detected_objects.csv"))
        objects_overlayed = corrected_frame.copy()
        detections = self.assign_ids(detections)

        for det in detections:
            cx, cy = int(det['center'][0] / conv), int(det['center'][1] / conv)
            label = f"ID {det['track_id']} | {det['predicted_category']} | {det['width']:.1f}x{det['height']:.1f} mm"
            cv2.putText(objects_overlayed, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(objects_overlayed, (cx, cy), 4, (0, 255, 0), -1)
            cv2.drawContours(objects_overlayed, [det['contour']], -1, (0, 255, 0), 2)

        return edges, objects_overlayed, detections

    def assign_ids(self, detections):
        track_centers = [track['last_center'] for track in self.tracks]
        detection_centers = [det['center'] for det in detections]
        assigned_detections = set()
        if track_centers and detection_centers:
            distances = np.linalg.norm(np.array(track_centers)[:, np.newaxis] - np.array(detection_centers), axis=2)
            row_ind, col_ind = linear_sum_assignment(distances)
            for r, c in zip(row_ind, col_ind):
                if distances[r, c] < 50:
                    self.tracks[r]['last_center'] = detections[c]['center']
                    self.tracks[r]['lost_frames'] = 0
                    detections[c]['track_id'] = self.tracks[r]['id']
                    if detections[c]['predicted_category'] != "unknown":
                        self.tracks[r]['object_name'] = detections[c]['predicted_category']
                    assigned_detections.add(c)

        for i, det in enumerate(detections):
            if i not in assigned_detections:
                det['track_id'] = self.next_track_id
                self.tracks.append({
                    'id': self.next_track_id,
                    'last_center': det['center'],
                    'lost_frames': 0,
                    'object_name': det['predicted_category']
                })
                self.next_track_id += 1

        for track in self.tracks:
            if not any(det.get('track_id') == track['id'] for det in detections):
                track['lost_frames'] += 1
        self.tracks = [t for t in self.tracks if t['lost_frames'] <= self.MAX_TRACK_LOST_FRAMES]

        for det in detections:
            if det.get('predicted_category') == "unknown":
                for track in self.tracks:
                    if track['id'] == det.get('track_id') and track.get('object_name'):
                        det['predicted_category'] = track['object_name']
        return detections

    def export_csv(self, path):
        df = pd.DataFrame(self.records)
        df.to_csv(path, index=False)
        print(f"[EXPORT] Saved {len(df)} records to {path}")

    def get_center(self, rect):
        return (int(rect[0][0]), int(rect[0][1]))

    def recognize_object(self, features_record, IDENTIFICATION_CONFIG, features_list):
        for category, conf in IDENTIFICATION_CONFIG.items():
            params = conf.get("parameters", {})
            if all(param in features_record and param in params and params[param]["min"] <= features_record[param] <= params[param]["max"] for param in features_list):
                return category
        return None

    def compute_shape_features(self, contour, rect):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        extent = area / (w * h) if w * h > 0 else 0
        moments = cv2.HuMoments(cv2.moments(contour)).flatten().tolist()
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = None
        if hull is not None and len(hull) > 3:
            hull_sorted = np.sort(hull.flatten())
            if np.array_equal(hull.flatten(), hull_sorted):
                defects = cv2.convexityDefects(contour, hull)
        defect_depths = [d[0][3] / 256.0 for d in defects] if defects is not None else []
        avg_defect_depth = np.mean(defect_depths) if defect_depths else 0
        circularity = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
        return {
            "area": area,
            "perimeter": perimeter,
            "extent": extent,
            "hu_moments": moments,
            "circularity": circularity,
            "convexity_defects_count": len(defect_depths),
            "avg_defect_depth": avg_defect_depth
        }

    def compute_average_color(self, frame, contour):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_color = cv2.mean(frame, mask=mask)
        return (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))
