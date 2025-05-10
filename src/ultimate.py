# ultimate.py

import cv2
import numpy as np
import random
import math
import pandas as pd
import json
import os
import logging # For logging
from scipy.optimize import linear_sum_assignment
from PyQt5.QtWidgets import QMessageBox
import variables as var


def undistort_frame(frame, calibration_file):
    if os.path.exists(calibration_file):
        try:
            data = np.load(calibration_file)
            cameraMatrix = data['cameraMatrix']
            distCoeffs = data['distCoeffs']
            #print("Calibration data loaded.")
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
        # Add a place to store the homography matrix if needed within this class instance,
        # though it's better to return it and store it in MainWindow.
        self.homography_matrix = None

    def objectDetection(self, frame):
        print("[DEBUG] Starting objectDetection for Working Area...")
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

        frame_height, frame_width = corrected_frame.shape[:2]
        # Adjust these ratios if needed, they now apply to the contour area
        # to find a candidate for the A4 sheet.
        min_area_ratio_for_A4 = 0.1 # Example: A4 sheet should occupy at least 10% of frame area
        max_area_ratio_for_A4 = 0.99 # Example: And no more than 80%
        min_contour_area = frame_height * frame_width * min_area_ratio_for_A4
        max_contour_area = frame_height * frame_width * max_area_ratio_for_A4


        # Helper function to order points: tl, tr, br, bl
        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect

        for cnt in contours:
            contour_area = cv2.contourArea(cnt)
            if not (min_contour_area <= contour_area <= max_contour_area):
                continue # Filter by overall area first

            epsilon = 0.02 * cv2.arcLength(cnt, True) # Epsilon for approxPolyDP
            approx_polygon = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx_polygon) == 4:
                print("[DEBUG] Found a 4-sided polygon candidate for A4 sheet.")
                
                # Order the points of the polygon: top-left, top-right, bottom-right, bottom-left
                src_points_px = order_points(approx_polygon.reshape(4, 2))

                # Define the destination points corresponding to the A4 sheet corners in millimeters
                # We need to decide if the A4 is portrait or landscape in the destination.
                # Let's assume we want to map it to a "portrait" orientation in mm.
                # The order of dst_points must match the order of src_points_px (tl, tr, br, bl)
                dst_points_mm = np.array([
                    [0, 0],                            # Top-left
                    [var.A4_WIDTH_MM -1, 0],           # Top-right
                    [var.A4_WIDTH_MM -1, var.A4_HEIGHT_MM -1], # Bottom-right
                    [0, var.A4_HEIGHT_MM -1]           # Bottom-left
                ], dtype="float32")
                
                # To handle potential landscape orientation of the detected A4 sheet in the image:
                # Check aspect ratio of the detected polygon in pixels
                # (this is a simplified check, more robust would be to check side lengths)
                poly_rect_px = cv2.boundingRect(approx_polygon) # x, y, w, h of non-rotated bounding box
                _, _, w_px, h_px = poly_rect_px
                
                # If detected polygon is wider than tall, assume it's landscape in image
                # and adjust dst_points_mm accordingly if we want to map it consistently.
                # Or, more robustly, calculate the lengths of the sides of src_points_px
                side1_len_px = np.linalg.norm(src_points_px[0] - src_points_px[1]) # Top side
                side2_len_px = np.linalg.norm(src_points_px[1] - src_points_px[2]) # Right side

                # If the detected "width" (side1_len_px) is longer than "height" (side2_len_px),
                # it means the A4 sheet is likely landscape in the image.
                # We then map its longer dimension to A4_HEIGHT_MM and shorter to A4_WIDTH_MM.
                if side1_len_px > side2_len_px: # Detected as landscape
                    print("[DEBUG] Detected A4 sheet appears landscape in image. Adjusting dst_points_mm.")
                    dst_points_mm_adjusted = np.array([
                        [0, 0],                                 # Top-left
                        [var.A4_HEIGHT_MM - 1, 0],              # Top-right (longer side)
                        [var.A4_HEIGHT_MM - 1, var.A4_WIDTH_MM - 1], # Bottom-right
                        [0, var.A4_WIDTH_MM - 1]                # Bottom-left (shorter side)
                    ], dtype="float32")
                    # Compute homography: pixels to millimeters
                    homography_matrix, status = cv2.findHomography(src_points_px, dst_points_mm_adjusted)
                else: # Detected as portrait or square-ish
                    print("[DEBUG] Detected A4 sheet appears portrait or square in image.")
                    # Compute homography: pixels to millimeters
                    homography_matrix, status = cv2.findHomography(src_points_px, dst_points_mm)

                if homography_matrix is None:
                    print("[WARNING] Could not compute homography matrix. Skipping this candidate.")
                    continue

                # Prepare frame for confirmation
                workArea_fr_confirm = corrected_frame.copy()
                cv2.drawContours(workArea_fr_confirm, [approx_polygon], 0, (0, 255, 0), 2) # Green polygon

                # Show confirmation dialog
                user_confirmed = self.confirmation_callback(workArea_fr_confirm) if self.confirmation_callback else True
                
                if user_confirmed:
                    working_mask = np.zeros(corrected_frame.shape[:2], dtype="uint8")
                    cv2.drawContours(working_mask, [approx_polygon], 0, 255, -1)
                    
                    self.homography_matrix = homography_matrix # Store if needed here, but better to return
                    print("[DEBUG] ✅ Returning confirmed working area mask and homography matrix.")
                    # Return: overlay_frame_for_confirmation, mask, homography_matrix
                    return workArea_fr_confirm, working_mask, homography_matrix
                else:
                    print("[DEBUG] User rejected candidate; trying next...")
        
        print("[WARNING] No suitable 4-sided polygon for A4 sheet found or confirmed.")
        return None, None, None # Return None for all if no area is confirmed

class ObjectDetector:
    def __init__(self, identification_config_path=None, detection_params=None, confirmation_callback=None, parent=None, homography_matrix=None):
        # confirmation_callback and parent are not used within ObjectDetector itself based on current usage
        # self.confirmation_callback = confirmation_callback
        # self.parent = parent
        self.records = []
        self.next_track_id = 1
        self.tracks = []
        self.IDENTIFICATION_CONFIG = {}
        self.features_list = []
        #self.mask = None # Mask loaded from config (if any)
        self.detection_params = detection_params or {}
        self.calibration_file = var.CALIBRATION_FILE
        self.MAX_TRACK_LOST_FRAMES = var.MAX_LOST_FRAMES # Use variable
        self.homography_matrix_to_mm = homography_matrix # Store the matrix
        if self.homography_matrix_to_mm is None:
            print("[WARNING] ObjectDetector initialized without a homography matrix. Measurements will be in pixels or inaccurate.")
        self.load_identification_config(identification_config_path) # Renamed method

    def load_identification_config(self, config_path=None): # Renamed method
        """Loads object identification parameters (categories, features) from a JSON config file."""
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, "r") as f:
                    data = json.load(f)
                    self.IDENTIFICATION_CONFIG = data.get("categories", {})
                    self.features_list = data.get("features", [])
                    # Only load mask from config if it exists and is needed as a fallback
                    # self.mask = np.array(data["working_area_mask"], dtype=np.uint8) if "working_area_mask" in data else None
                    print(f"[CONFIG] Loaded identification config from: {config_path}")
                    # print(f"Loaded features: {self.features_list}")
                    # print(f"Loaded categories: {list(self.IDENTIFICATION_CONFIG.keys())}")
            else:
                print("[WARNING] Identification config file not found or not provided. Using empty config.")
                self.IDENTIFICATION_CONFIG = {}
                self.features_list = []

        except Exception as e:
            print(f"[ERROR] Failed to load identification config from '{config_path}': {e}")
            self.IDENTIFICATION_CONFIG = {}
            self.features_list = []

    # ... (WorkingDetect method can remain if used elsewhere) ...

    def process_frame(self, corrected_frame: np.ndarray, frame_counter: int, working_area_mask: np.ndarray | None = None, mode: str = "process") -> tuple[np.ndarray, np.ndarray, list]:
        """
        Processes a single frame to detect, identify, and track objects within the working area.

        Args:
            corrected_frame: The undistorted input frame (BGR).
            frame_counter: The current frame number.
            working_area_mask: A binary mask (np.uint8) defining the valid detection zone.
                                Pixels with value 255 are considered inside the working area.
                                If None, detection occurs in the full frame.
            mode: Processing mode (currently 'process' or 'test' - potentially unused).

        Returns:
            A tuple containing:
            - masked_edges (np.ndarray): The Canny edges image masked by the working area (grayscale).
            - objects_overlayed (np.ndarray): The input frame with detected objects highlighted (BGR).
            - final_detections (list): A list of dictionaries, each representing a detected object
                                        with its properties and tracking ID.
        """
        if corrected_frame is None or corrected_frame.size == 0:
            print("[ERROR] process_frame received an invalid frame.")
            # Return empty/default values matching the expected tuple structure
            empty_edges = np.zeros((100, 100), dtype=np.uint8) # Placeholder shape
            empty_overlay = np.zeros((100, 100, 3), dtype=np.uint8) # Placeholder shape
            return empty_edges, empty_overlay, []

        frame_height, frame_width = corrected_frame.shape[:2]

        # --- 1. Determine and Validate Mask ---
        mask_to_use = working_area_mask
        if mask_to_use is None:
            print("[WARNING] No working area mask provided. Detecting in full frame.")
            mask_to_use = np.ones((frame_height, frame_width), dtype=np.uint8) * 255
        elif mask_to_use.shape != (frame_height, frame_width):
            print(f"[ERROR] Provided mask shape {mask_to_use.shape} does not match frame shape {(frame_height, frame_width)}! Using full frame.")
            mask_to_use = np.ones((frame_height, frame_width), dtype=np.uint8) * 255
        elif not np.any(mask_to_use):
             print("[WARNING] Provided working area mask is completely black! No detection possible.")
             # Return early as findContours will yield nothing meaningful
             masked_edges = np.zeros((frame_height, frame_width), dtype=np.uint8)
             return masked_edges, corrected_frame.copy(), []


        # --- 2. Preprocessing ---
        gray = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2GRAY)

        # Get parameters safely with defaults
        blur_kernel_size = self.detection_params.get("BLUR_KERNEL", 5)
        # Ensure blur kernel is odd and positive
        blur_kernel_size = max(1, int(blur_kernel_size) // 2 * 2 + 1)
        canny_low = self.detection_params.get("CANNY_LOW", 10)
        canny_high = self.detection_params.get("CANNY_HIGH", 50)

        blur = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
        edges = cv2.Canny(blur, canny_low, canny_high)

        # --- 3. Apply Mask to Edges ---
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask_to_use)

        # --- 4. Find and Filter Contours ---
        contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(f"[DEBUG] Frame {frame_counter}: Found {len(contours)} raw contours within mask.")

        min_area = self.detection_params.get("MIN_AREA", 100)
        max_area = self.detection_params.get("MAX_AREA", 50000) # Increased default max

        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                valid_contours.append(cnt)

        if not valid_contours:
            # No objects passed filters, return masked edges and original frame
            # print(f"[DEBUG] Frame {frame_counter}: No contours passed area filtering ({min_area}-{max_area}).")
            return masked_edges, corrected_frame.copy(), []

        # print(f"[DEBUG] Frame {frame_counter}: {len(valid_contours)} contours passed area filtering.")

        # --- 5. Process Valid Contours ---
        detections = []
        processed_centers = [] # Keep track of centers of processed objects this frame to avoid duplicates if needed

        for contour in valid_contours:
            try:
                rect_px = cv2.minAreaRect(contour) # This is still in pixels
                center_px_tuple = self.get_center(rect_px) # (x_px, y_px)
                (width_px, height_px), angle_deg = rect_px[1], rect_px[2]



                # Calculate features
                features = self.compute_shape_features(contour, rect_px)
                avg_color = self.compute_average_color(corrected_frame, contour)
                hu_norm = np.linalg.norm(np.array(features["hu_moments"]))
                avg_color_val = np.mean(avg_color) # Average intensity across BGR

# --- Transform key points to millimeters using homography ---
                center_mm = (0,0)
                width_mm = 0
                height_mm = 0
                area_mm2 = 0
                perimeter_mm = 0 # This will be harder to transform directly, better to calculate from transformed points.
                avg_defect_depth_mm = 0 # Same as perimeter

                if self.homography_matrix_to_mm is not None:
                    # Transform center point
                    center_px_arr = np.array([[center_px_tuple]], dtype="float32") # Needs to be [[[x,y]]] for perspectiveTransform
                    transformed_center_arr = cv2.perspectiveTransform(center_px_arr, self.homography_matrix_to_mm)
                    if transformed_center_arr is not None and transformed_center_arr.size > 0:
                        center_mm = (transformed_center_arr[0][0][0], transformed_center_arr[0][0][1])
                    
                    # To get width_mm and height_mm accurately, transform the box points
                    box_points_px = cv2.boxPoints(rect_px) # 4 corner points of the rotated rect in pixels
                    
                    # Ensure box_points_px is in the correct shape for perspectiveTransform: (N, 1, 2)
                    box_points_px_reshaped = box_points_px.reshape(-1, 1, 2).astype(np.float32)
                    
                    transformed_box_points_mm_arr = cv2.perspectiveTransform(box_points_px_reshaped, self.homography_matrix_to_mm)

                    if transformed_box_points_mm_arr is not None and len(transformed_box_points_mm_arr) == 4:
                        # Calculate distances between transformed points to get width_mm and height_mm
                        # Points are typically tl, tr, br, bl (or a rotation) after boxPoints
                        # Let's calculate lengths of two adjacent sides of the transformed quadrilateral
                        side1_mm = np.linalg.norm(transformed_box_points_mm_arr[0][0] - transformed_box_points_mm_arr[1][0])
                        side2_mm = np.linalg.norm(transformed_box_points_mm_arr[1][0] - transformed_box_points_mm_arr[2][0])
                        
                        # Assign width and height. Note: `angle_deg` from minAreaRect refers to pixel-space rotation.
                        # The concept of "width" and "height" might be relative to the object's orientation in mm space.
                        # For simplicity, we can sort them if `angle_deg` is not directly transferable.
                        # However, width_px and height_px from rect_px were already sorted by minAreaRect if angle was > 45 or < -45.
                        # Let's assume the order corresponds, or you might need a more robust way to define which is width/height
                        # if the object is significantly rotated in the mm-plane.
                        # For now, let's use the pixel-derived width/height and scale them by an average factor from box points,
                        # or directly use side1_mm, side2_mm. Using side1_mm and side2_mm is more accurate.
                        
                        # We need to decide if width_px corresponded to side1_mm or side2_mm.
                        # A simple way: assign them based on which pixel dimension was larger.
                        if width_px >= height_px: # Assuming width_px was the longer side from minAreaRect
                            width_mm = max(side1_mm, side2_mm)
                            height_mm = min(side1_mm, side2_mm)
                        else:
                            width_mm = min(side1_mm, side2_mm)
                            height_mm = max(side1_mm, side2_mm)

                        # Area in mm^2 can be calculated from the transformed box points (e.g., shoelace formula or cv2.contourArea)
                        area_mm2 = cv2.contourArea(transformed_box_points_mm_arr)
                        # Perimeter in mm
                        perimeter_mm = cv2.arcLength(transformed_box_points_mm_arr.astype(np.float32), closed=True)

                    # Features like avg_defect_depth would also need their defining points transformed
                    # For now, let's keep avg_defect_depth calculation in pixel space and then apply an *average* scale,
                    # or transform the defect points if crucial. Let's scale for now.
                    # This is less accurate than transforming points.
                    pixel_features = self.compute_shape_features(contour, rect_px) # rect_px is needed here
                    
                    # An approximate scale for defect depth
                    # This is a simplification; ideally, you'd transform defect points.
                    if width_px > 0 :
                        avg_scale_for_defects = width_mm / width_px
                        avg_defect_depth_mm = pixel_features["avg_defect_depth"] * avg_scale_for_defects
                    else:
                        avg_defect_depth_mm = 0


                else: # Fallback if no homography matrix
                    print("[WARNING] No homography matrix available, using pixel values for measurements.")
                    center_mm = center_px_tuple # Effectively pixels
                    width_mm = width_px
                    height_mm = height_px
                    pixel_features = self.compute_shape_features(contour, rect_px)
                    area_mm2 = pixel_features["area"] # pixels^2
                    perimeter_mm = pixel_features["perimeter"] # pixels
                    avg_defect_depth_mm = pixel_features["avg_defect_depth"] # pixels

                # Prepare features for recognition
                
                aspect_ratio_px = width_px / height_px if height_px != 0 else 0
                hu_norm = np.linalg.norm(np.array(pixel_features["hu_moments"]))
                avg_color = self.compute_average_color(corrected_frame, contour)


                features_for_recognition = {
                    "aspect_ratio": aspect_ratio_px,
                    "circularity": pixel_features["circularity"],
                    "extent": pixel_features["extent"],
                    "hu_norm": hu_norm,
                    "convexity_defects_count": pixel_features["convexity_defects_count"],
                    "avg_color_r": avg_color[2],
                    "avg_color_g": avg_color[1],
                    "avg_color_b": avg_color[0]
                }



                # Recognize object category
                predicted_category = self.recognize_object(features_for_recognition, self.IDENTIFICATION_CONFIG, self.features_list)
                if predicted_category is None:
                    predicted_category = "unknown" # Default category

                # Build detection dictionary
                detection_data = {
                    'center': center_mm, # Now (x, y) in mm
                    'width': width_mm,   # width in mm
                    'height': height_mm, # height in mm
                    'angle': angle_deg,  # angle from pixel-space minAreaRect, interpretation might need care
                    'area': area_mm2,    # area in mm^2
                    'aspect_ratio': aspect_ratio_px, # or width_mm / height_mm if preferred
                    'perimeter': perimeter_mm, # perimeter in mm
                    'extent': pixel_features["extent"],
                    'hu_moments': pixel_features["hu_moments"],
                    'circularity': pixel_features["circularity"],
                    'convexity_defects_count': pixel_features["convexity_defects_count"],
                    'avg_defect_depth': avg_defect_depth_mm, # avg defect depth, approximately in mm
                    'avg_color': avg_color,
                    'predicted_category': predicted_category,
                    'contour': contour # Original pixel contour for drawing
                }
                
                detections.append(detection_data)
                processed_centers.append(center_px_tuple) # Add center to processed list

                # Record data for potential CSV export
                # Using mm for physical measurements, unitless for ratios/counts
                self.records.append({

                    'frame': frame_counter,
                    'center_x_mm': center_mm[0],
                    'center_y_mm': center_mm[1],
                    'width_mm': width_mm,
                    'height_mm': height_mm,
                    'angle_deg': angle_deg,
                    'area_mm2': area_mm2,
                    'aspect_ratio': aspect_ratio_px,
                    'perimeter_mm': perimeter_mm,
                    'extent': pixel_features["extent"],
                    'hu_norm': hu_norm,
                    'circularity': pixel_features["circularity"],
                    'defects_count': pixel_features["convexity_defects_count"],
                    'defect_depth_mm': avg_defect_depth_mm,
                    'avg_color_r': avg_color[2],
                    'avg_color_g': avg_color[1],
                    'avg_color_b': avg_color[0],
                    'category': predicted_category
                })
            except Exception as e_contour:
                 print(f"[ERROR] Frame {frame_counter}: Failed processing contour: {e_contour}")
                 continue # Skip this contour

        # --- 6. Assign Tracking IDs ---
        # This modifies the 'detections' list in-place, adding 'track_id'
        final_detections = self.assign_ids(detections)

        # --- 7. Export Data (Optional) ---
        # Consider exporting less frequently or based on a condition
        if frame_counter % 30 == 0 and self.records: # Example: Export every 30 frames
             self.export_csv(os.path.join(var.APP_DIR, "detected_objects.csv"))


        # --- 8. Draw Overlays ---
        objects_overlayed = corrected_frame.copy()
        for det in final_detections:
            try:


                # Draw bounding box (optional, using boxPoints)
                # box = cv2.boxPoints(( (center_x_px, center_y_px), (det['width']/conversion_factor, det['height']/conversion_factor), det['angle'] ))
                # box = np.intp(box) # np.intp is recommended for indexing
                # cv2.drawContours(objects_overlayed, [box], 0, (255, 0, 0), 1) # Blue box

                    # Get pixel center from the original contour's bounding rect for label positioning
                rect_for_drawing = cv2.minAreaRect(det['contour'])
                center_x_px_draw, center_y_px_draw = int(rect_for_drawing[0][0]), int(rect_for_drawing[0][1])
                # Draw contour
                cv2.drawContours(objects_overlayed, [det['contour']], -1, (0, 255, 0), 2) # Green contour

                # Draw center point
                cv2.circle(objects_overlayed, (center_x_px_draw, center_y_px_draw), 4, (0, 0, 255), -1) # Red center

                # Prepare and draw label
                label = f"ID:{det.get('track_id','?')} {det.get('predicted_category','unk')} {det.get('width',0):.1f}x{det.get('height',0):.1f}"
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_origin = (center_x_px_draw + 8, center_y_px_draw + label_height // 2) # Position relative to center
                # Background rectangle for label
                cv2.rectangle(objects_overlayed, (label_origin[0] - 2, label_origin[1] - label_height - baseline + 2),
                              (label_origin[0] + label_width + 2, label_origin[1] + baseline -2), (0, 0, 0), cv2.FILLED)
                cv2.putText(objects_overlayed, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            except Exception as e_draw:
                 print(f"[ERROR] Frame {frame_counter}: Failed drawing overlay for detection {det.get('track_id', '?')}: {e_draw}")


        # --- 9. Return Results ---
        return masked_edges, objects_overlayed, final_detections


    def assign_ids(self, current_detections: list) -> list:
        """
        Assigns track IDs to current detections based on proximity to previous tracks.
        Uses the Hungarian algorithm for optimal assignment. Updates internal track list.

        Args:
            current_detections: List of detection dictionaries for the current frame.
                                Each dictionary must have a 'center' key (in mm).

        Returns:
            The input list of detections, with a 'track_id' key added to each dictionary.
            Also updates `self.tracks` and `self.next_track_id`.
        """
        # Get centers from active tracks and current detections (ensure they are in mm)
        track_centers_mm = np.array([track['last_center_mm'] for track in self.tracks]) if self.tracks else np.empty((0, 2))
        detection_centers_mm = np.array([det['center'] for det in current_detections]) if current_detections else np.empty((0, 2))

        assigned_detection_indices = set()
        matched_track_indices = set()

        # --- Matching using Hungarian Algorithm ---
        if track_centers_mm.size > 0 and detection_centers_mm.size > 0:
            # Calculate cost matrix (Euclidean distance in mm)
            # distance shape: (num_tracks, num_detections)
            distances_mm  = np.linalg.norm(track_centers_mm[:, np.newaxis, :] - detection_centers_mm[np.newaxis, :, :], axis=2)

            # Apply Hungarian algorithm
            track_indices, detection_indices = linear_sum_assignment(distances_mm)

            # Filter matches based on max distance threshold
            max_matching_distance_mm = getattr(var, 'MAX_DISTANCE_MM', 10.0)
            for r, c in zip(track_indices, detection_indices):
                if distances_mm[r, c] < max_matching_distance_mm:
                    track = self.tracks[r]
                    detection = current_detections[c]

                    # Update track state
                    track['last_center_mm'] = detection['center']
                    track['lost_frames'] = 0
                    detection['track_id'] = track['id'] # Assign existing ID

                    # Update track's known category if current detection is more specific
                    if detection['predicted_category'] != "unknown":
                        track['object_name'] = detection['predicted_category']

                    assigned_detection_indices.add(c)
                    matched_track_indices.add(r)
                    # print(f"[TRACK] Matched Track {track['id']} to Detection {c} (Dist: {distances[r,c]:.2f}mm)")



        for i, det in enumerate(current_detections):
            if i not in assigned_detection_indices:
                det['track_id'] = self.next_track_id
                self.tracks.append({
                    'id': self.next_track_id,
                    'last_center_mm': det['center'], # Store center in mm
                    'lost_frames': 0,
                    'object_name': det['predicted_category'] # Initial name
                })
                # print(f"[TRACK] Created New Track {self.next_track_id} for Detection {i}")
                self.next_track_id += 1


        # --- Handle Unmatched Tracks (Lost Frames) ---
        indices_to_remove = []
        for i, track in enumerate(self.tracks):
             if i not in matched_track_indices:
                 track['lost_frames'] += 1
                 # print(f"[TRACK] Track {track['id']} lost frame {track['lost_frames']}/{self.MAX_TRACK_LOST_FRAMES}")
                 if track['lost_frames'] > self.MAX_TRACK_LOST_FRAMES:
                      indices_to_remove.append(i)
                      # print(f"[TRACK] Removed Track {track['id']} (lost too many frames)")


        # --- Remove Lost Tracks ---
        # Iterate backwards to avoid index shifting issues
        for i in sorted(indices_to_remove, reverse=True):
            del self.tracks[i]


        # --- Fill 'unknown' categories with known track names (if available) ---
        # This helps maintain consistency if recognition flickers
        for det in current_detections:
            if det.get('predicted_category') == "unknown":
                track_id = det.get('track_id')
                if track_id is not None:
                    # Find the corresponding track
                    matching_track = next((t for t in self.tracks if t['id'] == track_id), None)
                    if matching_track and matching_track.get('object_name') and matching_track['object_name'] != "unknown":
                        det['predicted_category'] = matching_track['object_name']
                        # print(f"[TRACK] Updated category for Track {track_id} to '{matching_track['object_name']}'")


        return current_detections # Return list with 'track_id' added/updated


    def export_csv(self, path):
        """Exports the collected object records to a CSV file."""
        if not self.records:
            print("[EXPORT] No records to export.")
            return
        try:
            df = pd.DataFrame(self.records)
            df.to_csv(path, index=False)
            print(f"[EXPORT] Saved {len(df)} records to {path}")
            # Optionally clear records after export if memory is a concern
            # self.records.clear()
        except Exception as e:
            print(f"[ERROR] Failed to export records to CSV '{path}': {e}")

    def get_center(self, rect):
        """Calculates the integer center coordinates from a rotated rectangle tuple."""
        return (int(rect[0][0]), int(rect[0][1]))

    def recognize_object(self, features_record, identification_config, features_list):
        """
        Recognizes an object category based on feature ranges defined in the config.

        Args:
            features_record: Dictionary containing calculated feature values for the object.
            identification_config: Dictionary mapping category names to their feature parameters.
            features_list: List of feature names to use for matching.

        Returns:
            The name (str) of the matched category, or None if no match found.
        """
        if not identification_config or not features_list:
            # print("[DEBUG] Recognition skipped: No identification config or feature list.")
            return None

        for category, conf in identification_config.items():
            params = conf.get("parameters", {})
            match = True
            for feature_name in features_list:
                if feature_name not in features_record:
                    # print(f"[DEBUG] Feature '{feature_name}' not found in record for category '{category}'")
                    match = False; break
                if feature_name not in params:
                     # print(f"[DEBUG] Feature '{feature_name}' not configured for category '{category}'")
                     match = False; break

                # Check if the recorded value falls within the min/max range for this category
                min_val = params[feature_name].get("min", -float('inf'))
                max_val = params[feature_name].get("max", float('inf'))
                value = features_record[feature_name]

                if not (min_val <= value <= max_val):
                    # print(f"[DEBUG] Category '{category}' failed on '{feature_name}': {value} not in [{min_val}, {max_val}]")
                    match = False; break

            if match:
                 # print(f"[DEBUG] Matched category: {category}")
                 return category # Return the first category that matches all features

        # print("[DEBUG] No category matched.")
        return None # No category matched



    def compute_shape_features(self, contour, rect): # Ensure signature includes rect
        """Computes various shape features for a given contour."""
        features = {
            "area": 0.0, "perimeter": 0.0, "extent": 0.0, "hu_moments": [0.0]*7,
            "circularity": 0.0, "convexity_defects_count": 0, "avg_defect_depth": 0.0
        }
        # Initialize defect features to default
        features["convexity_defects_count"] = 0
        features["avg_defect_depth"] = 0.0

        try:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if area <= 1e-6:
                return features

            features["area"] = area
            features["perimeter"] = perimeter

            if perimeter > 1e-6:
                features["circularity"] = (4 * math.pi * area / (perimeter ** 2))

            _, _, w, h = cv2.boundingRect(contour)
            bounding_box_area = w * h
            if bounding_box_area > 1e-6:
                features["extent"] = area / bounding_box_area

            moments = cv2.moments(contour)
            if abs(moments['m00']) > 1e-6:
                hu_moments = cv2.HuMoments(moments).flatten()
                features["hu_moments"] = hu_moments.tolist()

            # --- Convexity Defects ---
            # Check contour length BEFORE attempting hull calculation
            if len(contour) <= 3:
                # print("[DEBUG] Skipping defects: Contour length <= 3")
                return features

            try:
                hull_indices = cv2.convexHull(contour, returnPoints=False)

                # --- Refined Validation BEFORE calling convexityDefects ---
                can_calculate_defects = False # Flag to control execution
                if hull_indices is not None and isinstance(hull_indices, np.ndarray) and len(hull_indices) >= 3:
                    # Ensure dtype is int32 right after calculation
                    if hull_indices.dtype != np.int32:
                        try:
                            hull_indices = hull_indices.astype(np.int32)
                        except Exception as e_astype:
                            print(f"[ERROR] Failed to cast hull_indices to int32: {e_astype}")
                            hull_indices = None # Mark as invalid if cast fails

                    # Proceed only if hull_indices are still valid
                    if hull_indices is not None:
                        # Check monotonicity explicitly using flatten()
                        try:
                            diffs = np.diff(hull_indices.flatten())
                            is_monotonic = np.all(diffs >= 0) or np.all(diffs <= 0)
                            if is_monotonic:
                                # Only if ALL checks pass, set the flag to True
                                can_calculate_defects = True
                            else:
                                print(f"[WARNING] Hull indices NOT monotonic. Skipping convexity defects.")
                                # Optional: print hull_indices here for debugging if needed
                                # print(f"    Indices: {hull_indices.flatten()}")
                        except Exception as e_monotonic_check:
                            print(f"[ERROR] Error checking monotonicity: {e_monotonic_check}")
                            # Assume cannot calculate defects if check fails
                            can_calculate_defects = False

                # --- Call convexityDefects ONLY if validation passed ---
                if can_calculate_defects:
                    try:
                        # print(f"[DEBUG] Attempting cv2.convexityDefects...")
                        defects = cv2.convexityDefects(contour, hull_indices)
                        # print("[DEBUG] cv2.convexityDefects call finished.")

                        if defects is not None and len(defects) > 0:
                            valid_defect_depths = [d[0][3] / 256.0 for d in defects if d[0][3] > 0]
                            features["convexity_defects_count"] = len(valid_defect_depths)
                            if valid_defect_depths:
                                features["avg_defect_depth"] = np.mean(valid_defect_depths)

                    except Exception as e_defects:
                        # Catch any other potential errors DURING defect calculation
                        print(f"[ERROR] Unexpected error DURING convexityDefects processing (even after checks): {e_defects}")
                        # Keep default values (0)
                        pass
                # else:
                    # print("[DEBUG] Skipping convexityDefects call due to validation failure.")

            except cv2.error as e_hull:
                    print(f"[DEBUG] cv2.error during convexHull calculation: {e_hull}")
                    pass # Keep defaults if hull fails

        except Exception as e_outer:
            print(f"[ERROR] Failed calculating basic shape features: {e_outer}")

        return features

    def compute_average_color(self, frame, contour):
        """Computes the average BGR color within a contour area."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        try:
            cv2.drawContours(mask, [contour], -1, 255, -1) # Fill the contour area
            # Calculate mean color ONLY where mask is non-zero
            mean_color_bgr = cv2.mean(frame, mask=mask)
            # Return as integer tuple (B, G, R)
            return (int(mean_color_bgr[0]), int(mean_color_bgr[1]), int(mean_color_bgr[2]))
        except Exception as e:
            print(f"[ERROR] Failed computing average color: {e}")
            return (0, 0, 0) # Return black on error

