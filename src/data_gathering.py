# data_gathering.py

import cv2
import numpy as np
import pandas as pd
import json
import os
import math
import time
from scipy.optimize import linear_sum_assignment
import variables as var # Use alias for clarity
from ultimate import undistort_frame # Reuse undistort function
import threading
import logging # Use logging instead of print for better control

# Setup logger for this module
logger = logging.getLogger(__name__)

# Create a dedicated directory for gathered data CSVs if it doesn't exist
DATA_GATHERING_DIR = os.path.join(var.APP_DIR, "gathered_data")
if not os.path.exists(DATA_GATHERING_DIR):
    try:
        os.makedirs(DATA_GATHERING_DIR)
        logger.info(f"Created directory: {DATA_GATHERING_DIR}")
    except OSError as e:
        logger.error(f"Failed to create directory {DATA_GATHERING_DIR}: {e}")
        # Depending on the application, might want to raise an error here

class DataGatheringProcessor:
    """
    Handles object detection, tracking, facilitates manual classification,
    and performs continuous data saving for confirmed objects within a session.
    """
    def __init__(self, detection_params, request_classification_callback, update_status_callback, homography_matrix=None):
        """
        Args:
            detection_params (dict): Dictionary containing detection parameters
                                     (blur, canny, area, conversion_factor).
            request_classification_callback (function): Callback function to signal the GUI
                                                        when an object needs classification.
                                                        Expected signature: callback(target_track_id, frame_with_highlight)
            update_status_callback (function): Callback to update a status message in the GUI.
                                                Expected signature: callback(message)
        """
        self.detection_params = detection_params
        self.request_classification_callback = request_classification_callback
        self.update_status_callback = update_status_callback
        
        self.homography_matrix_to_mm = homography_matrix # Store the homography matrix
        if self.homography_matrix_to_mm is None:
            logger.warning("DataGatheringProcessor initialized without a homography matrix. Measurements will be in PIXELS or use fallback CONVERSION_FACTOR.")

        self.calibration_file = var.CALIBRATION_FILE
        self.next_track_id = 1
        self.tracks = []
        self.MAX_TRACK_LOST_FRAMES = var.MAX_LOST_FRAMES
        self.MAX_DISTANCE_MM = getattr(var, 'MAX_DISTANCE_MM', 10.0)
        # --- State for Manual Classification ---
        self._current_target_track_id = None # Track ID currently presented to user
        self._pending_classification_data = {} # Stores {track_id: detection_data} for objects awaiting classification
        self._classified_track_ids = set() # Track IDs already classified OR skipped in this session
        self._confirmed_classifications = {} # *** Stores {track_id: category_string} for accepted objects ***
        self._last_processed_time = {} # {track_id: timestamp} to prevent immediate re-request/throttling
        self._save_lock = threading.Lock() # Lock for thread-safe CSV writing
        self._object_queue = [] # Simple queue of track IDs needing classification
        self._last_frame_processed_time = time.time() # To manage processing rate

        logger.info("DataGatheringProcessor initialized.")
        self.update_status_callback("Ready. Start processing.")


    def process_frame(self, frame: np.ndarray, frame_counter: int, working_area_mask: np.ndarray | None = None) -> np.ndarray | None:
        """
        Processes a single frame for detection, tracking, manages the
        manual classification workflow, and saves data for confirmed tracks.

        Args:
            frame: The input frame (BGR).
            frame_counter: The current frame number.
            working_area_mask: Binary mask for the detection zone.

        Returns:
            An overlay frame (np.ndarray) with the current target highlighted (if any),
            or None if processing fails.
        """
        current_time = time.time()
        # Optional: Add processing rate limit if needed
        # if current_time - self._last_frame_processed_time < 0.05: # ~20 FPS max
        #     return None # Or return last known good frame? Be careful with state updates.
        self._last_frame_processed_time = current_time

        # --- 1. Undistort ---
        if not os.path.exists(self.calibration_file):
             logger.error("Calibration file not found. Cannot undistort.")
             corrected_frame = frame # Process the original frame if no calibration
        else:
             corrected_frame = undistort_frame(frame, self.calibration_file)

        if corrected_frame is None or corrected_frame.size == 0:
            logger.error("process_frame received an invalid or undistorted frame.")
            return None

        frame_height, frame_width = corrected_frame.shape[:2]

        # --- 2. Determine and Validate Mask ---
        mask_to_use = working_area_mask
        if mask_to_use is None:
            mask_to_use = np.ones((frame_height, frame_width), dtype=np.uint8) * 255
        elif mask_to_use.shape != (frame_height, frame_width):
            logger.warning("Mask shape mismatch! Using full frame instead.")
            mask_to_use = np.ones((frame_height, frame_width), dtype=np.uint8) * 255
        elif not np.any(mask_to_use):
             # Mask is all black, no detection possible
             logger.debug("Working area mask is empty, skipping detection.")
             # Need to return the frame with potential overlays if needed, or just the corrected frame
             # Let's return the corrected frame with hotkey hints if applicable
             objects_overlayed = corrected_frame.copy() # Start with corrected frame
             self._add_hotkey_overlay(objects_overlayed, frame_height)
             return objects_overlayed # Return frame with no detection overlays

        # --- 3. Preprocessing ---
        try:
            gray = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2GRAY)
            # Ensure blur kernel is odd and >= 1
            blur_kernel_size = self.detection_params.get("BLUR_KERNEL", 5)
            blur_kernel_size = max(1, int(blur_kernel_size) // 2 * 2 + 1)
            canny_low = self.detection_params.get("CANNY_LOW", 10)
            canny_high = self.detection_params.get("CANNY_HIGH", 50)
            blur = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
            edges = cv2.Canny(blur, canny_low, canny_high)
        except cv2.error as e_preproc:
             logger.error(f"OpenCV error during preprocessing: {e_preproc}")
             return corrected_frame.copy() # Return original on error
        except Exception as e_preproc_other:
            logger.error(f"Unexpected error during preprocessing: {e_preproc_other}")
            return corrected_frame.copy() # Return original on error

        # --- 4. Apply Mask ---
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask_to_use)

        # --- 5. Find and Filter Contours ---
        contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.detection_params.get("MIN_AREA", 100)
        max_area = self.detection_params.get("MAX_AREA", 50000)
        valid_contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]

        # --- 6. Process Valid Contours & Feature Extraction ---
        detections = []

        for contour in valid_contours:
            try:
                rect_px = cv2.minAreaRect(contour) # In pixels
                center_px_tuple = self.get_center(rect_px) # (x_px, y_px)
                (width_px, height_px), angle_deg = rect_px[1], rect_px[2]
                width_px, height_px = abs(width_px), abs(height_px) # Ensure positive dimensions

                # Calculate pixel-based shape features (invariant to scale or used for ratios)
                pixel_features = self.compute_shape_features(contour, rect_px) # compute_shape_features uses pixel data
                avg_color = self.compute_average_color(corrected_frame, contour)
                
            # --- Initialize metric values ---
                center_mm = (0,0)
                width_mm = 0
                height_mm = 0
                area_mm2 = 0
                perimeter_mm = 0
                avg_defect_depth_mm = 0
                
                if self.homography_matrix_to_mm is not None:
                    # --- Transform points using homography ---
                    center_px_arr = np.array([[center_px_tuple]], dtype="float32")
                    transformed_center_arr = cv2.perspectiveTransform(center_px_arr, self.homography_matrix_to_mm)
                    if transformed_center_arr is not None and transformed_center_arr.size > 0:
                        center_mm = (transformed_center_arr[0][0][0], transformed_center_arr[0][0][1])
                    else: # Should not happen if matrix and points are valid
                        logger.warning(f"Center point transformation failed for contour at {center_px_tuple}")
                        center_mm = center_px_tuple # Fallback to pixels

                    box_points_px = cv2.boxPoints(rect_px)
                    box_points_px_reshaped = box_points_px.reshape(-1, 1, 2).astype(np.float32)
                    transformed_box_points_mm_arr = cv2.perspectiveTransform(box_points_px_reshaped, self.homography_matrix_to_mm)

                    if transformed_box_points_mm_arr is not None and len(transformed_box_points_mm_arr) == 4:
                        side1_mm = np.linalg.norm(transformed_box_points_mm_arr[0][0] - transformed_box_points_mm_arr[1][0])
                        side2_mm = np.linalg.norm(transformed_box_points_mm_arr[1][0] - transformed_box_points_mm_arr[2][0])
                        
                        if width_px >= height_px:
                            width_mm = max(side1_mm, side2_mm)
                            height_mm = min(side1_mm, side2_mm)
                        else:
                            width_mm = min(side1_mm, side2_mm)
                            height_mm = max(side1_mm, side2_mm)
                        
                        area_mm2 = cv2.contourArea(transformed_box_points_mm_arr)
                        perimeter_mm = cv2.arcLength(transformed_box_points_mm_arr.astype(np.float32), closed=True)
                        
                        # Approximate scaling for avg_defect_depth (ideally transform defect points)
                        if width_px > 0 and width_mm > 0:
                            avg_scale_for_defects = width_mm / width_px
                            avg_defect_depth_mm = pixel_features["avg_defect_depth"] * avg_scale_for_defects
                        else:
                             avg_defect_depth_mm = pixel_features["avg_defect_depth"] # Use pixel value if scale cannot be determined
                    else:
                        logger.warning(f"Box points transformation failed for contour at {center_px_tuple}. Using pixel dimensions.")
                        # Fallback to pixel dimensions if transform fails
                        width_mm = width_px
                        height_mm = height_px
                        area_mm2 = pixel_features["area"]
                        perimeter_mm = pixel_features["perimeter"]
                        avg_defect_depth_mm = pixel_features["avg_defect_depth"]
                else:
                    # --- Fallback to CONVERSION_FACTOR if homography is NOT available ---
                    # This part is crucial for graceful degradation or if you intend mixed mode.
                    # If homography is ALWAYS expected, this 'else' might be an error condition.
                    logger.debug("No homography matrix, using CONVERSION_FACTOR from detection_params as fallback.")
                    conversion_factor_fallback = self.detection_params.get("CONVERSION_FACTOR", 1.0)
                    if conversion_factor_fallback <= 0:
                        logger.warning(f"Fallback CONVERSION_FACTOR ({conversion_factor_fallback}) is invalid, using 1.0 (pixels).")
                        conversion_factor_fallback = 1.0
                    
                    center_mm = (center_px_tuple[0] * conversion_factor_fallback, center_px_tuple[1] * conversion_factor_fallback)
                    width_mm = width_px * conversion_factor_fallback
                    height_mm = height_px * conversion_factor_fallback
                    area_mm2 = pixel_features["area"] * (conversion_factor_fallback**2)
                    perimeter_mm = pixel_features["perimeter"] * conversion_factor_fallback
                    avg_defect_depth_mm = pixel_features["avg_defect_depth"] * conversion_factor_fallback

                aspect_ratio_px = width_px / height_px if height_px > 1e-6 else 0
                hu_norm = np.linalg.norm(np.array(pixel_features["hu_moments"]))

                detection_data = {
                    'frame': frame_counter,
                    'center': center_mm, # Tuple (x_mm, y_mm)
                    'width': width_mm,
                    'height': height_mm,
                    'angle': angle_deg, # Still from pixel-space minAreaRect
                    'area_mm2': area_mm2,
                    'aspect_ratio': aspect_ratio_px, # Based on pixel dimensions
                    'perimeter_mm': perimeter_mm,
                    'extent': pixel_features["extent"],
                    'hu_norm': hu_norm,
                    'circularity': pixel_features["circularity"],
                    'convexity_defects_count': pixel_features["convexity_defects_count"],
                    'avg_defect_depth_mm': avg_defect_depth_mm,
                    'avg_color_r': avg_color[2],
                    'avg_color_g': avg_color[1],
                    'avg_color_b': avg_color[0],
                    'contour': contour # Pixel contour for drawing
                }
                detections.append(detection_data)
            except cv2.error as e_contour_cv:
                 logger.error(f"Frame {frame_counter}: OpenCV error processing contour: {e_contour_cv}")
                 continue
            except Exception as e_contour:
                 logger.error(f"Frame {frame_counter}: Unexpected error processing contour: {e_contour}", exc_info=True)
                 continue

        # --- 7. Assign Tracking IDs ---
        final_detections = self.assign_ids(detections) # Pass conversion_factor

        # --- 8. Save Data for Confirmed Tracks ---
        # This happens BEFORE queuing new objects
        for det in final_detections:
            track_id = det.get('track_id')
            # Check if this track_id has a confirmed classification
            if track_id is not None and track_id in self._confirmed_classifications:
                category = self._confirmed_classifications[track_id]
                # Add the category to the data dictionary before saving
                det['assigned_category'] = category
                # logger.debug(f"[SAVE] Saving data for confirmed Track ID {track_id} (Category: {category}) Frame: {frame_counter}") # Verbose log
                self._save_to_csv(category, det)
                # Optional: Update last processed time if needed for throttling
                # self._last_processed_time[track_id] = current_time

        # --- 9. Manage Classification Queue ---
        # Add new, unclassified tracks to the queue
        for det in final_detections:
            track_id = det.get('track_id')
            # Check if ID is valid, not already processed (classified/skipped), not confirmed, not the current target, and not already in the queue
            if track_id is not None and \
               track_id not in self._classified_track_ids and \
               track_id not in self._confirmed_classifications and \
               track_id != self._current_target_track_id and \
               track_id not in self._object_queue:
                # Prevent re-adding recently processed (skipped/accepted) objects too quickly
                last_processed = self._last_processed_time.get(track_id, 0)
                if current_time - last_processed > var.DATA_GATHERING_REQUEUE_DELAY: # Use configurable delay
                    self._object_queue.append(track_id)
                    # Store the full data dictionary when adding to queue
                    self._pending_classification_data[track_id] = det
                    logger.debug(f"[QUEUE] Added Track ID {track_id} to queue.")

        # --- 10. Request Classification if Idle and Queue Not Empty ---
        target_found_in_queue = False
        if self._current_target_track_id is None and self._object_queue:
            target_found_in_queue = False # Flag to know if we picked a target this frame
            # Ensure the queue isn't referencing an ID that got lost or confirmed while waiting
            while self._object_queue:
                 potential_target_id = self._object_queue.pop(0)
                 if potential_target_id in self._confirmed_classifications or \
                    potential_target_id in self._classified_track_ids or \
                    potential_target_id not in self._pending_classification_data:
                      # This ID is no longer valid for classification, try the next one
                      logger.debug(f"Skipping stale ID {potential_target_id} from queue.")
                      if potential_target_id in self._pending_classification_data:
                           del self._pending_classification_data[potential_target_id] # Clean up pending data
                      continue

                 else:
                      self._current_target_track_id = potential_target_id
                      target_found_in_queue = True # Mark that we found one
                      logger.info(f"[CLASSIFY] Requesting classification for Track ID: {self._current_target_track_id}")
                      self.update_status_callback(f"Classify Object ID: {self._current_target_track_id}")
                      # Callback will be triggered *after* drawing overlays (in Step 13)
                      break # Exit the while loop, we have a target
            else:
                 # Queue became empty while checking for stale IDs
                 logger.debug("Classification queue is empty or contains only stale IDs.")


        # --- 11. Draw Overlays ---
        objects_overlayed = corrected_frame.copy()
        target_frame_for_callback = None # Store the frame to send ONLY when requesting

        for det in final_detections:
            track_id = det.get('track_id')
            if track_id is None: continue

            try:
                # For drawing, use the original pixel contour and its properties
                # The 'contour' in det is already in pixels.
                rect_for_drawing_px = cv2.minAreaRect(det['contour'])
                center_x_px_draw, center_y_px_draw = int(rect_for_drawing_px[0][0]), int(rect_for_drawing_px[0][1])
                
                # ... (logic for draw_color, thickness, label_prefix based on state - no change) ...
                draw_color = (128, 128, 128); thickness = 1
                label_prefix = f"ID:{track_id}"
                if track_id == self._current_target_track_id:
                    draw_color = (0, 0, 255); thickness = 4
                    label_prefix = f"CLASSIFY -> ID:{track_id}"
                    target_frame_for_callback = objects_overlayed.copy() # Before drawing current target
                elif track_id in self._confirmed_classifications:
                    draw_color = (0, 255, 0); thickness = 2
                    category = self._confirmed_classifications.get(track_id, "SAVED")
                    try:
                        # Create short category code like [CW] for "circle_white"
                        parts = category.split('_')
                        if len(parts) >= 2:
                            short_code = "".join(p[0].upper() for p in parts if p)
                            label_prefix += f" [{short_code}]"
                        else:
                            label_prefix += f" [{category.upper()[:3]}]" # Fallback for single word categories
                    except Exception: # Broad catch for any issue with string parsing
                         label_prefix += " [SAVED]"

                # --- Draw Contour and Label ---
                # Draw the contour using the determined style
                cv2.drawContours(objects_overlayed, [det['contour']], -1, draw_color, thickness)

                # Draw center point (always red for visibility)
                cv2.circle(objects_overlayed, (center_x_px_draw, center_y_px_draw), 5, (0, 0, 255), -1)

                # Prepare and draw label text
                label = label_prefix # Use the label determined above
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                text_thickness = 2
                (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
                label_origin = (center_x_px_draw + 10, center_y_px_draw + label_height // 2)

                # Background rectangle
                cv2.rectangle(objects_overlayed, (label_origin[0] - 3, label_origin[1] - label_height - baseline + 2),
                              (label_origin[0] + label_width + 3, label_origin[1] + baseline - 1), (0, 0, 0), cv2.FILLED)
                # Label text (White)
                cv2.putText(objects_overlayed, label, label_origin, font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

            except ZeroDivisionError:
                 # This specific check might be redundant now due to the check above, but keep for safety
                 logger.error(f"ZeroDivisionError during overlay drawing for track {track_id} (Conversion factor likely zero).")
                 continue
            except Exception as e_draw:
                 logger.error(f"Frame {frame_counter}: Failed drawing overlay for track {track_id}: {e_draw}", exc_info=True)
                 continue

        # --- 12. Add Hotkey Hints Overlay ---
        self._add_hotkey_overlay(objects_overlayed, frame_height)

        # --- 13. Trigger Callback ONLY if a NEW target was selected THIS frame ---
        if target_found_in_queue and self._current_target_track_id is not None and target_frame_for_callback is not None:
            if self._current_target_track_id in self._pending_classification_data:
                 logger.debug(f"Emitting requestClassificationCallback for NEW target ID: {self._current_target_track_id}")
                 # Pass the specifically copied frame for the callback
                 self.request_classification_callback(self._current_target_track_id, target_frame_for_callback)
            else:
                 logger.warning(f"Target ID {self._current_target_track_id} was selected but missing from pending data before callback.")
                 self._current_target_track_id = None
                 self.update_status_callback("Target became invalid. Ready for next.")

        # Return the final frame with all overlays applied
        return objects_overlayed

    def _add_hotkey_overlay(self, frame, frame_height):
        """Adds the classification hotkey hints to the frame."""
        hint_text = "'Space'=Accept / 'R'=Skip"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (hint_w, hint_h), hint_bl = cv2.getTextSize(hint_text, font, font_scale, thickness)
        # Position at bottom-left
        rect_start = (5, frame_height - hint_h - 10)
        rect_end = (5 + hint_w + 4, frame_height - 5)
        text_origin = (7, frame_height - 7 - hint_bl // 2)
        # Draw black background rectangle
        cv2.rectangle(frame, rect_start, rect_end, (0,0,0), cv2.FILLED)
        # Draw white text
        cv2.putText(frame, hint_text, text_origin, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


    # --- Classification Handling Methods (Called by GUI) ---

    def accept_current_object(self, shape: str, color: str):
        """
        Confirms the classification for the target object, stores it for the session,
        and performs an initial save.
        """
        if self._current_target_track_id is None:
            logger.warning("Accept called but no object is targeted.")
            return

        target_id = self._current_target_track_id
        if shape == "unknown" or color == "unknown":
             logger.warning("Accept called with 'unknown' shape or color. Treating as skip.")
             self.skip_current_object() # Treat as skip
             return

        # Retrieve the data associated with the target ID *at the moment it was presented*
        if target_id not in self._pending_classification_data:
            logger.error(f"Data for target Track ID {target_id} not found in pending data. Cannot accept.")
            # Don't skip automatically, just clear the target state
            self._current_target_track_id = None
            self.update_status_callback("Error: Target data lost. Ready for next.")
            return

        category = f"{shape}_{color}".lower() # e.g., "circle_white"
        # Use the data stored when the object was added to the queue/presented
        object_data_for_initial_save = self._pending_classification_data[target_id]

        logger.info(f"[CLASSIFY] Confirming Track ID {target_id} as '{category}'. Continuous saving enabled.")

        # *** Store the confirmed classification for the session ***
        self._confirmed_classifications[target_id] = category

        # Add the assigned category to the data *before* the initial save
        object_data_for_initial_save['assigned_category'] = category
        # Perform the initial save for this specific frame/data point
        self._save_to_csv(category, object_data_for_initial_save)

        # Cleanup state
        self._classified_track_ids.add(target_id) # Mark as processed (accepted)
        self._last_processed_time[target_id] = time.time() # Record confirmation time
        if target_id in self._pending_classification_data:
            del self._pending_classification_data[target_id] # Remove from pending queue

        self._current_target_track_id = None # Ready for next object
        self.update_status_callback(f"Confirmed ID {target_id} as {category}. Saving. Ready.")

    def skip_current_object(self):
        """Marks the currently targeted object as skipped (unknown) for this session."""
        if self._current_target_track_id is None:
            logger.warning("Skip called but no object is targeted.")
            return

        target_id = self._current_target_track_id
        logger.info(f"[CLASSIFY] Skipping Track ID {target_id} (Marked as Unknown/Skip for this session)")

        # Mark as classified (so we don't ask again immediately) but don't save or confirm
        self._classified_track_ids.add(target_id) # Add to the set of processed IDs for this session
        self._last_processed_time[target_id] = time.time() # Record skip time to prevent immediate re-queueing

        # Remove from pending data if it was there
        if target_id in self._pending_classification_data:
            del self._pending_classification_data[target_id]

        # *** Ensure it's NOT in the confirmed list ***
        if target_id in self._confirmed_classifications:
            logger.warning(f"Skipping track {target_id} which was previously confirmed? Removing confirmation.")
            del self._confirmed_classifications[target_id] # Safety check

        self._current_target_track_id = None # Ready for next object
        self.update_status_callback("Skipped. Ready for next object.")

    # --- Internal Helper Methods ---

    def _save_to_csv(self, category: str, data: dict):
        """Appends object data to the corresponding category CSV file."""
        # --- Check if essential data is present ---
        track_id = data.get('track_id')
        assigned_category = data.get('assigned_category')
        if track_id is None or assigned_category is None:
             logger.error(f"Missing track_id ({track_id}) or assigned_category ({assigned_category}) in data. Aborting save. Data: {data}")
             return
        if assigned_category != category:
            logger.warning(f"Category mismatch! Trying to save to '{category}.csv' but data category is '{assigned_category}'. Using '{category}'.")
            # Proceeding with the passed 'category' for the filename

        # --- End Check ---

        filename = os.path.join(DATA_GATHERING_DIR, f"{category}.csv")
        # Define the exact columns expected in the CSV
        columns_to_save = [
            'frame', 'track_id', 'assigned_category',
            'center_x_mm', 'center_y_mm', 'width_mm', 'height_mm', 'angle_deg',
            'area_mm2', 'aspect_ratio', 'perimeter_mm', 'extent', 'hu_norm',
            'circularity', 'convexity_defects_count', 'avg_defect_depth_mm',
            'avg_color_r', 'avg_color_g', 'avg_color_b'
        ]

        # Prepare data dictionary for DataFrame, extracting values safely
        center_mm = data.get('center', (None, None)) # Default to tuple
        save_data = {
            'frame': data.get('frame'),
            'track_id': track_id,
            'assigned_category': assigned_category,
            'center_x_mm': center_mm[0] if isinstance(center_mm, (list, tuple)) and len(center_mm) > 0 else None,
            'center_y_mm': center_mm[1] if isinstance(center_mm, (list, tuple)) and len(center_mm) > 1 else None,
            'width_mm': data.get('width'),
            'height_mm': data.get('height'),
            'angle_deg': data.get('angle'),
            'area_mm2': data.get('area_mm2'),
            'aspect_ratio': data.get('aspect_ratio'),
            'perimeter_mm': data.get('perimeter_mm'),
            'extent': data.get('extent'),
            'hu_norm': data.get('hu_norm'),
            'circularity': data.get('circularity'),
            'convexity_defects_count': data.get('convexity_defects_count'),
            'avg_defect_depth_mm': data.get('avg_defect_depth_mm'),
            'avg_color_r': data.get('avg_color_r'),
            'avg_color_g': data.get('avg_color_g'),
            'avg_color_b': data.get('avg_color_b')
        }

        # Ensure all defined columns exist in the dictionary, even if None
        for col in columns_to_save:
            save_data.setdefault(col, None)

        # Create a single-row DataFrame using the correct columns order
        df_row = pd.DataFrame([save_data], columns=columns_to_save)

        with self._save_lock: # Ensure thread-safe file access
            try:
                # Append to CSV, write header if file doesn't exist or is empty
                file_exists = os.path.exists(filename)
                header = not file_exists or os.path.getsize(filename) == 0
                df_row.to_csv(filename, mode='a', header=header, index=False, lineterminator='\n') # Specify lineterminator
                # logger.debug(f"Saved data for ID {track_id} to {filename}") # Reduce log noise
            except IOError as e_io:
                 logger.error(f"IOError saving data to CSV '{filename}': {e_io}")
            except Exception as e:
                logger.error(f"Failed to save data to CSV '{filename}': {e}", exc_info=True)

    # --- Tracking and Feature Methods ---




    def assign_ids(self, current_detections: list) -> list:
        """
        Assigns track IDs using Hungarian algorithm based on distance in mm.
        Adds 'track_id' key to each detection dictionary. Handles lost tracks.

        Args:
            current_detections (list): List of detection data dictionaries for the current frame.
                                        Each dict must contain 'center' key with (x_mm, y_mm).
            conversion_factor (float): The pixels-to-mm conversion factor.

        Returns:
            list: The input list of detections with 'track_id' added or updated.
        """
        track_centers_mm = np.array([track['last_center_mm'] for track in self.tracks]) if self.tracks else np.empty((0, 2))
        detection_centers_mm = np.array([det['center'] for det in current_detections]) if current_detections else np.empty((0, 2))

        assigned_detection_indices = set()
        matched_track_indices = set()

        # --- Matching using Hungarian Algorithm ---
        if track_centers_mm.size > 0 and detection_centers_mm.size > 0:
            distances_mm = np.linalg.norm(track_centers_mm[:, np.newaxis, :] - detection_centers_mm[np.newaxis, :, :], axis=2)
            max_matching_distance_mm = self.MAX_DISTANCE_MM 

            try:
                track_indices, detection_indices = linear_sum_assignment(distances_mm)
            except ValueError as e_assign:
                logger.error(f"Error during linear sum assignment: {e_assign}. Skipping matching for this frame.")
                track_indices, detection_indices = [], []

            for r, c in zip(track_indices, detection_indices):

                if r < len(self.tracks) and c < len(current_detections):
                    if distances_mm[r, c] < max_matching_distance_mm:
                        track = self.tracks[r]
                        detection = current_detections[c]
                        track['last_center_mm'] = detection['center'] # Already in mm
                        track['lost_frames'] = 0
                        detection['track_id'] = track['id']
                        assigned_detection_indices.add(c)
                        matched_track_indices.add(r)
                    else:
                        logger.error(f"Invalid indices from linear_sum_assignment: track={r}, detection={c}")


        # --- Handle Unmatched Detections (New Tracks) ---
        for i, det in enumerate(current_detections):
            if i not in assigned_detection_indices:
                new_track_id = self.next_track_id
                det['track_id'] = new_track_id
                self.tracks.append({
                    'id': new_track_id,
                    'last_center_mm': det['center'],
                    'lost_frames': 0,
                })
                logger.info(f"[TRACK] New Track {new_track_id} created at {det['center'][0]:.1f}, {det['center'][1]:.1f} mm.")
                self.next_track_id += 1

        # --- Handle Unmatched Tracks (Lost Frames & Cleanup) ---
        indices_to_remove = []
        for i, track in enumerate(self.tracks):
            if i not in matched_track_indices:
                track['lost_frames'] += 1
                if track['lost_frames'] > self.MAX_TRACK_LOST_FRAMES:
                    indices_to_remove.append(i)
                    lost_track_id = track['id']
                    logger.info(f"[TRACK] Track {lost_track_id} marked for removal (lost > {self.MAX_TRACK_LOST_FRAMES} frames).")

                    # Cleanup state associated with the lost track ID
                    if lost_track_id == self._current_target_track_id:
                        logger.warning(f"[TRACK] Target Track ID {lost_track_id} was lost. Clearing target.")
                        self._current_target_track_id = None
                        self.update_status_callback("Target lost. Ready for next object.")

                    self._pending_classification_data.pop(lost_track_id, None)
                    self._confirmed_classifications.pop(lost_track_id, None)
                    self._classified_track_ids.discard(lost_track_id)
                    self._last_processed_time.pop(lost_track_id, None)

                    # --- Safely remove from queue if present ---
                    # THIS IS THE CRITICAL BLOCK
                    try:
                        logger.debug(f"Attempting to remove lost track ID {lost_track_id} from queue: {self._object_queue}")
                        if lost_track_id in self._object_queue:
                            self._object_queue.remove(lost_track_id)
                            logger.debug(f"Successfully removed lost track ID {lost_track_id} from object queue.")
                        else:
                            logger.debug(f"Lost track ID {lost_track_id} not found in object queue. No action needed.")
                    except ValueError:
                        # This is OKAY and EXPECTED if the ID was already processed/removed
                        logger.debug(f"Lost track ID {lost_track_id} was not found in object queue (already processed?). No action needed.")
                        pass # Ignore the error silently
                    except Exception as e_queue_remove:
                        # Log other potential errors during removal
                        logger.error(f"Unexpected error removing {lost_track_id} from queue: {e_queue_remove}", exc_info=True)
                    # --- End of safe removal ---

        # --- Remove Lost Tracks from the self.tracks list ---
        if indices_to_remove:
            logger.debug(f"Removing track indices: {sorted(indices_to_remove, reverse=True)}")
            for i in sorted(indices_to_remove, reverse=True):
                if 0 <= i < len(self.tracks):
                    del self.tracks[i]
                else:
                    logger.error(f"Attempted to remove track at invalid index {i} (track list size: {len(self.tracks)})")
            logger.debug(f"Track list size after removal: {len(self.tracks)}")

        # Ensure 'track_id' is present in all returned detections
        for det in current_detections:
            det.setdefault('track_id', None)

        return current_detections

    def get_center(self, rect):
        """Calculates the integer pixel center coordinates from a cv2.minAreaRect tuple."""
        # rect[0] is the center (x, y)
        return (int(rect[0][0]), int(rect[0][1]))

    # Inside data_gathering.py -> class DataGatheringProcessor

    def compute_shape_features(self, contour, rect):
        """Computes various shape features for a given contour."""
        features = {
            "area": 0.0, "perimeter": 0.0, "extent": 0.0, "hu_moments": [0.0]*7,
            "circularity": 0.0, "convexity_defects_count": 0, "avg_defect_depth": 0.0
        }
        hull_indices = None # Initialize hull_indices outside try block

        try:
            area = cv2.contourArea(contour)
            if area <= 1e-6: # Avoid calculations for zero or near-zero area contours
                    logger.debug("Skipping feature calculation for zero/small area contour.")
                    return features

            perimeter = cv2.arcLength(contour, True)
            features["area"] = area
            features["perimeter"] = perimeter

            # Circularity
            if perimeter > 1e-6:
                features["circularity"] = (4 * math.pi * area / (perimeter ** 2))

            # Extent
            x, y, w, h = cv2.boundingRect(contour)
            bounding_box_area = w * h
            if bounding_box_area > 1e-6:
                features["extent"] = area / bounding_box_area

            # Hu Moments
            moments = cv2.moments(contour)
            if abs(moments['m00']) > 1e-6:
                    hu_moments = cv2.HuMoments(moments).flatten()
                    features["hu_moments"] = hu_moments.tolist()

            # --- Convexity Defects ---
            # Check contour length BEFORE attempting hull calculation
            if len(contour) <= 3:
                logger.debug(f"Skipping convexity defects: Contour length ({len(contour)}) <= 3")
                return features # Return features calculated so far

            try:
                # Calculate convex hull indices
                hull_indices = cv2.convexHull(contour, returnPoints=False)

                # Validate hull indices *before* calling convexityDefects
                if hull_indices is None:
                    logger.debug("Convex hull calculation returned None. Skipping defects.")
                elif len(hull_indices) < 3:
                    logger.debug(f"Convex hull has too few points ({len(hull_indices)}). Skipping defects.")
                else:
                    # Check if hull indices are monotonous (simple check)
                    is_monotonic = np.all(np.diff(hull_indices.flatten()) >= 0) or np.all(np.diff(hull_indices.flatten()) <= 0)
                    if not is_monotonic:
                        # If not monotonic, try sorting them. This might fix the issue for convexityDefects.
                        # logger.warning(f"Hull indices are not monotonic (shape: {hull_indices.shape}, dtype: {hull_indices.dtype}). Attempting to sort.")
                        # Flatten, sort uniquely, and reshape
                        # unique_sorted_indices = np.unique(hull_indices.flatten()) # Get unique sorted indices
                        # hull_indices = unique_sorted_indices.reshape(-1, 1).astype(np.int32) # Reshape and ensure int32
                        # logger.warning(f"Indices after sort/unique: shape {hull_indices.shape}, dtype {hull_indices.dtype}")
                        # Re-check length after potential modification
                        # if len(hull_indices) < 3:
                        #     logger.warning("Hull has < 3 points after sorting/unique. Skipping defects.")
                        #     hull_indices = None # Prevent further processing
                        # --- Alternative: Just skip if not monotonic ---
                        logger.warning(f"Hull indices are not monotonic (shape: {hull_indices.shape}, dtype: {hull_indices.dtype}). Skipping convexity defects calculation.")
                        hull_indices = None # Mark as invalid to skip defect calculation


                    # Proceed only if hull_indices are still valid after checks
                    if hull_indices is not None and len(hull_indices) >= 3:
                        try:
                            # --- Log right before the call ---
                            logger.debug(f"Calling cv2.convexityDefects with contour len {len(contour)}, hull len {len(hull_indices)}, hull dtype {hull_indices.dtype}")
                            # --- The potentially failing call ---
                            defects = cv2.convexityDefects(contour, hull_indices)
                            # --- Log success ---
                            logger.debug("cv2.convexityDefects call succeeded.")

                            if defects is not None:
                                valid_defect_depths = [d[0][3] / 256.0 for d in defects if d[0][3] > 0]
                                features["convexity_defects_count"] = len(valid_defect_depths)
                                if valid_defect_depths:
                                    features["avg_defect_depth"] = np.mean(valid_defect_depths)
                            else:
                                logger.debug("cv2.convexityDefects returned None.")

                        except Exception as e_defects:
                            logger.critical("*** CRITICAL: EXCEPTION CAUGHT IN CONVEXITY DEFECTS ***")
                            logger.critical(f"    Contour len: {len(contour)}, Hull len: {len(hull_indices) if hull_indices is not None else 'None'}")
                            logger.critical(f"    Hull dtype: {hull_indices.dtype if hull_indices is not None else 'N/A'}")
                            logger.critical(f"    Error: {e_defects}", exc_info=True)
                            # Do not re-raise, keep default values (0) for defect features
                            pass

            except cv2.error as e_hull:
                    logger.debug(f"cv2 error during convexHull calculation: {e_hull}")
                    pass # Keep defaults if hull fails

        except Exception as e:
            logger.error(f"Failed calculating shape features: {e}", exc_info=True)

        return features
    
    def compute_average_color(self, frame, contour):
        """Computes the average BGR color within the contour using a mask."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        try:
            # Draw filled contour on the mask (value 255)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            # Check if the mask has any non-zero pixels (i.e., contour area > 0)
            if cv2.countNonZero(mask) == 0:
                 logger.debug("Cannot compute average color: contour mask is empty.")
                 return (0, 0, 0) # Return black if contour is empty/invalid

            # Calculate mean color only where mask is non-zero
            mean_color_bgr = cv2.mean(frame, mask=mask)
            # Return as tuple of integers (B, G, R)
            return (int(mean_color_bgr[0]), int(mean_color_bgr[1]), int(mean_color_bgr[2]))
        except cv2.error as e_color_cv:
            logger.error(f"OpenCV error computing average color: {e_color_cv}")
            return (0, 0, 0) # Return black on error
        except Exception as e:
            logger.error(f"Failed computing average color: {e}", exc_info=True)
            return (0, 0, 0) # Return black on error

    def reset_state(self):
        """Resets tracking, classification, and confirmation state for a new session."""
        logger.info("Resetting DataGatheringProcessor state.")
        self.next_track_id = 1
        self.tracks.clear() # Use clear() for lists/dicts/sets
        self._current_target_track_id = None
        self._pending_classification_data.clear()
        self._classified_track_ids.clear()
        self._confirmed_classifications.clear() # *** Clear the confirmed classifications ***
        self._object_queue.clear()
        self._last_processed_time.clear()
        # Optionally reset _last_frame_processed_time ?
        # self._last_frame_processed_time = time.time()
        self.update_status_callback("State reset. Ready.")