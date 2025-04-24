#calib.py


import cv2
import numpy as np
import os
import time
import logging # For logging
from variables import (CALIBRATION_FILE, CALIBRATION_VIDEO_PATH, SQUARE_LENGTH, MARKER_LENGTH,
                       SQUARES_X, SQUARES_Y, MIN_CHARUCO_CORNERS, CAMERA_INDEX, MAX_CAPTURES, source_type, DICT_TYPE)

# ===== Настраиваемые переменные =====

# ====================================

# ===== Функция калибровки =====
def calibrate_camera(source_type="video", CAMERA_INDEX=0):
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                                   SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
#prep 
    allCorners = []
    allIds = []
    imageSize = None

    #part 1
    if source_type == "video":
        cap = cv2.VideoCapture(CALIBRATION_VIDEO_PATH)
    elif source_type == "camera":
        cap = cv2.VideoCapture(CAMERA_INDEX)
    else:
        print("Invalid source type. Use 'video' or 'camera'.")
        
        return None, None
    if not cap.isOpened():
        print("No video.")
        return None, None

    window_name = "Calib vid"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    frame_count = 0
    saved_frames = 0
    recording = True
    
    #part 2
    while True:


        ret, frame = cap.read()
        if not ret:
            print("Error or vid end")
            break
        
        frame_count += 1
        
        # prep frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if imageSize is None:
            imageSize = gray.shape[::-1]  # (width, height)

        # detect params
        try:
            parameters = cv2.aruco.DetectorParameters()
        except AttributeError:
            parameters = cv2.aruco.DetectorParameters()
        parameters.minMarkerPerimeterRate = 0.01

        # detect markers
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        print(f"frame {frame_count}: Found markers: {len(ids) if ids is not None else 0}")

        if ids is not None and len(ids) > 0:
            ret_interp, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners, markerIds=ids, image=gray, board=board
            )
            
            # check min val for corners
            if (charucoCorners is not None and charucoIds is not None and 
                len(charucoCorners) > MIN_CHARUCO_CORNERS):
                print(f"frame {frame_count}: Corners interpolated: {len(charucoCorners)}")
                print(f"[INFO] Saved calibration frame #{saved_frames}/{MAX_CAPTURES}")
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds)
                
                if recording and frame_count % 30 == 0 and saved_frames < MAX_CAPTURES :
                    allCorners.append(charucoCorners)
                    allIds.append(charucoIds)

                    saved_frames += 1                    
                    print(f"Frame {frame_count} captured, corners: {len(charucoCorners)}")

            else:
                print(f"frame {frame_count}: No corners or not enaugth")
        else:
            print(f"Frame {frame_count}: No markers")

        if saved_frames >= MAX_CAPTURES:
            recording = False
            print(f"[INFO] Reached MAX_CAPTURES = {MAX_CAPTURES}. Stopping recording...")

        
        # Для отображения: динамическое масштабирование с сохранением пропорций
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
            if win_w == 0 or win_h == 0:
                win_w, win_h = 1280, 720
        except Exception:
            win_w, win_h = 1280, 720

        orig_h, orig_w = frame.shape[:2]
        aspect_ratio = orig_w / orig_h
        print(f"Init Aspect ratio: {aspect_ratio:.2f}")
        scale_factor = min(win_w / orig_w, win_h / orig_h)
        new_w = max(int(orig_w * scale_factor), 1)
        new_h = max(int(orig_h * scale_factor), 1)
        frame_resized = cv2.resize(frame, (new_w, new_h))
        
        cv2.putText(frame_resized, f"Saved frames: {saved_frames}/{MAX_CAPTURES}", (10, new_h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame_resized, f"Aspect Ratio: {aspect_ratio:.2f}", (10, new_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow(window_name, frame_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("[INFO] Stopping recording and starting calibration...")
            break
        if key == ord('q'):
            print("[INFO] Quit without calibration.")
            cap.release()
            cv2.destroyAllWindows()
            return None, None

    cap.release()
    cv2.destroyAllWindows()

    if len(allCorners) == 0:
        print("No frames captured for calibration.")
        return None, None

    # firs estimations(placeholder)
    cameraMatrix = np.array([[1000.0, 0.0, imageSize[0] / 2.0],
                             [0.0, 1000.0, imageSize[1] / 2.0],
                             [0.0, 0.0, 1.0]], dtype=np.float64)
    
    distCoeffs = np.zeros((5, 1), dtype=np.float64)
    
    
    #Camera calibration
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=allCorners, charucoIds=allIds, board=board, imageSize=imageSize,
                cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, flags=cv2.CALIB_RATIONAL_MODEL
    )

    print("\n===== Calib res =====")
    print(f"Mean Reprojection Error: {ret}")
    print("Camera matrix:")
    print(cameraMatrix)
    print("Distortion coeff:")
    print(distCoeffs)

    # Rewrite or create (delete if exists)

    if os.path.exists(CALIBRATION_FILE):
        os.remove(CALIBRATION_FILE)
    try:
        np.savez(CALIBRATION_FILE,
                 cameraMatrix=cameraMatrix, distCoeffs=distCoeffs,  
                 rvecs=rvecs, tvecs=tvecs, reprojectionError=ret)
        print(f"Distort correction saved in {CALIBRATION_FILE}.")
    except Exception as e:
        print("Error:", e)

    return cameraMatrix, distCoeffs

def calibrate_camera_gui(source_type="video", camera_index=0, video_path=None, max_captures=50,
                         frame_callback=None, done_callback=None, stop_flag_getter=None):
    """
    Run Charuco calibration and stream frames via callback into GUI.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                                   SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)

    allCorners = []
    allIds = []
    imageSize = None

    if source_type == "video" and video_path:
        cap = cv2.VideoCapture(video_path)
    elif source_type == "camera":
        cap = cv2.VideoCapture(camera_index)
    else:
        print("Invalid source.")
        if done_callback:
            done_callback(None, None)
        return

    if not cap.isOpened():
        print("Could not open video source.")
        if done_callback:
            done_callback(None, None)
        return

    frame_count = 0
    saved_frames = 0
    recording = True

    try:
        parameters = cv2.aruco.DetectorParameters()
    except AttributeError:
        parameters = cv2.aruco.DetectorParameters()

    parameters.minMarkerPerimeterRate = 0.01

    while True:
        if stop_flag_getter and stop_flag_getter():
            print("[INFO] Stop flag triggered, exiting frame loop...")
            break
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if imageSize is None:
            imageSize = gray.shape[::-1]

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) > 0:
            ret_interp, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners, markerIds=ids, image=gray, board=board
            )

            if (charucoCorners is not None and charucoIds is not None and 
                len(charucoCorners) > MIN_CHARUCO_CORNERS):

                if recording and frame_count % 30 == 0 and saved_frames < max_captures:
                    allCorners.append(charucoCorners)
                    allIds.append(charucoIds)
                    saved_frames += 1

                # Draw markers for GUI
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds)

        # Add overlay text: saved frames and aspect ratio
        aspect_ratio = imageSize[0] / imageSize[1] if imageSize else 1.0
        cv2.putText(frame, f"Saved frames: {saved_frames}/{max_captures}",
                    (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Aspect Ratio: {aspect_ratio:.2f}",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)


        # Send frame to GUI
        if frame_callback:
            frame_callback(frame)

        if saved_frames >= max_captures:
            print(f"[INFO] Reached max captures = {max_captures}")
            break


    cap.release()

    if len(allCorners) == 0:
        print("No usable frames for calibration.")
        if done_callback:
            done_callback(None, None)
        return

    cameraMatrix = np.array([[1000.0, 0.0, imageSize[0] / 2.0],
                             [0.0, 1000.0, imageSize[1] / 2.0],
                             [0.0, 0.0, 1.0]], dtype=np.float64)
    distCoeffs = np.zeros((5, 1), dtype=np.float64)

    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=allCorners, charucoIds=allIds, board=board, imageSize=imageSize,
        cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, flags=cv2.CALIB_RATIONAL_MODEL
    )

    try:
        np.savez(CALIBRATION_FILE,
                 cameraMatrix=cameraMatrix, distCoeffs=distCoeffs,  
                 rvecs=rvecs, tvecs=tvecs, reprojectionError=ret)
        print(f"Calibration saved to {CALIBRATION_FILE}")
    except Exception as e:
        print("Error saving calibration:", e)

    if done_callback:
        done_callback(cameraMatrix, distCoeffs)


if __name__ == "__main__":
    calibrate_camera(source_type="video", CAMERA_INDEX=0)
