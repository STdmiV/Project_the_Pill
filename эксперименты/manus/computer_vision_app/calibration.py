#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibration.py - Модуль калибровки камеры

Этот модуль реализует сбор кадров для калибровки камеры с использованием доски ChArUco,
вычисление матрицы камеры и коэффициентов дисторсии, а также сохранение полученных
параметров в файл.
"""

import cv2
import numpy as np
import os
import logging
from variables import (
    CALIBRATION_FILE, MIN_CHARUCO_CORNERS, SQUARE_LENGTH, 
    MARKER_LENGTH, SQUARES_X, SQUARES_Y
)

# Настройка логирования
logging.basicConfig(
    filename='error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CalibrationManager:
    """
    Класс для управления процессом калибровки камеры с использованием доски ChArUco.
    """
    
    def __init__(self):
        """
        Инициализирует внутренние переменные для калибровки.
        """
        self.all_corners = []  # Список всех углов
        self.all_ids = []  # Список всех идентификаторов
        self.image_size = None  # Размер изображения
        self.calibrated = False  # Флаг калибровки
        self.camera_matrix = None  # Матрица камеры
        self.dist_coeffs = None  # Коэффициенты дисторсии
        
        # Создание словаря ArUco и доски ChArUco
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
        self.board = cv2.aruco.CharucoBoard_create(
            squaresX=SQUARES_X,
            squaresY=SQUARES_Y,
            squareLength=SQUARE_LENGTH,
            markerLength=MARKER_LENGTH,
            dictionary=self.aruco_dict
        )
        
        # Параметры детектора
        self.parameters = cv2.aruco.DetectorParameters_create()
        
        # Проверка наличия файла калибровки
        self.load_calibration()
    
    def reset(self):
        """
        Сбрасывает внутреннее состояние для повторной калибровки.
        """
        self.all_corners = []
        self.all_ids = []
        self.calibrated = False
        self.camera_matrix = None
        self.dist_coeffs = None
    
    def add_frame(self, frame):
        """
        Добавляет кадр для калибровки.
        
        Args:
            frame: Кадр изображения для калибровки
            
        Returns:
            tuple: (успех, изображение с визуализацией)
        """
        if frame is None:
            return False, None
        
        # Создаем копию кадра для визуализации
        vis_frame = frame.copy()
        
        # Преобразуем в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Устанавливаем размер изображения, если он еще не установлен
        if self.image_size is None:
            self.image_size = gray.shape[::-1]
        
        # Обнаружение маркеров ArUco
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters
        )
        
        # Рисуем обнаруженные маркеры на визуализации
        cv2.aruco.drawDetectedMarkers(vis_frame, corners, ids)
        
        # Если маркеры обнаружены, интерполируем углы доски ChArUco
        if ids is not None and len(ids) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board
            )
            
            # Если найдено достаточное количество углов, добавляем их в списки
            if ret and charuco_corners is not None and len(charuco_corners) >= MIN_CHARUCO_CORNERS:
                self.all_corners.append(charuco_corners)
                self.all_ids.append(charuco_ids)
                
                # Рисуем углы доски ChArUco на визуализации
                cv2.aruco.drawDetectedCornersCharuco(vis_frame, charuco_corners, charuco_ids)
                
                return True, vis_frame
        
        return False, vis_frame
    
    def compute_calibration(self):
        """
        Вычисляет параметры калибровки на основе собранных кадров.
        
        Returns:
            bool: Успешность калибровки
        """
        if not self.all_corners or not self.all_ids or self.image_size is None:
            return False
        
        try:
            # Начальное приближение для матрицы камеры
            camera_matrix_init = np.array([
                [1000, 0, self.image_size[0] / 2],
                [0, 1000, self.image_size[1] / 2],
                [0, 0, 1]
            ])
            
            # Начальное приближение для коэффициентов дисторсии
            dist_coeffs_init = np.zeros((5, 1))
            
            # Флаги для калибровки
            flags = (
                cv2.CALIB_USE_INTRINSIC_GUESS + 
                cv2.CALIB_RATIONAL_MODEL
            )
            
            # Вычисление параметров калибровки
            ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=self.all_corners,
                charucoIds=self.all_ids,
                board=self.board,
                imageSize=self.image_size,
                cameraMatrix=camera_matrix_init,
                distCoeffs=dist_coeffs_init,
                flags=flags
            )
            
            # Сохранение параметров калибровки
            if ret:
                self.calibrated = True
                self.save_calibration()
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Ошибка при вычислении калибровки: {str(e)}")
            return False
    
    def save_calibration(self):
        """
        Сохраняет параметры калибровки в файл.
        
        Returns:
            bool: Успешность сохранения
        """
        if not self.calibrated or self.camera_matrix is None or self.dist_coeffs is None:
            return False
        
        try:
            np.savez(
                CALIBRATION_FILE,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                image_size=self.image_size
            )
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при сохранении калибровки: {str(e)}")
            return False
    
    def load_calibration(self):
        """
        Загружает параметры калибровки из файла.
        
        Returns:
            bool: Успешность загрузки
        """
        if not os.path.exists(CALIBRATION_FILE):
            return False
        
        try:
            data = np.load(CALIBRATION_FILE)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.image_size = tuple(data['image_size'])
            self.calibrated = True
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке калибровки: {str(e)}")
            return False
    
    def is_calibrated(self):
        """
        Проверяет, откалибрована ли камера.
        
        Returns:
            bool: True, если камера откалибрована, иначе False
        """
        return self.calibrated
    
    def get_calibration_data(self):
        """
        Возвращает данные калибровки.
        
        Returns:
            tuple: (camera_matrix, dist_coeffs) или (None, None), если калибровка не выполнена
        """
        if self.calibrated:
            return self.camera_matrix, self.dist_coeffs
        return None, None
    
    def get_calibration_board_image(self, width=1000, height=1400):
        """
        Создает изображение доски ChArUco для печати.
        
        Args:
            width: Ширина изображения
            height: Высота изображения
            
        Returns:
            numpy.ndarray: Изображение доски ChArUco
        """
        return self.board.draw((width, height))
