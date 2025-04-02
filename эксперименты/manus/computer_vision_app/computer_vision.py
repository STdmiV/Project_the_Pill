#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
computer_vision.py - Модуль обработки изображений

Этот модуль реализует алгоритмы обработки изображений, включая предварительную обработку,
вычисление геометрических признаков, распознавание объектов и трекинг объектов между кадрами.
"""

import cv2
import numpy as np
import json
import logging
from scipy.optimize import linear_sum_assignment
import os
from variables import (
    SCALE, BLUR_KERNEL, CANNY_LOW, CANNY_HIGH, MIN_AREA, MAX_AREA,
    MAX_LOST_FRAMES, MAX_DISTANCE, CONFIG_PATH
)

# Настройка логирования
logging.basicConfig(
    filename='error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def make_odd(value):
    """
    Приводит число к нечётному значению (минимум 3).
    
    Args:
        value: Исходное число
        
    Returns:
        int: Нечётное число >= 3
    """
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1

def get_center(rect):
    """
    Возвращает центр повёрнутого прямоугольника.
    
    Args:
        rect: Повёрнутый прямоугольник (x, y, width, height, angle)
        
    Returns:
        tuple: Координаты центра (x, y)
    """
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    center = np.mean(box, axis=0)
    return (int(center[0]), int(center[1]))

def compute_shape_features(contour, rect):
    """
    Вычисляет геометрические признаки контура.
    
    Args:
        contour: Контур объекта
        rect: Повёрнутый прямоугольник
        
    Returns:
        dict: Словарь с признаками формы
    """
    # Основные признаки
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Прямоугольность (отношение площади контура к площади ограничивающего прямоугольника)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0
    
    # Вычисление моментов Hu
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Кругловость (4 * pi * area / perimeter^2)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Выпуклость (отношение площади контура к площади выпуклой оболочки)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # Выпуклые дефекты
    if len(contour) > 3:  # Минимум 4 точки для вычисления выпуклых дефектов
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        
        if defects is not None:
            convexity_defects_count = len(defects)
            
            # Средняя глубина дефектов
            depths = [d[0][3] / 256.0 for d in defects]
            avg_depth = np.mean(depths) if depths else 0
        else:
            convexity_defects_count = 0
            avg_depth = 0
    else:
        convexity_defects_count = 0
        avg_depth = 0
    
    # Соотношение сторон повёрнутого прямоугольника
    _, (width, height), _ = rect
    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
    
    return {
        'area': area,
        'perimeter': perimeter,
        'extent': extent,
        'hu_moments': hu_moments.tolist(),
        'circularity': circularity,
        'solidity': solidity,
        'convexity_defects_count': convexity_defects_count,
        'avg_defect_depth': avg_depth,
        'aspect_ratio': aspect_ratio
    }

def compute_average_color(frame, contour):
    """
    Вычисляет среднее значение цвета внутри контура.
    
    Args:
        frame: Исходное изображение
        contour: Контур объекта
        
    Returns:
        list: Среднее значение цвета [B, G, R]
    """
    # Создаем маску для контура
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    
    # Вычисляем среднее значение цвета
    mean_color = cv2.mean(frame, mask=mask)
    
    # Возвращаем BGR значения (без альфа-канала)
    return [mean_color[0], mean_color[1], mean_color[2]]

def process_frame_canny(frame, blur_kernel=BLUR_KERNEL, canny_low=CANNY_LOW, 
                        canny_high=CANNY_HIGH, min_area=MIN_AREA, max_area=MAX_AREA, mask=None):
    """
    Обрабатывает кадр с использованием детектора Canny для выделения контуров.
    
    Args:
        frame: Исходное изображение
        blur_kernel: Размер ядра для размытия
        canny_low: Нижний порог для детектора Canny
        canny_high: Верхний порог для детектора Canny
        min_area: Минимальная площадь контура
        max_area: Максимальная площадь контура
        mask: Маска для обработки (опционально)
        
    Returns:
        tuple: (edges, visualization, rectangles, contours)
    """
    # Проверка входных данных
    if frame is None:
        return None, None, [], []
    
    # Преобразуем в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Применяем размытие по Гауссу с нечётным ядром
    blur_kernel = make_odd(blur_kernel)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    
    # Применяем детектор Canny
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    # Применяем маску, если она предоставлена
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    # Находим контуры
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Создаем изображение для визуализации
    vis = frame.copy()
    
    # Фильтруем контуры по площади и вычисляем минимальные повёрнутые прямоугольники
    filtered_contours = []
    rectangles = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_area <= area <= max_area:
            rect = cv2.minAreaRect(contour)
            filtered_contours.append(contour)
            rectangles.append(rect)
            
            # Рисуем контур случайным цветом
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            cv2.drawContours(vis, [contour], 0, color, 2)
            
            # Рисуем повёрнутый прямоугольник
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
            
            # Рисуем центр
            center = get_center(rect)
            cv2.circle(vis, center, 5, (0, 0, 255), -1)
    
    return edges, vis, rectangles, filtered_contours

def recognize_object(features_record, identification_config, features_list):
    """
    Распознает объект на основе его признаков и конфигурации.
    
    Args:
        features_record: Словарь с признаками объекта
        identification_config: Конфигурация для идентификации
        features_list: Список признаков для сравнения
        
    Returns:
        str: Предсказанная категория объекта или None, если не удалось распознать
    """
    if not identification_config or not features_record:
        return None
    
    # Проверяем каждую категорию в конфигурации
    for category, config in identification_config.items():
        matches = True
        
        # Проверяем каждый признак в списке
        for feature in features_list:
            if feature in config and feature in features_record:
                # Получаем диапазон значений из конфигурации
                min_val, max_val = config[feature]
                
                # Получаем значение признака
                value = features_record[feature]
                
                # Проверяем, находится ли значение в диапазоне
                if not (min_val <= value <= max_val):
                    matches = False
                    break
        
        # Если все признаки соответствуют, возвращаем категорию
        if matches:
            return category
    
    return None

class ObjectDetector:
    """
    Класс для обнаружения и трекинга объектов на видеопотоке.
    """
    
    def __init__(self, config_path=CONFIG_PATH):
        """
        Инициализирует детектор объектов.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = config_path
        self.config = None
        self.config_timestamp = 0
        
        # Параметры обработки
        self.scale = SCALE
        self.blur_kernel = BLUR_KERNEL
        self.canny_low = CANNY_LOW
        self.canny_high = CANNY_HIGH
        self.min_area = MIN_AREA
        self.max_area = MAX_AREA
        
        # Параметры трекинга
        self.tracks = []  # Список треков объектов
        self.next_id = 0  # Следующий идентификатор объекта
        
        # Загружаем конфигурацию
        self.load_config()
    
    def load_config(self, force=False):
        """
        Загружает конфигурацию из файла.
        
        Args:
            force: Принудительная загрузка, даже если файл не изменился
            
        Returns:
            bool: Успешность загрузки
        """
        try:
            # Проверяем, существует ли файл конфигурации
            if not os.path.exists(self.config_path):
                # Создаем пустую конфигурацию, если файл не существует
                self.config = {
                    "identification": {},
                    "processing": {
                        "scale": self.scale,
                        "blur_kernel": self.blur_kernel,
                        "canny_low": self.canny_low,
                        "canny_high": self.canny_high,
                        "min_area": self.min_area,
                        "max_area": self.max_area
                    }
                }
                return True
            
            # Получаем время последнего изменения файла
            current_timestamp = os.path.getmtime(self.config_path)
            
            # Загружаем конфигурацию, только если файл изменился или принудительно
            if force or current_timestamp > self.config_timestamp:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                
                # Обновляем параметры обработки
                if "processing" in self.config:
                    processing = self.config["processing"]
                    self.scale = processing.get("scale", SCALE)
                    self.blur_kernel = processing.get("blur_kernel", BLUR_KERNEL)
                    self.canny_low = processing.get("canny_low", CANNY_LOW)
                    self.canny_high = processing.get("canny_high", CANNY_HIGH)
                    self.min_area = processing.get("min_area", MIN_AREA)
                    self.max_area = processing.get("max_area", MAX_AREA)
                
                # Обновляем временную метку
                self.config_timestamp = current_timestamp
                
                return True
            
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {str(e)}")
            return False
    
    def save_config(self):
        """
        Сохраняет конфигурацию в файл.
        
        Returns:
            bool: Успешность сохранения
        """
        try:
            # Обновляем параметры обработки в конфигурации
            if self.config is None:
                self.config = {}
            
            self.config["processing"] = {
                "scale": self.scale,
                "blur_kernel": self.blur_kernel,
                "canny_low": self.canny_low,
                "canny_high": self.canny_high,
                "min_area": self.min_area,
                "max_area": self.max_area
            }
            
            # Сохраняем конфигурацию в файл
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            # Обновляем временную метку
            self.config_timestamp = os.path.getmtime(self.config_path)
            
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при сохранении конфигурации: {str(e)}")
            return False
    
    def process_frame(self, frame, update_config=False):
        """
        Обрабатывает кадр для обнаружения и трекинга объектов.
        
        Args:
            frame: Исходное изображение
            update_config: Обновить конфигурацию перед обработкой
            
        Returns:
            tuple: (visualization, objects_info)
        """
        if frame is None:
            return None, []
        
        # Обновляем конфигурацию, если требуется
        if update_config:
            self.load_config()
        
        # Масштабируем изображение, если нужно
        if self.scale != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * self.scale), int(h * self.scale)))
        
        # Обрабатываем кадр для выделения контуров
        edges, vis, rectangles, contours = process_frame_canny(
            frame,
            blur_kernel=self.blur_kernel,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            min_area=self.min_area,
            max_area=self.max_area
        )
        
        # Список новых детекций
        detections = []
        
        # Обрабатываем каждый контур
        for i, (contour, rect) in enumerate(zip(contours, rectangles)):
            # Получаем центр объекта
            center = get_center(rect)
            
            # Вычисляем признаки формы
            shape_features = compute_shape_features(contour, rect)
            
            # Вычисляем среднее значение цвета
            avg_color = compute_average_color(frame, contour)
            
            # Формируем запись с признаками
            features_record = {
                'center_x': center[0],
                'center_y': center[1],
                'area': shape_features['area'],
                'perimeter': shape_features['perimeter'],
                'extent': shape_features['extent'],
                'circularity': shape_features['circularity'],
                'solidity': shape_features['solidity'],
                'convexity_defects_count': shape_features['convexity_defects_count'],
                'avg_defect_depth': shape_features['avg_defect_depth'],
                'aspect_ratio': shape_features['aspect_ratio'],
                'avg_color': avg_color
            }
            
            # Распознаем объект, если есть конфигурация
            category = None
            if self.config and "identification" in self.config:
                features_list = [
                    'area', 'perimeter', 'extent', 'circularity', 'solidity',
                    'convexity_defects_count', 'avg_defect_depth', 'aspect_ratio'
                ]
                category = recognize_object(
                    features_record,
                    self.config["identification"],
                    features_list
                )
            
            # Добавляем детекцию
            detections.append({
                'center': center,
                'rect': rect,
                'contour': contour,
                'features': features_record,
                'category': category
            })
        
        # Обновляем треки
        objects_info = self.update_tracks(detections)
        
        # Рисуем информацию о треках на визуализации
        for obj in objects_info:
            # Рисуем идентификатор объекта
            cv2.putText(
                vis,
                f"ID: {obj['id']}",
                (obj['center'][0] + 10, obj['center'][1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )
            
            # Рисуем категорию объекта, если она определена
            if obj['category']:
                cv2.putText(
                    vis,
                    f"Cat: {obj['category']}",
                    (obj['center'][0] + 10, obj['center'][1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
        
        return vis, objects_info
    
    def update_tracks(self, detections):
        """
        Обновляет треки объектов на основе новых детекций.
        
        Args:
            detections: Список новых детекций
            
        Returns:
            list: Информация об объектах
        """
        # Если нет треков, создаем новые для всех детекций
        if not self.tracks:
            for detection in detections:
                self.tracks.append({
                    'id': self.next_id,
                    'center': detection['center'],
                    'rect': detection['rect'],
                    'contour': detection['contour'],
                    'features': detection['features'],
                    'category': detection['category'],
                    'lost_frames': 0
                })
                self.next_id += 1
            
            return self.get_objects_info()
        
        # Если нет детекций, увеличиваем счетчик потерянных кадров для всех треков
        if not detections:
            for track in self.tracks:
                track['lost_frames'] += 1
            
            # Удаляем треки, которые потеряны слишком долго
            self.tracks = [track for track in self.tracks if track['lost_frames'] < MAX_LOST_FRAMES]
            
            return self.get_objects_info()
        
        # Создаем матрицу расстояний между треками и детекциями
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                # Вычисляем евклидово расстояние между центрами
                dx = track['center'][0] - detection['center'][0]
                dy = track['center'][1] - detection['center'][1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Если расстояние слишком большое, устанавливаем бесконечное значение
                if distance > MAX_DISTANCE:
                    cost_matrix[i, j] = float('inf')
                else:
                    cost_matrix[i, j] = distance
        
        # Решаем задачу о назначениях
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Помечаем все треки как непривязанные
        unmatched_tracks = list(range(len(self.tracks)))
        
        # Помечаем все детекции как непривязанные
        unmatched_detections = list(range(len(detections)))
        
        # Обновляем привязанные треки
        for row, col in zip(row_indices, col_indices):
            # Проверяем, что расстояние не бесконечное
            if cost_matrix[row, col] != float('inf'):
                # Обновляем трек
                self.tracks[row]['center'] = detections[col]['center']
                self.tracks[row]['rect'] = detections[col]['rect']
                self.tracks[row]['contour'] = detections[col]['contour']
                self.tracks[row]['features'] = detections[col]['features']
                
                # Обновляем категорию, только если она определена
                if detections[col]['category']:
                    self.tracks[row]['category'] = detections[col]['category']
                
                # Сбрасываем счетчик потерянных кадров
                self.tracks[row]['lost_frames'] = 0
                
                # Удаляем из списков непривязанных
                unmatched_tracks.remove(row)
                unmatched_detections.remove(col)
        
        # Увеличиваем счетчик потерянных кадров для непривязанных треков
        for i in unmatched_tracks:
            self.tracks[i]['lost_frames'] += 1
        
        # Удаляем треки, которые потеряны слишком долго
        self.tracks = [track for i, track in enumerate(self.tracks) if i not in unmatched_tracks or track['lost_frames'] < MAX_LOST_FRAMES]
        
        # Создаем новые треки для непривязанных детекций
        for i in unmatched_detections:
            self.tracks.append({
                'id': self.next_id,
                'center': detections[i]['center'],
                'rect': detections[i]['rect'],
                'contour': detections[i]['contour'],
                'features': detections[i]['features'],
                'category': detections[i]['category'],
                'lost_frames': 0
            })
            self.next_id += 1
        
        return self.get_objects_info()
    
    def get_objects_info(self):
        """
        Возвращает информацию об объектах для внешнего использования.
        
        Returns:
            list: Информация об объектах
        """
        objects_info = []
        
        for track in self.tracks:
            # Получаем размеры объекта
            _, (width, height), angle = track['rect']
            
            objects_info.append({
                'id': track['id'],
                'center': track['center'],
                'width': width,
                'height': height,
                'angle': angle,
                'category': track['category'],
                'features': track['features']
            })
        
        return objects_info
    
    def update_parameters(self, blur_kernel=None, canny_low=None, canny_high=None, min_area=None, max_area=None):
        """
        Обновляет параметры обработки.
        
        Args:
            blur_kernel: Размер ядра для размытия
            canny_low: Нижний порог для детектора Canny
            canny_high: Верхний порог для детектора Canny
            min_area: Минимальная площадь контура
            max_area: Максимальная площадь контура
            
        Returns:
            bool: Успешность обновления
        """
        if blur_kernel is not None:
            self.blur_kernel = make_odd(blur_kernel)
        
        if canny_low is not None:
            self.canny_low = canny_low
        
        if canny_high is not None:
            self.canny_high = canny_high
        
        if min_area is not None:
            self.min_area = min_area
        
        if max_area is not None:
            self.max_area = max_area
        
        # Сохраняем обновленные параметры в конфигурацию
        return self.save_config()
