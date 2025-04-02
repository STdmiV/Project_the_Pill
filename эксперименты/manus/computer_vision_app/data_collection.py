#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_collection.py - Модуль сбора данных

Этот модуль реализует интерактивный интерфейс для ручной маркировки объектов,
когда автоматическое распознавание не является надёжным. Позволяет пользователю
выбрать форму и цвет объекта через показ композитного изображения.
"""

import cv2
import numpy as np
import os
import csv
import logging
from datetime import datetime
from variables import BOTTOM_HEIGHT, CSV_DIRECTORY

# Настройка логирования
logging.basicConfig(
    filename='error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def label_object_composite(top_img, roi_img, window_name="Label Object", 
                          bottom_height=BOTTOM_HEIGHT, prev_selection=None):
    """
    Реализует интерактивный интерфейс для ручной маркировки объектов.
    
    Args:
        top_img: Верхнее изображение (результат обработки)
        roi_img: Изображение региона интереса (ROI)
        window_name: Имя окна для отображения
        bottom_height: Высота нижней панели
        prev_selection: Предыдущий выбор (shape, color)
        
    Returns:
        tuple: (shape, color) или None, если выбор отменен
    """
    if top_img is None or roi_img is None:
        logger.error("Ошибка в label_object_composite: изображения не предоставлены")
        return None
    
    # Создаем копии изображений
    top_img_copy = top_img.copy()
    roi_img_copy = roi_img.copy()
    
    # Получаем размеры верхнего изображения
    h, w = top_img_copy.shape[:2]
    
    # Создаем нижнюю панель
    bottom_panel = np.zeros((bottom_height, w, 3), dtype=np.uint8)
    
    # Масштабируем ROI для отображения на нижней панели
    roi_height = bottom_height - 20  # Оставляем место для текста
    roi_width = int(roi_height * roi_img_copy.shape[1] / roi_img_copy.shape[0])
    roi_resized = cv2.resize(roi_img_copy, (roi_width, roi_height))
    
    # Размещаем ROI на нижней панели
    roi_x = 10
    roi_y = 10
    bottom_panel[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = roi_resized
    
    # Добавляем инструкции для выбора формы
    instructions = "Select shape: 1-circular, 2-rhombus, 3-cylinder, 4-skip, [Space]-prev"
    cv2.putText(
        bottom_panel,
        instructions,
        (roi_x + roi_width + 20, roi_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    
    # Объединяем верхнее изображение и нижнюю панель
    combined_img = np.vstack((top_img_copy, bottom_panel))
    
    # Отображаем комбинированное изображение
    cv2.imshow(window_name, combined_img)
    
    # Ждем нажатия клавиши для выбора формы
    shape = None
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        # Если нажата клавиша пробела и есть предыдущий выбор, возвращаем его
        if key == ord(' ') and prev_selection:
            return prev_selection
        
        # Выбор формы
        if key == ord('1'):
            shape = "circular"
            break
        elif key == ord('2'):
            shape = "rhombus"
            break
        elif key == ord('3'):
            shape = "cylinder"
            break
        elif key == ord('4'):
            return None  # Пропустить объект
        elif key == 27:  # ESC
            return None  # Отмена
    
    # Создаем новую нижнюю панель для выбора цвета
    bottom_panel = np.zeros((bottom_height, w, 3), dtype=np.uint8)
    
    # Размещаем ROI на нижней панели
    bottom_panel[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = roi_resized
    
    # Добавляем инструкции для выбора цвета
    instructions = "Select color: q-white, w-pink, e-black, [Space]-prev"
    cv2.putText(
        bottom_panel,
        instructions,
        (roi_x + roi_width + 20, roi_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    
    # Объединяем верхнее изображение и нижнюю панель
    combined_img = np.vstack((top_img_copy, bottom_panel))
    
    # Отображаем комбинированное изображение
    cv2.imshow(window_name, combined_img)
    
    # Ждем нажатия клавиши для выбора цвета
    color = None
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        # Если нажата клавиша пробела, возвращаемся к выбору формы
        if key == ord(' '):
            return label_object_composite(top_img, roi_img, window_name, bottom_height, prev_selection)
        
        # Выбор цвета
        if key == ord('q'):
            color = "white"
            break
        elif key == ord('w'):
            color = "pink"
            break
        elif key == ord('e'):
            color = "black"
            break
        elif key == 27:  # ESC
            return None  # Отмена
    
    return (shape, color)

def save_object_data(obj_data, shape, color):
    """
    Сохраняет данные об объекте в CSV-файл.
    
    Args:
        obj_data: Словарь с данными об объекте
        shape: Форма объекта
        color: Цвет объекта
        
    Returns:
        bool: Успешность сохранения
    """
    try:
        # Создаем имя файла на основе формы и цвета
        filename = f"{shape}_{color}.csv"
        filepath = os.path.join(CSV_DIRECTORY, filename)
        
        # Проверяем, существует ли файл
        file_exists = os.path.isfile(filepath)
        
        # Открываем файл для добавления
        with open(filepath, 'a', newline='') as csvfile:
            # Определяем заголовки
            fieldnames = [
                'timestamp', 'id', 'center_x', 'center_y', 'width', 'height', 
                'angle', 'area', 'perimeter', 'extent', 'circularity', 'solidity',
                'convexity_defects_count', 'avg_defect_depth', 'aspect_ratio', 
                'avg_color', 'hu_moments'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Записываем заголовки, если файл новый
            if not file_exists:
                writer.writeheader()
            
            # Подготавливаем данные для записи
            features = obj_data['features']
            row = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'id': obj_data['id'],
                'center_x': obj_data['center'][0],
                'center_y': obj_data['center'][1],
                'width': obj_data['width'],
                'height': obj_data['height'],
                'angle': obj_data['angle'],
                'area': features['area'],
                'perimeter': features['perimeter'],
                'extent': features['extent'],
                'circularity': features['circularity'],
                'solidity': features['solidity'],
                'convexity_defects_count': features['convexity_defects_count'],
                'avg_defect_depth': features['avg_defect_depth'],
                'aspect_ratio': features['aspect_ratio'],
                'avg_color': str(features['avg_color']),
                'hu_moments': str(features.get('hu_moments', []))
            }
            
            # Записываем данные
            writer.writerow(row)
            
            return True
    
    except Exception as e:
        logger.error(f"Ошибка при сохранении данных объекта: {str(e)}")
        return False

def extract_roi(frame, contour, padding=20):
    """
    Извлекает регион интереса (ROI) вокруг контура.
    
    Args:
        frame: Исходное изображение
        contour: Контур объекта
        padding: Отступ вокруг контура
        
    Returns:
        numpy.ndarray: Изображение ROI
    """
    # Получаем ограничивающий прямоугольник
    x, y, w, h = cv2.boundingRect(contour)
    
    # Добавляем отступ
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame.shape[1] - x, w + 2 * padding)
    h = min(frame.shape[0] - y, h + 2 * padding)
    
    # Извлекаем ROI
    roi = frame[y:y+h, x:x+w]
    
    return roi

def process_unlabeled_objects(frame, objects_info, detector):
    """
    Обрабатывает объекты без категории, запрашивая ручную маркировку.
    
    Args:
        frame: Исходное изображение
        objects_info: Информация об объектах
        detector: Экземпляр ObjectDetector
        
    Returns:
        list: Обновленная информация об объектах
    """
    # Создаем копию изображения для визуализации
    vis_frame = frame.copy()
    
    # Обрабатываем каждый объект
    for obj in objects_info:
        # Если категория не определена
        if obj['category'] is None:
            # Находим контур объекта
            for track in detector.tracks:
                if track['id'] == obj['id']:
                    contour = track['contour']
                    
                    # Извлекаем ROI
                    roi = extract_roi(frame, contour)
                    
                    # Запрашиваем ручную маркировку
                    label = label_object_composite(vis_frame, roi)
                    
                    if label:
                        shape, color = label
                        
                        # Обновляем категорию объекта
                        obj['category'] = f"{shape}_{color}"
                        
                        # Обновляем категорию в треке
                        track['category'] = obj['category']
                        
                        # Сохраняем данные об объекте
                        save_object_data(obj, shape, color)
                    
                    break
    
    return objects_info
