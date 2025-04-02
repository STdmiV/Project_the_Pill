#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
working_mode.py - Модуль рабочего режима

Этот модуль реализует режим работы в реальном времени для обработки видеопотока
с камеры или из видеофайла. В этом режиме применяется коррекция искажений,
выводится обработанное изображение с наложенной информацией и осуществляется
контроль завершения работы.
"""

import cv2
import numpy as np
import logging
import time
from variables import TARGET_WIDTH, TARGET_HEIGHT_DISPLAY
from robot_comm import send_data_to_robot

# Настройка логирования
logging.basicConfig(
    filename='error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_working_mode(source, detector, camera_matrix=None, dist_coeffs=None, 
                    send_to_robot=False, display_callback=None):
    """
    Запускает рабочий режим обработки видеопотока.
    
    Args:
        source: Индекс камеры (целое число) или путь к видеофайлу (строка)
        detector: Экземпляр ObjectDetector для обработки кадров
        camera_matrix: Матрица камеры для коррекции дисторсии
        dist_coeffs: Коэффициенты дисторсии
        send_to_robot: Флаг отправки данных на робота
        display_callback: Функция обратного вызова для отображения кадра в GUI
        
    Returns:
        bool: Успешность выполнения
    """
    try:
        # Инициализируем видеозахват
        cap = cv2.VideoCapture(source)
        
        # Проверяем, открыт ли видеозахват
        if not cap.isOpened():
            logger.error(f"Не удалось открыть видеопоток: {source}")
            return False
        
        # Получаем частоту кадров
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Значение по умолчанию, если не удалось получить
        
        # Вычисляем задержку между кадрами
        frame_delay = int(1000 / fps)
        
        # Флаг для контроля цикла
        running = True
        
        # Счетчик кадров
        frame_count = 0
        
        # Время начала обработки
        start_time = time.time()
        
        # Основной цикл обработки
        while running:
            # Читаем кадр
            ret, frame = cap.read()
            
            # Если кадр не прочитан, пробуем сбросить позицию видеозахвата
            if not ret:
                # Для видеофайла можно сбросить позицию для циклического воспроизведения
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    break
            
            # Применяем коррекцию дисторсии, если параметры предоставлены
            if camera_matrix is not None and dist_coeffs is not None:
                frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
            
            # Масштабируем кадр для отображения
            h, w = frame.shape[:2]
            scale = TARGET_WIDTH / w
            display_frame = cv2.resize(frame, (TARGET_WIDTH, int(h * scale)))
            
            # Обрабатываем кадр
            vis_frame, objects_info = detector.process_frame(frame, update_config=False)
            
            # Масштабируем визуализацию для отображения
            h, w = vis_frame.shape[:2]
            scale = TARGET_WIDTH / w
            vis_display = cv2.resize(vis_frame, (TARGET_WIDTH, int(h * scale)))
            
            # Накладываем информационный текст
            cv2.putText(
                vis_display,
                f"Working Mode - Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Отправляем данные на робота, если требуется
            if send_to_robot and objects_info:
                for obj in objects_info:
                    # Преобразуем координаты в миллиметры (примерное преобразование)
                    # В реальной системе нужно использовать калибровку для точного преобразования
                    x_mm = obj['center'][0] * 0.264583  # Примерное преобразование пикселей в мм
                    y_mm = obj['center'][1] * 0.264583
                    width_mm = obj['width'] * 0.264583
                    height_mm = obj['height'] * 0.264583
                    
                    # Отправляем данные
                    send_data_to_robot(
                        obj_id=obj['id'],
                        group_name=obj['category'] or "unknown",
                        x_mm=x_mm,
                        y_mm=y_mm,
                        width_mm=width_mm,
                        height_mm=height_mm,
                        angle=obj['angle']
                    )
            
            # Если предоставлена функция обратного вызова для отображения, используем её
            if display_callback:
                running = display_callback(vis_display)
            else:
                # Иначе отображаем кадр в окне OpenCV
                cv2.imshow("Working Mode", vis_display)
                
                # Ждем нажатия клавиши
                key = cv2.waitKey(frame_delay) & 0xFF
                
                # Выход при нажатии 'q'
                if key == ord('q'):
                    running = False
            
            # Увеличиваем счетчик кадров
            frame_count += 1
            
            # Вычисляем и выводим FPS каждые 30 кадров
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps_actual = frame_count / elapsed_time
                print(f"FPS: {fps_actual:.2f}")
        
        # Освобождаем ресурсы
        cap.release()
        
        # Закрываем все окна, если не используется функция обратного вызова
        if not display_callback:
            cv2.destroyAllWindows()
        
        return True
    
    except Exception as e:
        logger.error(f"Ошибка в рабочем режиме: {str(e)}")
        return False

def get_video_properties(source):
    """
    Получает свойства видеопотока.
    
    Args:
        source: Индекс камеры (целое число) или путь к видеофайлу (строка)
        
    Returns:
        dict: Свойства видеопотока или None в случае ошибки
    """
    try:
        # Инициализируем видеозахват
        cap = cv2.VideoCapture(source)
        
        # Проверяем, открыт ли видеозахват
        if not cap.isOpened():
            logger.error(f"Не удалось открыть видеопоток: {source}")
            return None
        
        # Получаем свойства
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Освобождаем ресурсы
        cap.release()
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count
        }
    
    except Exception as e:
        logger.error(f"Ошибка при получении свойств видеопотока: {str(e)}")
        return None
