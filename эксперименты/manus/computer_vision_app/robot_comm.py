#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
robot_comm.py - Модуль коммуникации с роботом

Этот модуль обеспечивает передачу данных обнаруженных объектов на систему управления роботом.
На начальном этапе реализуется демонстрационный вариант с выводом параметров в консоль,
но структура модуля рассчитана на дальнейшее расширение.
"""

import logging
import json
import time

# Настройка логирования
logging.basicConfig(
    filename='error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def send_data_to_robot(obj_id, group_name, x_mm, y_mm, width_mm, height_mm, angle):
    """
    Отправляет данные об объекте на робота.
    
    Args:
        obj_id: Уникальный идентификатор объекта
        group_name: Категория или группа объекта
        x_mm: Координата X объекта в миллиметрах
        y_mm: Координата Y объекта в миллиметрах
        width_mm: Ширина объекта в миллиметрах
        height_mm: Высота объекта в миллиметрах
        angle: Угол поворота объекта в градусах
        
    Returns:
        bool: Успешность отправки
    """
    try:
        # Формируем данные для отправки
        data = {
            'id': obj_id,
            'group': group_name,
            'position': {
                'x': round(x_mm, 2),
                'y': round(y_mm, 2)
            },
            'size': {
                'width': round(width_mm, 2),
                'height': round(height_mm, 2)
            },
            'angle': round(angle, 2),
            'timestamp': time.time()
        }
        
        # Преобразуем в JSON
        json_data = json.dumps(data)
        
        # На начальном этапе просто выводим данные в консоль
        print(f"[ROBOT] Sending data: {json_data}")
        
        # Здесь в будущем будет реализована реальная отправка данных
        # Например, через последовательный порт, сеть или API
        
        # Пример для будущей реализации с использованием последовательного порта:
        # import serial
        # ser = serial.Serial('/dev/ttyUSB0', 9600)
        # ser.write(json_data.encode())
        # ser.close()
        
        # Пример для будущей реализации с использованием сетевого сокета:
        # import socket
        # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # sock.connect(('robot_ip', 12345))
        # sock.sendall(json_data.encode())
        # sock.close()
        
        return True
    
    except Exception as e:
        logger.error(f"Ошибка при отправке данных на робота: {str(e)}")
        return False

class RobotCommunicator:
    """
    Класс для управления коммуникацией с роботом.
    Предназначен для будущего расширения функциональности.
    """
    
    def __init__(self, connection_type='console', connection_params=None):
        """
        Инициализирует коммуникатор с роботом.
        
        Args:
            connection_type: Тип соединения ('console', 'serial', 'tcp')
            connection_params: Параметры соединения (зависят от типа)
        """
        self.connection_type = connection_type
        self.connection_params = connection_params or {}
        self.connected = False
        self.connection = None
    
    def connect(self):
        """
        Устанавливает соединение с роботом.
        
        Returns:
            bool: Успешность соединения
        """
        try:
            if self.connection_type == 'console':
                # Для консольного вывода не требуется реальное соединение
                self.connected = True
                return True
            
            elif self.connection_type == 'serial':
                # Пример реализации для последовательного порта
                # import serial
                # port = self.connection_params.get('port', '/dev/ttyUSB0')
                # baudrate = self.connection_params.get('baudrate', 9600)
                # self.connection = serial.Serial(port, baudrate)
                # self.connected = True
                pass
            
            elif self.connection_type == 'tcp':
                # Пример реализации для TCP-соединения
                # import socket
                # host = self.connection_params.get('host', 'localhost')
                # port = self.connection_params.get('port', 12345)
                # self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # self.connection.connect((host, port))
                # self.connected = True
                pass
            
            return self.connected
        
        except Exception as e:
            logger.error(f"Ошибка при установке соединения с роботом: {str(e)}")
            self.connected = False
            return False
    
    def disconnect(self):
        """
        Закрывает соединение с роботом.
        
        Returns:
            bool: Успешность закрытия соединения
        """
        try:
            if not self.connected:
                return True
            
            if self.connection_type == 'console':
                # Для консольного вывода не требуется закрытие соединения
                self.connected = False
                return True
            
            elif self.connection_type in ['serial', 'tcp'] and self.connection:
                # Закрываем соединение
                self.connection.close()
                self.connected = False
                return True
            
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединения с роботом: {str(e)}")
            return False
    
    def send_data(self, obj_id, group_name, x_mm, y_mm, width_mm, height_mm, angle):
        """
        Отправляет данные об объекте на робота.
        
        Args:
            obj_id: Уникальный идентификатор объекта
            group_name: Категория или группа объекта
            x_mm: Координата X объекта в миллиметрах
            y_mm: Координата Y объекта в миллиметрах
            width_mm: Ширина объекта в миллиметрах
            height_mm: Высота объекта в миллиметрах
            angle: Угол поворота объекта в градусах
            
        Returns:
            bool: Успешность отправки
        """
        if not self.connected:
            if not self.connect():
                return False
        
        return send_data_to_robot(obj_id, group_name, x_mm, y_mm, width_mm, height_mm, angle)
