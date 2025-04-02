#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py - Основной модуль приложения

Этот модуль организует запуск всего приложения, инициализирует пользовательский интерфейс (GUI)
и объединяет работу всех модулей (калибровка, обработка изображений, сбор данных, анализ,
рабочий режим, передача данных).
"""

import sys
import os
import cv2
import numpy as np
import logging
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QRadioButton, QButtonGroup, QGroupBox, QLineEdit,
    QMessageBox, QSplitter, QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer, pyqtSlot

# Импортируем модули приложения
from variables import *
from calibration import CalibrationManager
from computer_vision import ObjectDetector
from data_collection import process_unlabeled_objects, extract_roi, label_object_composite, save_object_data
from analysis import load_and_process_data, analyze_data, get_feature_ranges
from working_mode import run_working_mode, get_video_properties
from robot_comm import RobotCommunicator

# Настройка логирования
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """
    Главное окно приложения с вкладками для разных режимов работы.
    """
    
    def __init__(self):
        super().__init__()
        
        # Инициализация компонентов
        self.calibration_manager = CalibrationManager()
        self.object_detector = ObjectDetector()
        self.robot_communicator = RobotCommunicator()
        
        # Флаги состояния
        self.is_running = False
        self.current_mode = None
        self.video_source = None
        self.cap = None
        
        # Настройка интерфейса
        self.init_ui()
        
        # Проверка калибровки при запуске
        self.check_calibration_on_startup()
    
    def init_ui(self):
        """
        Инициализирует пользовательский интерфейс.
        """
        # Настройка главного окна
        self.setWindowTitle("Computer Vision Application")
        self.setGeometry(100, 100, 1200, 800)
        
        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Создаем основной макет
        main_layout = QVBoxLayout(central_widget)
        
        # Создаем виджет с вкладками
        self.tabs = QTabWidget()
        
        # Создаем вкладки для разных режимов
        self.tab_calibration = QWidget()
        self.tab_data_collection = QWidget()
        self.tab_analysis = QWidget()
        self.tab_working = QWidget()
        self.tab_settings = QWidget()
        
        # Добавляем вкладки в виджет
        self.tabs.addTab(self.tab_calibration, "Калибровка")
        self.tabs.addTab(self.tab_data_collection, "Сбор данных")
        self.tabs.addTab(self.tab_analysis, "Анализ")
        self.tabs.addTab(self.tab_working, "Рабочий режим")
        self.tabs.addTab(self.tab_settings, "Настройки")
        
        # Инициализируем содержимое вкладок
        self.init_calibration_tab()
        self.init_data_collection_tab()
        self.init_analysis_tab()
        self.init_working_tab()
        self.init_settings_tab()
        
        # Добавляем виджет с вкладками в основной макет
        main_layout.addWidget(self.tabs)
        
        # Создаем строку состояния
        self.statusBar().showMessage("Готов к работе")
        
        # Создаем таймер для обновления видео
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Подключаем сигнал изменения вкладки
        self.tabs.currentChanged.connect(self.on_tab_changed)
    
    def init_calibration_tab(self):
        """
        Инициализирует вкладку калибровки.
        """
        # Создаем макет для вкладки
        layout = QVBoxLayout(self.tab_calibration)
        
        # Создаем верхнюю панель с кнопками
        top_panel = QHBoxLayout()
        
        # Кнопка для запуска калибровки
        self.btn_start_calibration = QPushButton("Запустить калибровку")
        self.btn_start_calibration.clicked.connect(self.on_start_calibration)
        top_panel.addWidget(self.btn_start_calibration)
        
        # Кнопка для сброса калибровки
        self.btn_reset_calibration = QPushButton("Сбросить калибровку")
        self.btn_reset_calibration.clicked.connect(self.on_reset_calibration)
        top_panel.addWidget(self.btn_reset_calibration)
        
        # Кнопка для сохранения калибровки
        self.btn_save_calibration = QPushButton("Сохранить калибровку")
        self.btn_save_calibration.clicked.connect(self.on_save_calibration)
        self.btn_save_calibration.setEnabled(False)
        top_panel.addWidget(self.btn_save_calibration)
        
        # Кнопка для генерации доски ChArUco
        self.btn_generate_board = QPushButton("Сгенерировать доску ChArUco")
        self.btn_generate_board.clicked.connect(self.on_generate_board)
        top_panel.addWidget(self.btn_generate_board)
        
        # Добавляем верхнюю панель в макет
        layout.addLayout(top_panel)
        
        # Создаем панель для выбора источника видео
        source_panel = QHBoxLayout()
        
        # Группа радиокнопок для выбора источника
        self.source_group = QButtonGroup()
        
        # Радиокнопка для выбора камеры
        self.radio_camera = QRadioButton("Камера")
        self.radio_camera.setChecked(True)
        self.source_group.addButton(self.radio_camera, 0)
        source_panel.addWidget(self.radio_camera)
        
        # Радиокнопка для выбора видеофайла
        self.radio_video = QRadioButton("Видеофайл")
        self.source_group.addButton(self.radio_video, 1)
        source_panel.addWidget(self.radio_video)
        
        # Поле для ввода пути к видеофайлу
        self.txt_calibration_video_path = QLineEdit()
        self.txt_calibration_video_path.setPlaceholderText("Путь к видеофайлу для калибровки")
        self.txt_calibration_video_path.setEnabled(False)
        source_panel.addWidget(self.txt_calibration_video_path)
        
        # Кнопка для выбора видеофайла
        self.btn_browse_calibration_video = QPushButton("Обзор...")
        self.btn_browse_calibration_video.clicked.connect(self.on_browse_calibration_video)
        self.btn_browse_calibration_video.setEnabled(False)
        source_panel.addWidget(self.btn_browse_calibration_video)
        
        # Добавляем панель выбора источника в макет
        layout.addLayout(source_panel)
        
        # Подключаем сигналы радиокнопок
        self.radio_camera.toggled.connect(self.on_source_toggled)
        self.radio_video.toggled.connect(self.on_source_toggled)
        
        # Создаем метку для отображения видео
        self.lbl_calibration_video = QLabel()
        self.lbl_calibration_video.setAlignment(Qt.AlignCenter)
        self.lbl_calibration_video.setMinimumSize(640, 480)
        self.lbl_calibration_video.setStyleSheet("background-color: black;")
        layout.addWidget(self.lbl_calibration_video)
        
        # Создаем метку для отображения статуса калибровки
        self.lbl_calibration_status = QLabel("Статус калибровки: Не выполнена")
        layout.addWidget(self.lbl_calibration_status)
        
        # Обновляем статус калибровки
        self.update_calibration_status()
    
    def init_data_collection_tab(self):
        """
        Инициализирует вкладку сбора данных.
        """
        # Создаем макет для вкладки
        layout = QVBoxLayout(self.tab_data_collection)
        
        # Создаем верхнюю панель с кнопками
        top_panel = QHBoxLayout()
        
        # Кнопка для запуска сбора данных
        self.btn_start_collection = QPushButton("Запустить сбор данных")
        self.btn_start_collection.clicked.connect(self.on_start_collection)
        top_panel.addWidget(self.btn_start_collection)
        
        # Кнопка для остановки сбора данных
        self.btn_stop_collection = QPushButton("Остановить сбор данных")
        self.btn_stop_collection.clicked.connect(self.on_stop_collection)
        self.btn_stop_collection.setEnabled(False)
        top_panel.addWidget(self.btn_stop_collection)
        
        # Добавляем верхнюю панель в макет
        layout.addLayout(top_panel)
        
        # Создаем панель для выбора источника видео
        source_panel = QHBoxLayout()
        
        # Группа радиокнопок для выбора источника
        self.collection_source_group = QButtonGroup()
        
        # Радиокнопка для выбора камеры
        self.radio_collection_camera = QRadioButton("Камера")
        self.radio_collection_camera.setChecked(True)
        self.collection_source_group.addButton(self.radio_collection_camera, 0)
        source_panel.addWidget(self.radio_collection_camera)
        
        # Радиокнопка для выбора видеофайла
        self.radio_collection_video = QRadioButton("Видеофайл")
        self.collection_source_group.addButton(self.radio_collection_video, 1)
        source_panel.addWidget(self.radio_collection_video)
        
        # Поле для ввода пути к видеофайлу
        self.txt_collection_video_path = QLineEdit()
        self.txt_collection_video_path.setPlaceholderText("Путь к видеофайлу для сбора данных")
        self.txt_collection_video_path.setEnabled(False)
        source_panel.addWidget(self.txt_collection_video_path)
        
        # Кнопка для выбора видеофайла
        self.btn_browse_collection_video = QPushButton("Обзор...")
        self.btn_browse_collection_video.clicked.connect(self.on_browse_collection_video)
        self.btn_browse_collection_video.setEnabled(False)
        source_panel.addWidget(self.btn_browse_collection_video)
        
        # Добавляем панель выбора источника в макет
        layout.addLayout(source_panel)
        
        # Подключаем сигналы радиокнопок
        self.radio_collection_camera.toggled.connect(self.on_collection_source_toggled)
        self.radio_collection_video.toggled.connect(self.on_collection_source_toggled)
        
        # Создаем метку для отображения видео
        self.lbl_collection_video = QLabel()
        self.lbl_collection_video.setAlignment(Qt.AlignCenter)
        self.lbl_collection_video.setMinimumSize(640, 480)
        self.lbl_collection_video.setStyleSheet("background-color: black;")
        layout.addWidget(self.lbl_collection_video)
        
        # Создаем метку для отображения статуса сбора данных
        self.lbl_collection_status = QLabel("Статус сбора данных: Не запущен")
        layout.addWidget(self.lbl_collection_status)
    
    def init_analysis_tab(self):
        """
        Инициализирует вкладку анализа данных.
        """
        # Создаем макет для вкладки
        layout = QVBoxLayout(self.tab_analysis)
        
        # Создаем верхнюю панель с элементами управления
        top_panel = QHBoxLayout()
        
        # Метка для выбора файла
        top_panel.addWidget(QLabel("Выберите файл данных:"))
        
        # Выпадающий список для выбора файла
        self.cmb_analysis_file = QComboBox()
        self.cmb_analysis_file.setMinimumWidth(300)
        top_panel.addWidget(self.cmb_analysis_file)
        
        # Кнопка для обновления списка файлов
        self.btn_refresh_files = QPushButton("Обновить список")
        self.btn_refresh_files.clicked.connect(self.on_refresh_files)
        top_panel.addWidget(self.btn_refresh_files)
        
        # Кнопка для анализа данных
        self.btn_analyze_data = QPushButton("Анализировать данные")
        self.btn_analyze_data.clicked.connect(self.on_analyze_data)
        top_panel.addWidget(self.btn_analyze_data)
        
        # Кнопка для генерации диапазонов
        self.btn_generate_ranges = QPushButton("Сгенерировать диапазоны")
        self.btn_generate_ranges.clicked.connect(self.on_generate_ranges)
        top_panel.addWidget(self.btn_generate_ranges)
        
        # Добавляем верхнюю панель в макет
        layout.addLayout(top_panel)
        
        # Создаем метку для отображения результатов анализа
        self.lbl_analysis_results = QLabel("Выберите файл и нажмите 'Анализировать данные'")
        self.lbl_analysis_results.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_analysis_results.setWordWrap(True)
        self.lbl_analysis_results.setMinimumHeight(400)
        self.lbl_analysis_results.setStyleSheet("background-color: white; padding: 10px;")
        layout.addWidget(self.lbl_analysis_results)
        
        # Обновляем список файлов
        self.on_refresh_files()
    
    def init_working_tab(self):
        """
        Инициализирует вкладку рабочего режима.
        """
        # Создаем макет для вкладки
        layout = QVBoxLayout(self.tab_working)
        
        # Создаем верхнюю панель с кнопками
        top_panel = QHBoxLayout()
        
        # Кнопка для запуска рабочего режима
        self.btn_start_working = QPushButton("Запустить рабочий режим")
        self.btn_start_working.clicked.connect(self.on_start_working)
        top_panel.addWidget(self.btn_start_working)
        
        # Кнопка для остановки рабочего режима
        self.btn_stop_working = QPushButton("Остановить рабочий режим")
        self.btn_stop_working.clicked.connect(self.on_stop_working)
        self.btn_stop_working.setEnabled(False)
        top_panel.addWidget(self.btn_stop_working)
        
        # Флажок для отправки данных на робота
        self.chk_send_to_robot = QCheckBox("Отправлять данные на робота")
        top_panel.addWidget(self.chk_send_to_robot)
        
        # Добавляем верхнюю панель в макет
        layout.addLayout(top_panel)
        
        # Создаем панель для выбора источника видео
        source_panel = QHBoxLayout()
        
        # Группа радиокнопок для выбора источника
        self.working_source_group = QButtonGroup()
        
        # Радиокнопка для выбора камеры
        self.radio_working_camera = QRadioButton("Камера")
        self.radio_working_camera.setChecked(True)
        self.working_source_group.addButton(self.radio_working_camera, 0)
        source_panel.addWidget(self.radio_working_camera)
        
        # Радиокнопка для выбора видеофайла
        self.radio_working_video = QRadioButton("Видеофайл")
        self.working_source_group.addButton(self.radio_working_video, 1)
        source_panel.addWidget(self.radio_working_video)
        
        # Поле для ввода пути к видеофайлу
        self.txt_working_video_path = QLineEdit()
        self.txt_working_video_path.setPlaceholderText("Путь к видеофайлу для обработки")
        self.txt_working_video_path.setEnabled(False)
        source_panel.addWidget(self.txt_working_video_path)
        
        # Кнопка для выбора видеофайла
        self.btn_browse_working_video = QPushButton("Обзор...")
        self.btn_browse_working_video.clicked.connect(self.on_browse_working_video)
        self.btn_browse_working_video.setEnabled(False)
        source_panel.addWidget(self.btn_browse_working_video)
        
        # Добавляем панель выбора источника в макет
        layout.addLayout(source_panel)
        
        # Подключаем сигналы радиокнопок
        self.radio_working_camera.toggled.connect(self.on_working_source_toggled)
        self.radio_working_video.toggled.connect(self.on_working_source_toggled)
        
        # Создаем метку для отображения видео
        self.lbl_working_video = QLabel()
        self.lbl_working_video.setAlignment(Qt.AlignCenter)
        self.lbl_working_video.setMinimumSize(640, 480)
        self.lbl_working_video.setStyleSheet("background-color: black;")
        layout.addWidget(self.lbl_working_video)
        
        # Создаем метку для отображения статуса рабочего режима
        self.lbl_working_status = QLabel("Статус рабочего режима: Не запущен")
        layout.addWidget(self.lbl_working_status)
    
    def init_settings_tab(self):
        """
        Инициализирует вкладку настроек.
        """
        # Создаем макет для вкладки
        layout = QVBoxLayout(self.tab_settings)
        
        # Создаем группу для настроек детекции
        detection_group = QGroupBox("Настройки детекции")
        detection_layout = QVBoxLayout(detection_group)
        
        # Создаем слайдеры для настройки параметров
        # Слайдер для нижнего порога Canny
        canny_low_layout = QHBoxLayout()
        canny_low_layout.addWidget(QLabel("Нижний порог Canny:"))
        self.slider_canny_low = QSlider(Qt.Horizontal)
        self.slider_canny_low.setRange(0, 255)
        self.slider_canny_low.setValue(CANNY_LOW)
        self.slider_canny_low.valueChanged.connect(self.on_canny_low_changed)
        canny_low_layout.addWidget(self.slider_canny_low)
        self.lbl_canny_low = QLabel(str(CANNY_LOW))
        canny_low_layout.addWidget(self.lbl_canny_low)
        detection_layout.addLayout(canny_low_layout)
        
        # Слайдер для верхнего порога Canny
        canny_high_layout = QHBoxLayout()
        canny_high_layout.addWidget(QLabel("Верхний порог Canny:"))
        self.slider_canny_high = QSlider(Qt.Horizontal)
        self.slider_canny_high.setRange(0, 255)
        self.slider_canny_high.setValue(CANNY_HIGH)
        self.slider_canny_high.valueChanged.connect(self.on_canny_high_changed)
        canny_high_layout.addWidget(self.slider_canny_high)
        self.lbl_canny_high = QLabel(str(CANNY_HIGH))
        canny_high_layout.addWidget(self.lbl_canny_high)
        detection_layout.addLayout(canny_high_layout)
        
        # Слайдер для размера ядра размытия
        blur_kernel_layout = QHBoxLayout()
        blur_kernel_layout.addWidget(QLabel("Размер ядра размытия:"))
        self.slider_blur_kernel = QSlider(Qt.Horizontal)
        self.slider_blur_kernel.setRange(1, 21)
        self.slider_blur_kernel.setValue(BLUR_KERNEL)
        self.slider_blur_kernel.setSingleStep(2)
        self.slider_blur_kernel.valueChanged.connect(self.on_blur_kernel_changed)
        blur_kernel_layout.addWidget(self.slider_blur_kernel)
        self.lbl_blur_kernel = QLabel(str(BLUR_KERNEL))
        blur_kernel_layout.addWidget(self.lbl_blur_kernel)
        detection_layout.addLayout(blur_kernel_layout)
        
        # Поля для минимальной и максимальной площади
        area_layout = QHBoxLayout()
        area_layout.addWidget(QLabel("Минимальная площадь:"))
        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(0, 100000)
        self.spin_min_area.setValue(MIN_AREA)
        self.spin_min_area.valueChanged.connect(self.on_min_area_changed)
        area_layout.addWidget(self.spin_min_area)
        
        area_layout.addWidget(QLabel("Максимальная площадь:"))
        self.spin_max_area = QSpinBox()
        self.spin_max_area.setRange(0, 1000000)
        self.spin_max_area.setValue(MAX_AREA)
        self.spin_max_area.valueChanged.connect(self.on_max_area_changed)
        area_layout.addWidget(self.spin_max_area)
        detection_layout.addLayout(area_layout)
        
        # Кнопка для сохранения настроек
        self.btn_save_settings = QPushButton("Сохранить настройки")
        self.btn_save_settings.clicked.connect(self.on_save_settings)
        detection_layout.addWidget(self.btn_save_settings)
        
        # Добавляем группу настроек детекции в макет
        layout.addWidget(detection_group)
        
        # Создаем группу для настроек камеры
        camera_group = QGroupBox("Настройки камеры")
        camera_layout = QVBoxLayout(camera_group)
        
        # Поле для индекса камеры
        camera_index_layout = QHBoxLayout()
        camera_index_layout.addWidget(QLabel("Индекс камеры:"))
        self.spin_camera_index = QSpinBox()
        self.spin_camera_index.setRange(0, 10)
        self.spin_camera_index.setValue(CAMERA_INDEX)
        camera_index_layout.addWidget(self.spin_camera_index)
        camera_layout.addLayout(camera_index_layout)
        
        # Добавляем группу настроек камеры в макет
        layout.addWidget(camera_group)
        
        # Добавляем растягивающийся элемент
        layout.addStretch()
    
    def check_calibration_on_startup(self):
        """
        Проверяет наличие калибровочных данных при запуске.
        """
        if not self.calibration_manager.is_calibrated():
            reply = QMessageBox.question(
                self,
                "Калибровка не выполнена",
                "Калибровка камеры не выполнена. Хотите выполнить калибровку сейчас?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.tabs.setCurrentIndex(0)  # Переключаемся на вкладку калибровки
                self.on_start_calibration()
    
    def update_calibration_status(self):
        """
        Обновляет статус калибровки.
        """
        if self.calibration_manager.is_calibrated():
            self.lbl_calibration_status.setText("Статус калибровки: Выполнена")
            self.btn_save_calibration.setEnabled(True)
        else:
            self.lbl_calibration_status.setText("Статус калибровки: Не выполнена")
            self.btn_save_calibration.setEnabled(False)
    
    def on_tab_changed(self, index):
        """
        Обрабатывает изменение активной вкладки.
        
        Args:
            index: Индекс новой активной вкладки
        """
        # Останавливаем текущий режим, если он запущен
        if self.is_running:
            self.stop_current_mode()
    
    def on_source_toggled(self):
        """
        Обрабатывает переключение источника видео для калибровки.
        """
        if self.radio_video.isChecked():
            self.txt_calibration_video_path.setEnabled(True)
            self.btn_browse_calibration_video.setEnabled(True)
        else:
            self.txt_calibration_video_path.setEnabled(False)
            self.btn_browse_calibration_video.setEnabled(False)
    
    def on_collection_source_toggled(self):
        """
        Обрабатывает переключение источника видео для сбора данных.
        """
        if self.radio_collection_video.isChecked():
            self.txt_collection_video_path.setEnabled(True)
            self.btn_browse_collection_video.setEnabled(True)
        else:
            self.txt_collection_video_path.setEnabled(False)
            self.btn_browse_collection_video.setEnabled(False)
    
    def on_working_source_toggled(self):
        """
        Обрабатывает переключение источника видео для рабочего режима.
        """
        if self.radio_working_video.isChecked():
            self.txt_working_video_path.setEnabled(True)
            self.btn_browse_working_video.setEnabled(True)
        else:
            self.txt_working_video_path.setEnabled(False)
            self.btn_browse_working_video.setEnabled(False)
    
    def on_browse_calibration_video(self):
        """
        Обрабатывает нажатие кнопки выбора видеофайла для калибровки.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видеофайл для калибровки",
            "",
            "Видеофайлы (*.mp4 *.avi *.mkv);;Все файлы (*.*)"
        )
        
        if file_path:
            self.txt_calibration_video_path.setText(file_path)
    
    def on_browse_collection_video(self):
        """
        Обрабатывает нажатие кнопки выбора видеофайла для сбора данных.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видеофайл для сбора данных",
            "",
            "Видеофайлы (*.mp4 *.avi *.mkv);;Все файлы (*.*)"
        )
        
        if file_path:
            self.txt_collection_video_path.setText(file_path)
    
    def on_browse_working_video(self):
        """
        Обрабатывает нажатие кнопки выбора видеофайла для рабочего режима.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видеофайл для обработки",
            "",
            "Видеофайлы (*.mp4 *.avi *.mkv);;Все файлы (*.*)"
        )
        
        if file_path:
            self.txt_working_video_path.setText(file_path)
    
    def on_start_calibration(self):
        """
        Обрабатывает нажатие кнопки запуска калибровки.
        """
        # Проверяем, не запущен ли уже какой-то режим
        if self.is_running:
            self.stop_current_mode()
        
        # Сбрасываем калибровку
        self.calibration_manager.reset()
        
        # Определяем источник видео
        if self.radio_camera.isChecked():
            self.video_source = self.spin_camera_index.value()
        else:
            video_path = self.txt_calibration_video_path.text()
            if not video_path:
                QMessageBox.warning(
                    self,
                    "Ошибка",
                    "Не указан путь к видеофайлу для калибровки"
                )
                return
            self.video_source = video_path
        
        # Открываем видеопоток
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось открыть видеопоток: {self.video_source}"
            )
            return
        
        # Устанавливаем флаги
        self.is_running = True
        self.current_mode = MODE_CALIBRATION
        
        # Обновляем интерфейс
        self.btn_start_calibration.setEnabled(False)
        self.btn_reset_calibration.setEnabled(True)
        self.btn_save_calibration.setEnabled(False)
        
        # Запускаем таймер для обновления кадров
        self.timer.start(30)  # ~30 FPS
        
        # Обновляем статус
        self.lbl_calibration_status.setText("Статус калибровки: В процессе")
        self.statusBar().showMessage("Калибровка запущена. Перемещайте доску ChArUco в разные положения.")
    
    def on_reset_calibration(self):
        """
        Обрабатывает нажатие кнопки сброса калибровки.
        """
        # Сбрасываем калибровку
        self.calibration_manager.reset()
        
        # Обновляем статус
        self.update_calibration_status()
        
        # Если режим калибровки запущен, останавливаем его
        if self.is_running and self.current_mode == MODE_CALIBRATION:
            self.stop_current_mode()
        
        # Обновляем интерфейс
        self.btn_start_calibration.setEnabled(True)
        self.btn_reset_calibration.setEnabled(True)
        self.btn_save_calibration.setEnabled(False)
        
        # Обновляем статус
        self.statusBar().showMessage("Калибровка сброшена")
    
    def on_save_calibration(self):
        """
        Обрабатывает нажатие кнопки сохранения калибровки.
        """
        # Сохраняем калибровку
        if self.calibration_manager.save_calibration():
            QMessageBox.information(
                self,
                "Успех",
                "Калибровочные данные успешно сохранены"
            )
            self.statusBar().showMessage("Калибровочные данные сохранены")
        else:
            QMessageBox.critical(
                self,
                "Ошибка",
                "Не удалось сохранить калибровочные данные"
            )
            self.statusBar().showMessage("Ошибка при сохранении калибровочных данных")
    
    def on_generate_board(self):
        """
        Обрабатывает нажатие кнопки генерации доски ChArUco.
        """
        # Генерируем изображение доски
        board_img = self.calibration_manager.get_calibration_board_image()
        
        # Сохраняем изображение
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить изображение доски ChArUco",
            "charuco_board.png",
            "Изображения (*.png *.jpg)"
        )
        
        if file_path:
            cv2.imwrite(file_path, board_img)
            QMessageBox.information(
                self,
                "Успех",
                f"Изображение доски ChArUco сохранено в {file_path}"
            )
            self.statusBar().showMessage(f"Изображение доски ChArUco сохранено в {file_path}")
    
    def on_start_collection(self):
        """
        Обрабатывает нажатие кнопки запуска сбора данных.
        """
        # Проверяем, не запущен ли уже какой-то режим
        if self.is_running:
            self.stop_current_mode()
        
        # Проверяем, выполнена ли калибровка
        if not self.calibration_manager.is_calibrated():
            reply = QMessageBox.question(
                self,
                "Калибровка не выполнена",
                "Калибровка камеры не выполнена. Хотите выполнить калибровку сейчас?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.tabs.setCurrentIndex(0)  # Переключаемся на вкладку калибровки
                self.on_start_calibration()
                return
        
        # Определяем источник видео
        if self.radio_collection_camera.isChecked():
            self.video_source = self.spin_camera_index.value()
        else:
            video_path = self.txt_collection_video_path.text()
            if not video_path:
                QMessageBox.warning(
                    self,
                    "Ошибка",
                    "Не указан путь к видеофайлу для сбора данных"
                )
                return
            self.video_source = video_path
        
        # Открываем видеопоток
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось открыть видеопоток: {self.video_source}"
            )
            return
        
        # Устанавливаем флаги
        self.is_running = True
        self.current_mode = MODE_DATA_COLLECTION
        
        # Обновляем интерфейс
        self.btn_start_collection.setEnabled(False)
        self.btn_stop_collection.setEnabled(True)
        
        # Запускаем таймер для обновления кадров
        self.timer.start(30)  # ~30 FPS
        
        # Обновляем статус
        self.lbl_collection_status.setText("Статус сбора данных: Запущен")
        self.statusBar().showMessage("Сбор данных запущен")
    
    def on_stop_collection(self):
        """
        Обрабатывает нажатие кнопки остановки сбора данных.
        """
        # Останавливаем режим
        self.stop_current_mode()
        
        # Обновляем интерфейс
        self.btn_start_collection.setEnabled(True)
        self.btn_stop_collection.setEnabled(False)
        
        # Обновляем статус
        self.lbl_collection_status.setText("Статус сбора данных: Остановлен")
        self.statusBar().showMessage("Сбор данных остановлен")
    
    def on_refresh_files(self):
        """
        Обрабатывает нажатие кнопки обновления списка файлов.
        """
        # Получаем список файлов
        csv_files, _, _ = load_and_process_data()
        
        # Очищаем выпадающий список
        self.cmb_analysis_file.clear()
        
        # Добавляем файлы в список
        if csv_files:
            self.cmb_analysis_file.addItems(csv_files)
            self.btn_analyze_data.setEnabled(True)
            self.btn_generate_ranges.setEnabled(True)
        else:
            self.cmb_analysis_file.addItem("Нет доступных файлов")
            self.btn_analyze_data.setEnabled(False)
            self.btn_generate_ranges.setEnabled(False)
    
    def on_analyze_data(self):
        """
        Обрабатывает нажатие кнопки анализа данных.
        """
        # Получаем выбранный файл
        filename = self.cmb_analysis_file.currentText()
        
        if filename == "Нет доступных файлов":
            return
        
        # Анализируем данные
        results = analyze_data(filename)
        
        if results:
            # Формируем текст с результатами
            text = f"Результаты анализа файла: {results['filename']}\n"
            text += f"Количество записей: {results['record_count']}\n\n"
            
            text += "Статистика признаков:\n"
            for feature, stats in results['features'].items():
                text += f"\n{feature}:\n"
                text += f"  Минимум: {stats['min']:.2f}\n"
                text += f"  Максимум: {stats['max']:.2f}\n"
                text += f"  Среднее: {stats['mean']:.2f}\n"
                text += f"  Медиана: {stats['median']:.2f}\n"
                text += f"  Стандартное отклонение: {stats['std']:.2f}\n"
            
            # Отображаем результаты
            self.lbl_analysis_results.setText(text)
            self.statusBar().showMessage(f"Анализ файла {filename} выполнен")
        else:
            self.lbl_analysis_results.setText(f"Не удалось проанализировать файл {filename}")
            self.statusBar().showMessage(f"Ошибка при анализе файла {filename}")
    
    def on_generate_ranges(self):
        """
        Обрабатывает нажатие кнопки генерации диапазонов.
        """
        # Получаем выбранный файл
        filename = self.cmb_analysis_file.currentText()
        
        if filename == "Нет доступных файлов":
            return
        
        # Получаем диапазоны значений признаков
        ranges = get_feature_ranges(filename)
        
        if ranges:
            # Формируем текст с диапазонами
            text = f"Диапазоны значений признаков для файла: {filename}\n\n"
            
            for feature, range_values in ranges.items():
                text += f"{feature}: [{range_values[0]:.2f}, {range_values[1]:.2f}]\n"
            
            # Отображаем диапазоны
            self.lbl_analysis_results.setText(text)
            
            # Спрашиваем, нужно ли сохранить диапазоны в конфигурацию
            reply = QMessageBox.question(
                self,
                "Сохранение диапазонов",
                "Хотите сохранить диапазоны в конфигурацию для распознавания?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # Получаем категорию из имени файла
                category = os.path.splitext(filename)[0]
                
                # Загружаем текущую конфигурацию
                config = {}
                if os.path.exists(CONFIG_PATH):
                    try:
                        with open(CONFIG_PATH, 'r') as f:
                            config = json.load(f)
                    except Exception as e:
                        logger.error(f"Ошибка при загрузке конфигурации: {str(e)}")
                
                # Добавляем или обновляем категорию
                if "identification" not in config:
                    config["identification"] = {}
                
                config["identification"][category] = {}
                
                # Добавляем диапазоны
                for feature, range_values in ranges.items():
                    config["identification"][category][feature] = range_values
                
                # Сохраняем конфигурацию
                try:
                    with open(CONFIG_PATH, 'w') as f:
                        json.dump(config, f, indent=4)
                    
                    QMessageBox.information(
                        self,
                        "Успех",
                        f"Диапазоны для категории {category} сохранены в конфигурацию"
                    )
                    self.statusBar().showMessage(f"Диапазоны для категории {category} сохранены")
                except Exception as e:
                    logger.error(f"Ошибка при сохранении конфигурации: {str(e)}")
                    QMessageBox.critical(
                        self,
                        "Ошибка",
                        f"Не удалось сохранить конфигурацию: {str(e)}"
                    )
            
            self.statusBar().showMessage(f"Диапазоны для файла {filename} сгенерированы")
        else:
            self.lbl_analysis_results.setText(f"Не удалось сгенерировать диапазоны для файла {filename}")
            self.statusBar().showMessage(f"Ошибка при генерации диапазонов для файла {filename}")
    
    def on_start_working(self):
        """
        Обрабатывает нажатие кнопки запуска рабочего режима.
        """
        # Проверяем, не запущен ли уже какой-то режим
        if self.is_running:
            self.stop_current_mode()
        
        # Проверяем, выполнена ли калибровка
        if not self.calibration_manager.is_calibrated():
            reply = QMessageBox.question(
                self,
                "Калибровка не выполнена",
                "Калибровка камеры не выполнена. Хотите выполнить калибровку сейчас?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.tabs.setCurrentIndex(0)  # Переключаемся на вкладку калибровки
                self.on_start_calibration()
                return
        
        # Определяем источник видео
        if self.radio_working_camera.isChecked():
            self.video_source = self.spin_camera_index.value()
        else:
            video_path = self.txt_working_video_path.text()
            if not video_path:
                QMessageBox.warning(
                    self,
                    "Ошибка",
                    "Не указан путь к видеофайлу для обработки"
                )
                return
            self.video_source = video_path
        
        # Открываем видеопоток
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось открыть видеопоток: {self.video_source}"
            )
            return
        
        # Устанавливаем флаги
        self.is_running = True
        self.current_mode = MODE_WORKING
        
        # Обновляем интерфейс
        self.btn_start_working.setEnabled(False)
        self.btn_stop_working.setEnabled(True)
        
        # Запускаем таймер для обновления кадров
        self.timer.start(30)  # ~30 FPS
        
        # Обновляем статус
        self.lbl_working_status.setText("Статус рабочего режима: Запущен")
        self.statusBar().showMessage("Рабочий режим запущен")
    
    def on_stop_working(self):
        """
        Обрабатывает нажатие кнопки остановки рабочего режима.
        """
        # Останавливаем режим
        self.stop_current_mode()
        
        # Обновляем интерфейс
        self.btn_start_working.setEnabled(True)
        self.btn_stop_working.setEnabled(False)
        
        # Обновляем статус
        self.lbl_working_status.setText("Статус рабочего режима: Остановлен")
        self.statusBar().showMessage("Рабочий режим остановлен")
    
    def on_canny_low_changed(self, value):
        """
        Обрабатывает изменение нижнего порога Canny.
        
        Args:
            value: Новое значение порога
        """
        self.lbl_canny_low.setText(str(value))
        self.object_detector.update_parameters(canny_low=value)
    
    def on_canny_high_changed(self, value):
        """
        Обрабатывает изменение верхнего порога Canny.
        
        Args:
            value: Новое значение порога
        """
        self.lbl_canny_high.setText(str(value))
        self.object_detector.update_parameters(canny_high=value)
    
    def on_blur_kernel_changed(self, value):
        """
        Обрабатывает изменение размера ядра размытия.
        
        Args:
            value: Новое значение размера ядра
        """
        # Приводим к нечётному значению
        if value % 2 == 0:
            value += 1
            self.slider_blur_kernel.setValue(value)
        
        self.lbl_blur_kernel.setText(str(value))
        self.object_detector.update_parameters(blur_kernel=value)
    
    def on_min_area_changed(self, value):
        """
        Обрабатывает изменение минимальной площади.
        
        Args:
            value: Новое значение минимальной площади
        """
        self.object_detector.update_parameters(min_area=value)
    
    def on_max_area_changed(self, value):
        """
        Обрабатывает изменение максимальной площади.
        
        Args:
            value: Новое значение максимальной площади
        """
        self.object_detector.update_parameters(max_area=value)
    
    def on_save_settings(self):
        """
        Обрабатывает нажатие кнопки сохранения настроек.
        """
        # Сохраняем настройки
        if self.object_detector.save_config():
            QMessageBox.information(
                self,
                "Успех",
                "Настройки успешно сохранены"
            )
            self.statusBar().showMessage("Настройки сохранены")
        else:
            QMessageBox.critical(
                self,
                "Ошибка",
                "Не удалось сохранить настройки"
            )
            self.statusBar().showMessage("Ошибка при сохранении настроек")
    
    def update_frame(self):
        """
        Обновляет кадр видео в зависимости от текущего режима.
        """
        if not self.is_running or self.cap is None:
            return
        
        # Читаем кадр
        ret, frame = self.cap.read()
        
        # Если кадр не прочитан, пробуем сбросить позицию видеозахвата
        if not ret:
            # Для видеофайла можно сбросить позицию для циклического воспроизведения
            if isinstance(self.video_source, str):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    self.stop_current_mode()
                    return
            else:
                self.stop_current_mode()
                return
        
        # Получаем данные калибровки
        camera_matrix, dist_coeffs = self.calibration_manager.get_calibration_data()
        
        # Обрабатываем кадр в зависимости от режима
        if self.current_mode == MODE_CALIBRATION:
            # Добавляем кадр для калибровки
            success, vis_frame = self.calibration_manager.add_frame(frame)
            
            # Если кадр успешно добавлен, обновляем счетчик
            if success:
                # Если собрано достаточно кадров, вычисляем калибровку
                if len(self.calibration_manager.all_corners) >= MAX_CAPTURES:
                    if self.calibration_manager.compute_calibration():
                        self.update_calibration_status()
                        self.stop_current_mode()
                        QMessageBox.information(
                            self,
                            "Успех",
                            "Калибровка успешно выполнена"
                        )
                        return
            
            # Отображаем кадр
            self.show_frame(vis_frame, self.lbl_calibration_video)
            
            # Обновляем статус
            self.lbl_calibration_status.setText(
                f"Статус калибровки: В процессе ({len(self.calibration_manager.all_corners)}/{MAX_CAPTURES})"
            )
        
        elif self.current_mode == MODE_DATA_COLLECTION:
            # Применяем коррекцию дисторсии, если параметры предоставлены
            if camera_matrix is not None and dist_coeffs is not None:
                frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
            
            # Обрабатываем кадр
            vis_frame, objects_info = self.object_detector.process_frame(frame, update_config=True)
            
            # Обрабатываем объекты без категории
            objects_info = process_unlabeled_objects(frame, objects_info, self.object_detector)
            
            # Отображаем кадр
            self.show_frame(vis_frame, self.lbl_collection_video)
            
            # Обновляем статус
            self.lbl_collection_status.setText(
                f"Статус сбора данных: Запущен (Обнаружено объектов: {len(objects_info)})"
            )
        
        elif self.current_mode == MODE_WORKING:
            # Применяем коррекцию дисторсии, если параметры предоставлены
            if camera_matrix is not None and dist_coeffs is not None:
                frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
            
            # Обрабатываем кадр
            vis_frame, objects_info = self.object_detector.process_frame(frame, update_config=True)
            
            # Отправляем данные на робота, если требуется
            if self.chk_send_to_robot.isChecked() and objects_info:
                for obj in objects_info:
                    # Преобразуем координаты в миллиметры (примерное преобразование)
                    # В реальной системе нужно использовать калибровку для точного преобразования
                    x_mm = obj['center'][0] * 0.264583  # Примерное преобразование пикселей в мм
                    y_mm = obj['center'][1] * 0.264583
                    width_mm = obj['width'] * 0.264583
                    height_mm = obj['height'] * 0.264583
                    
                    # Отправляем данные
                    self.robot_communicator.send_data(
                        obj_id=obj['id'],
                        group_name=obj['category'] or "unknown",
                        x_mm=x_mm,
                        y_mm=y_mm,
                        width_mm=width_mm,
                        height_mm=height_mm,
                        angle=obj['angle']
                    )
            
            # Отображаем кадр
            self.show_frame(vis_frame, self.lbl_working_video)
            
            # Обновляем статус
            self.lbl_working_status.setText(
                f"Статус рабочего режима: Запущен (Обнаружено объектов: {len(objects_info)})"
            )
    
    def show_frame(self, frame, label):
        """
        Отображает кадр в метке.
        
        Args:
            frame: Кадр для отображения
            label: Метка для отображения
        """
        if frame is None:
            return
        
        # Преобразуем кадр из BGR в RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Получаем размеры метки
        label_width = label.width()
        label_height = label.height()
        
        # Масштабируем кадр до размеров метки
        h, w = rgb_frame.shape[:2]
        
        # Вычисляем коэффициент масштабирования
        scale_w = label_width / w
        scale_h = label_height / h
        scale = min(scale_w, scale_h)
        
        # Масштабируем кадр
        new_width = int(w * scale)
        new_height = int(h * scale)
        rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
        
        # Создаем QImage из кадра
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Создаем QPixmap из QImage
        pixmap = QPixmap.fromImage(q_img)
        
        # Отображаем QPixmap в метке
        label.setPixmap(pixmap)
    
    def stop_current_mode(self):
        """
        Останавливает текущий режим.
        """
        # Останавливаем таймер
        self.timer.stop()
        
        # Освобождаем ресурсы видеозахвата
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Сбрасываем флаги
        self.is_running = False
        
        # Обновляем интерфейс в зависимости от режима
        if self.current_mode == MODE_CALIBRATION:
            self.btn_start_calibration.setEnabled(True)
            self.btn_reset_calibration.setEnabled(True)
        elif self.current_mode == MODE_DATA_COLLECTION:
            self.btn_start_collection.setEnabled(True)
            self.btn_stop_collection.setEnabled(False)
        elif self.current_mode == MODE_WORKING:
            self.btn_start_working.setEnabled(True)
            self.btn_stop_working.setEnabled(False)
        
        # Сбрасываем текущий режим
        self.current_mode = None
    
    def closeEvent(self, event):
        """
        Обрабатывает событие закрытия окна.
        
        Args:
            event: Событие закрытия
        """
        # Останавливаем текущий режим
        if self.is_running:
            self.stop_current_mode()
        
        # Принимаем событие закрытия
        event.accept()

def show_frame_on_label(frame, label):
    """
    Преобразует и отображает кадр в метке.
    
    Args:
        frame: Кадр для отображения
        label: Метка для отображения
    """
    if frame is None:
        return
    
    # Преобразуем кадр из BGR в RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Получаем размеры метки
    label_width = label.width()
    label_height = label.height()
    
    # Масштабируем кадр до размеров метки
    h, w = rgb_frame.shape[:2]
    
    # Вычисляем коэффициент масштабирования
    scale_w = label_width / w
    scale_h = label_height / h
    scale = min(scale_w, scale_h)
    
    # Масштабируем кадр
    new_width = int(w * scale)
    new_height = int(h * scale)
    rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
    
    # Создаем QImage из кадра
    h, w, ch = rgb_frame.shape
    bytes_per_line = ch * w
    q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
    
    # Создаем QPixmap из QImage
    pixmap = QPixmap.fromImage(q_img)
    
    # Отображаем QPixmap в метке
    label.setPixmap(pixmap)

def stack_three_images(img1, img2, img3, target_width=None):
    """
    Объединяет три изображения горизонтально.
    
    Args:
        img1: Первое изображение
        img2: Второе изображение
        img3: Третье изображение
        target_width: Целевая ширина для каждого изображения
        
    Returns:
        numpy.ndarray: Объединенное изображение
    """
    # Проверяем, что все изображения предоставлены
    if img1 is None or img2 is None or img3 is None:
        return None
    
    # Масштабируем изображения, если указана целевая ширина
    if target_width is not None:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]
        
        # Вычисляем коэффициенты масштабирования
        scale1 = target_width / w1
        scale2 = target_width / w2
        scale3 = target_width / w3
        
        # Масштабируем изображения
        img1 = cv2.resize(img1, (target_width, int(h1 * scale1)))
        img2 = cv2.resize(img2, (target_width, int(h2 * scale2)))
        img3 = cv2.resize(img3, (target_width, int(h3 * scale3)))
    
    # Приводим изображения к одинаковой высоте
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h3, w3 = img3.shape[:2]
    
    max_height = max(h1, h2, h3)
    
    # Создаем пустые изображения с одинаковой высотой
    img1_resized = np.zeros((max_height, w1, 3), dtype=np.uint8)
    img2_resized = np.zeros((max_height, w2, 3), dtype=np.uint8)
    img3_resized = np.zeros((max_height, w3, 3), dtype=np.uint8)
    
    # Копируем исходные изображения
    img1_resized[:h1, :, :] = img1
    img2_resized[:h2, :, :] = img2
    img3_resized[:h3, :, :] = img3
    
    # Объединяем изображения горизонтально
    return np.hstack((img1_resized, img2_resized, img3_resized))

def display_results(original, processed, result, label):
    """
    Отображает результаты обработки в метке.
    
    Args:
        original: Исходное изображение
        processed: Обработанное изображение
        result: Результат обработки
        label: Метка для отображения
    """
    # Объединяем изображения
    combined = stack_three_images(original, processed, result, target_width=TARGET_WIDTH // 3)
    
    # Отображаем объединенное изображение
    show_frame_on_label(combined, label)

def choose_mode(mode):
    """
    Выбирает режим работы.
    
    Args:
        mode: Режим работы
        
    Returns:
        str: Выбранный режим
    """
    return mode

def create_interactive_interface():
    """
    Создает интерактивный интерфейс.
    
    Returns:
        QMainWindow: Главное окно приложения
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return window, app

def main():
    """
    Основная функция для запуска приложения.
    """
    # Создаем интерактивный интерфейс
    window, app = create_interactive_interface()
    
    # Запускаем цикл обработки событий
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
