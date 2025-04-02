#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis.py - Модуль анализа данных

Этот модуль обеспечивает загрузку, обработку и предварительный анализ собранных данных
(например, результаты распознавания объектов). Позволяет сопоставлять столбцы,
вычислять дополнительные признаки и строить статистические распределения.
"""

import pandas as pd
import numpy as np
import os
import ast
import logging
from variables import CSV_DIRECTORY

# Настройка логирования
logging.basicConfig(
    filename='error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_hu_norm(hu_moments_str):
    """
    Вычисляет евклидову норму Hu-моментов.
    
    Args:
        hu_moments_str: Строковое представление списка Hu-моментов
        
    Returns:
        float: Евклидова норма Hu-моментов
    """
    try:
        # Преобразуем строку в список
        hu_moments = ast.literal_eval(hu_moments_str)
        
        # Вычисляем евклидову норму
        return np.linalg.norm(hu_moments)
    
    except Exception as e:
        logger.error(f"Ошибка при вычислении нормы Hu-моментов: {str(e)}")
        return 0.0

def compute_avg_color_val(avg_color_str):
    """
    Вычисляет среднее значение цвета.
    
    Args:
        avg_color_str: Строковое представление цвета [B, G, R]
        
    Returns:
        float: Среднее значение цвета
    """
    try:
        # Преобразуем строку в список
        avg_color = ast.literal_eval(avg_color_str)
        
        # Вычисляем среднее значение
        return np.mean(avg_color)
    
    except Exception as e:
        logger.error(f"Ошибка при вычислении среднего значения цвета: {str(e)}")
        return 0.0

def load_and_process_data(filename=None):
    """
    Загружает и обрабатывает данные из CSV-файла.
    
    Args:
        filename: Имя CSV-файла (без пути)
        
    Returns:
        tuple: (DataFrame, список признаков, словарь соответствия имён столбцов)
    """
    try:
        # Если имя файла не указано, возвращаем список доступных файлов
        if filename is None:
            csv_files = [f for f in os.listdir(CSV_DIRECTORY) if f.endswith('.csv')]
            if not csv_files:
                logger.warning("Не найдены CSV-файлы в директории данных")
                return None, [], {}
            return csv_files, [], {}
        
        # Полный путь к файлу
        filepath = os.path.join(CSV_DIRECTORY, filename)
        
        # Проверяем существование файла
        if not os.path.exists(filepath):
            logger.error(f"Файл {filepath} не существует")
            return None, [], {}
        
        # Загружаем данные
        df = pd.read_csv(filepath)
        
        # Определяем список признаков
        features = [
            'area', 'perimeter', 'extent', 'circularity', 'solidity',
            'convexity_defects_count', 'avg_defect_depth', 'aspect_ratio'
        ]
        
        # Словарь альтернативных имён столбцов
        alt_columns = {
            'area': ['area', 'Area'],
            'perimeter': ['perimeter', 'Perimeter'],
            'extent': ['extent', 'Extent'],
            'circularity': ['circularity', 'Circularity'],
            'solidity': ['solidity', 'Solidity'],
            'convexity_defects_count': ['convexity_defects_count', 'ConvexityDefectsCount'],
            'avg_defect_depth': ['avg_defect_depth', 'AvgDefectDepth'],
            'aspect_ratio': ['aspect_ratio', 'AspectRatio'],
            'hu_moments': ['hu_moments', 'HuMoments'],
            'avg_color': ['avg_color', 'AvgColor']
        }
        
        # Сопоставляем имена столбцов
        column_mapping = {}
        for feature, alternatives in alt_columns.items():
            for alt in alternatives:
                if alt in df.columns:
                    column_mapping[feature] = alt
                    break
        
        # Преобразуем числовые столбцы
        for feature in features:
            if feature in column_mapping and column_mapping[feature] in df.columns:
                df[column_mapping[feature]] = pd.to_numeric(df[column_mapping[feature]], errors='coerce')
        
        # Вычисляем дополнительные признаки
        if 'hu_moments' in column_mapping and column_mapping['hu_moments'] in df.columns:
            df['hu_moments_norm'] = df[column_mapping['hu_moments']].apply(compute_hu_norm)
        
        if 'avg_color' in column_mapping and column_mapping['avg_color'] in df.columns:
            df['avg_color_val'] = df[column_mapping['avg_color']].apply(compute_avg_color_val)
        
        return df, features, column_mapping
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке и обработке данных: {str(e)}")
        return None, [], {}

def get_feature_statistics(df, feature, column_mapping):
    """
    Вычисляет статистику для указанного признака.
    
    Args:
        df: DataFrame с данными
        feature: Имя признака
        column_mapping: Словарь соответствия имён столбцов
        
    Returns:
        dict: Статистика признака
    """
    try:
        if feature not in column_mapping or column_mapping[feature] not in df.columns:
            return None
        
        # Получаем столбец
        column = df[column_mapping[feature]]
        
        # Вычисляем статистику
        stats = {
            'min': column.min(),
            'max': column.max(),
            'mean': column.mean(),
            'median': column.median(),
            'std': column.std()
        }
        
        return stats
    
    except Exception as e:
        logger.error(f"Ошибка при вычислении статистики признака: {str(e)}")
        return None

def analyze_data(filename):
    """
    Анализирует данные из CSV-файла.
    
    Args:
        filename: Имя CSV-файла (без пути)
        
    Returns:
        dict: Результаты анализа
    """
    try:
        # Загружаем и обрабатываем данные
        df, features, column_mapping = load_and_process_data(filename)
        
        if df is None:
            return None
        
        # Результаты анализа
        results = {
            'filename': filename,
            'record_count': len(df),
            'features': {}
        }
        
        # Вычисляем статистику для каждого признака
        for feature in features:
            stats = get_feature_statistics(df, feature, column_mapping)
            if stats:
                results['features'][feature] = stats
        
        # Добавляем статистику для дополнительных признаков
        if 'hu_moments_norm' in df.columns:
            results['features']['hu_moments_norm'] = {
                'min': df['hu_moments_norm'].min(),
                'max': df['hu_moments_norm'].max(),
                'mean': df['hu_moments_norm'].mean(),
                'median': df['hu_moments_norm'].median(),
                'std': df['hu_moments_norm'].std()
            }
        
        if 'avg_color_val' in df.columns:
            results['features']['avg_color_val'] = {
                'min': df['avg_color_val'].min(),
                'max': df['avg_color_val'].max(),
                'mean': df['avg_color_val'].mean(),
                'median': df['avg_color_val'].median(),
                'std': df['avg_color_val'].std()
            }
        
        return results
    
    except Exception as e:
        logger.error(f"Ошибка при анализе данных: {str(e)}")
        return None

def get_feature_ranges(filename, features=None, margin=0.1):
    """
    Вычисляет диапазоны значений признаков для идентификации объектов.
    
    Args:
        filename: Имя CSV-файла (без пути)
        features: Список признаков для анализа (если None, используются все)
        margin: Запас по краям диапазона (в процентах)
        
    Returns:
        dict: Диапазоны значений признаков
    """
    try:
        # Загружаем и обрабатываем данные
        df, all_features, column_mapping = load_and_process_data(filename)
        
        if df is None:
            return None
        
        # Если список признаков не указан, используем все
        if features is None:
            features = all_features
        
        # Диапазоны значений признаков
        ranges = {}
        
        # Вычисляем диапазоны для каждого признака
        for feature in features:
            if feature in column_mapping and column_mapping[feature] in df.columns:
                column = df[column_mapping[feature]]
                
                # Вычисляем минимум и максимум
                min_val = column.min()
                max_val = column.max()
                
                # Добавляем запас
                range_size = max_val - min_val
                min_val = min_val - range_size * margin
                max_val = max_val + range_size * margin
                
                ranges[feature] = [min_val, max_val]
        
        # Добавляем диапазоны для дополнительных признаков
        if 'hu_moments_norm' in df.columns:
            min_val = df['hu_moments_norm'].min()
            max_val = df['hu_moments_norm'].max()
            range_size = max_val - min_val
            ranges['hu_moments_norm'] = [
                min_val - range_size * margin,
                max_val + range_size * margin
            ]
        
        if 'avg_color_val' in df.columns:
            min_val = df['avg_color_val'].min()
            max_val = df['avg_color_val'].max()
            range_size = max_val - min_val
            ranges['avg_color_val'] = [
                min_val - range_size * margin,
                max_val + range_size * margin
            ]
        
        return ranges
    
    except Exception as e:
        logger.error(f"Ошибка при вычислении диапазонов значений признаков: {str(e)}")
        return None
