"""
변화점 탐지 알고리즘 모듈
시계열 데이터에서 급격한 변화 지점 탐지
"""

from typing import List, Tuple
import numpy as np


class ChangeDetector:
    """변화점 탐지 클래스"""
    
    def __init__(self, method: str = 'simple'):
        """
        Args:
            method: 탐지 방법 (simple, cusum, zscore, bayesian)
        """
        self.method = method
    
    def simple_detection(self, data: List[float], threshold: float = 0.3) -> List[int]:
        """
        간단한 변화점 탐지
        
        Args:
            data: 시계열 데이터
            threshold: 변화 임계값
            
        Returns:
            변화점 인덱스 리스트
        """
        if len(data) < 2:
            return []
        
        change_points = []
        for i in range(1, len(data)):
            change_rate = abs(data[i] - data[i-1]) / (abs(data[i-1]) + 1e-6)
            if change_rate > threshold:
                change_points.append(i)
        
        return change_points
    
    def cusum_detection(self, data: List[float], threshold: float = 5.0) -> List[int]:
        """
        CUSUM 알고리즘 기반 변화점 탐지
        
        Args:
            data: 시계열 데이터
            threshold: CUSUM 임계값
            
        Returns:
            변화점 인덱스 리스트
        """
        if len(data) < 2:
            return []
        
        mean = np.mean(data)
        cumsum = 0
        change_points = []
        
        for i, value in enumerate(data):
            cumsum += (value - mean)
            if abs(cumsum) > threshold:
                change_points.append(i)
                cumsum = 0
        
        return change_points
    
    def zscore_detection(self, data: List[float], threshold: float = 2.0) -> List[int]:
        """
        Z-score 기반 변화점 탐지
        
        Args:
            data: 시계열 데이터
            threshold: Z-score 임계값
            
        Returns:
            변화점 인덱스 리스트
        """
        if len(data) < 2:
            return []
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return []
        
        change_points = []
        for i, value in enumerate(data):
            z_score = abs((value - mean) / std)
            if z_score > threshold:
                change_points.append(i)
        
        return change_points
    
    def detect(self, data: List[float], **kwargs) -> List[int]:
        """
        변화점 탐지 (메서드에 따라 자동 선택)
        
        Args:
            data: 시계열 데이터
            **kwargs: 메서드별 파라미터
            
        Returns:
            변화점 인덱스 리스트
        """
        if self.method == 'simple':
            threshold = kwargs.get('threshold', 0.3)
            return self.simple_detection(data, threshold)
        elif self.method == 'cusum':
            threshold = kwargs.get('threshold', 5.0)
            return self.cusum_detection(data, threshold)
        elif self.method == 'zscore':
            threshold = kwargs.get('threshold', 2.0)
            return self.zscore_detection(data, threshold)
        else:
            return self.simple_detection(data)

