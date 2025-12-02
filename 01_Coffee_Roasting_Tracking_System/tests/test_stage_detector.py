"""
로스팅 단계 감지 알고리즘 테스트
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.algorithms.stage_detector import RoastingStageDetector


class TestRoastingStageDetector:
    """로스팅 단계 감지 알고리즘 테스트 클래스"""
    
    def test_detect_drying_stage(self):
        """건조 단계 감지 테스트"""
        detector = RoastingStageDetector()
        
        # 건조 단계 조건: 50~150°C, 높은 습도
        temp = 100.0
        humidity = 80.0
        ror = 10.0
        
        stage = detector.detect_stage(temp, humidity, ror)
        assert stage in ['drying', 'early_stage']
    
    def test_detect_browning_stage(self):
        """갈변 단계 감지 테스트"""
        detector = RoastingStageDetector()
        
        # 갈변 단계 조건: 150~190°C
        temp = 170.0
        humidity = 50.0
        ror = 5.0
        
        stage = detector.detect_stage(temp, humidity, ror)
        assert stage in ['browning', 'middle_stage']
    
    def test_detect_first_crack(self):
        """1차 크랙 감지 테스트"""
        detector = RoastingStageDetector()
        
        # 1차 크랙 조건: 190~205°C, RoR 급격 감소
        temp = 200.0
        humidity = 30.0
        ror = -5.0  # 급격한 감소
        
        stage = detector.detect_stage(temp, humidity, ror)
        assert stage in ['first_crack', 'crack_stage']


if __name__ == "__main__":
    pytest.main([__file__])

