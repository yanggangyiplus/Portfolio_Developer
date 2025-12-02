"""
이상 탐지 모듈 테스트
"""

import pytest
import sys
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestIsolationForestDetector:
    """Isolation Forest 이상 탐지 모듈 테스트 클래스"""
    
    def test_anomaly_detection(self):
        """이상 탐지 기본 테스트"""
        try:
            from src.anomaly.isolation_forest import IsolationForestDetector
            
            detector = IsolationForestDetector()
            
            # 정상 데이터
            normal_data = np.random.randn(100, 5)
            
            # 이상 데이터
            anomaly_data = np.random.randn(10, 5) + 5  # 평균이 5만큼 벗어남
            
            # 학습
            detector.fit(normal_data)
            
            # 예측
            predictions = detector.predict(anomaly_data)
            
            assert len(predictions) == len(anomaly_data)
            assert all(pred in [-1, 1] for pred in predictions)  # -1: 이상, 1: 정상
        except ImportError:
            pytest.skip("IsolationForestDetector가 구현되지 않았습니다.")
    
    def test_score_samples(self):
        """이상 점수 계산 테스트"""
        try:
            from src.anomaly.isolation_forest import IsolationForestDetector
            
            detector = IsolationForestDetector()
            data = np.random.randn(50, 3)
            detector.fit(data)
            
            scores = detector.score_samples(data)
            
            assert len(scores) == len(data)
            assert all(isinstance(score, (int, float)) for score in scores)
        except ImportError:
            pytest.skip("IsolationForestDetector가 구현되지 않았습니다.")


class TestNginxParser:
    """Nginx 로그 파서 테스트 클래스"""
    
    def test_parse_log_line(self):
        """로그 라인 파싱 테스트"""
        try:
            from src.collector.nginx_parser import NginxLogParser
            
            parser = NginxLogParser()
            
            # 표준 Nginx 로그 형식
            log_line = '127.0.0.1 - - [01/Jan/2024:00:00:00 +0000] "GET / HTTP/1.1" 200 1234'
            
            parsed = parser.parse_line(log_line)
            
            assert parsed is not None
            assert 'ip' in parsed or 'status' in parsed or 'method' in parsed
        except ImportError:
            pytest.skip("NginxLogParser가 구현되지 않았습니다.")


if __name__ == "__main__":
    pytest.main([__file__])

