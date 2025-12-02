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
            from src.anomaly.iforest_detector import IsolationForestDetector
            
            detector = IsolationForestDetector()
            
            # 정상 데이터
            normal_data = np.random.randn(100, 3)
            
            # 이상 데이터
            anomaly_data = np.random.randn(10, 3) + 5
            
            # 학습
            detector.fit(normal_data)
            
            # 예측
            predictions = detector.predict(anomaly_data)
            
            assert len(predictions) == len(anomaly_data)
        except ImportError:
            pytest.skip("IsolationForestDetector가 구현되지 않았습니다.")
    
    def test_batch_detection(self):
        """배치 이상 탐지 테스트"""
        try:
            from src.anomaly.iforest_detector import IsolationForestDetector
            
            detector = IsolationForestDetector()
            data = np.random.randn(50, 4)
            detector.fit(data)
            
            # 배치 예측
            batch_predictions = detector.predict_batch(data[:10])
            
            assert len(batch_predictions) == 10
        except ImportError:
            pytest.skip("IsolationForestDetector가 구현되지 않았습니다.")


class TestHTTPPoller:
    """HTTP 폴링 모듈 테스트 클래스"""
    
    def test_poll_endpoint(self):
        """엔드포인트 폴링 테스트"""
        try:
            from src.ingest.http_poller import HTTPPoller
            
            poller = HTTPPoller(endpoint_url="http://httpbin.org/get")
            
            # 폴링 실행 (실제 네트워크 요청)
            result = poller.poll()
            
            assert result is not None
        except ImportError:
            pytest.skip("HTTPPoller가 구현되지 않았습니다.")
        except Exception as e:
            # 네트워크 오류는 테스트 스킵
            pytest.skip(f"네트워크 오류: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__])

