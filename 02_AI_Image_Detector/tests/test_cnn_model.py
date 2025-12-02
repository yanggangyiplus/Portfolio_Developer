"""
CNN 모델 테스트
"""

import pytest
import sys
from pathlib import Path
import torch

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestCNNModel:
    """CNN 모델 테스트 클래스"""
    
    def test_model_initialization(self):
        """모델 초기화 테스트"""
        try:
            from src.models.cnn import ResNet18Model
            
            model = ResNet18Model(num_classes=2)
            assert model is not None
        except ImportError:
            pytest.skip("CNN 모델이 구현되지 않았습니다.")
    
    def test_model_forward_pass(self):
        """모델 forward pass 테스트"""
        try:
            from src.models.cnn import ResNet18Model
            
            model = ResNet18Model(num_classes=2)
            model.eval()
            
            # 더미 입력 생성 (배치 크기 1, 채널 3, 높이 224, 너비 224)
            dummy_input = torch.randn(1, 3, 224, 224)
            
            with torch.no_grad():
                output = model(dummy_input)
                assert output.shape[0] == 1  # 배치 크기
                assert output.shape[1] == 2  # 클래스 수
        except ImportError:
            pytest.skip("CNN 모델이 구현되지 않았습니다.")


if __name__ == "__main__":
    pytest.main([__file__])

