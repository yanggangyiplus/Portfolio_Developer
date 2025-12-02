"""
감정 분석 모듈 테스트
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSentimentAnalyzer:
    """감정 분석 모듈 테스트 클래스"""
    
    def test_positive_sentiment(self):
        """긍정 감정 분석 테스트"""
        try:
            from src.nlp.sentiment_analyzer import SentimentAnalyzer
            
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("이 뉴스는 정말 좋은 소식이네요!")
            
            assert result is not None
            assert 'sentiment' in result or 'label' in result
        except ImportError:
            pytest.skip("SentimentAnalyzer가 구현되지 않았습니다.")
    
    def test_negative_sentiment(self):
        """부정 감정 분석 테스트"""
        try:
            from src.nlp.sentiment_analyzer import SentimentAnalyzer
            
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze("이 뉴스는 정말 나쁜 소식이네요.")
            
            assert result is not None
        except ImportError:
            pytest.skip("SentimentAnalyzer가 구현되지 않았습니다.")
    
    def test_batch_analysis(self):
        """배치 분석 테스트"""
        try:
            from src.nlp.sentiment_analyzer import SentimentAnalyzer
            
            analyzer = SentimentAnalyzer()
            texts = [
                "좋은 뉴스입니다.",
                "나쁜 뉴스입니다.",
                "보통 뉴스입니다."
            ]
            
            results = analyzer.analyze_batch(texts)
            assert len(results) == len(texts)
        except ImportError:
            pytest.skip("SentimentAnalyzer가 구현되지 않았습니다.")


if __name__ == "__main__":
    pytest.main([__file__])

