"""
규칙 기반 감정 분석 모듈 테스트
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sentiment.rule_based_analyzer import RuleBasedAnalyzer


class TestRuleBasedAnalyzer:
    """규칙 기반 감정 분석 모듈 테스트 클래스"""
    
    def test_positive_sentiment(self):
        """긍정 감정 분석 테스트"""
        analyzer = RuleBasedAnalyzer()
        result = analyzer.analyze("이 영상 정말 좋아요!")
        
        assert 'positive' in result
        assert result['positive'] > 0
    
    def test_negative_sentiment(self):
        """부정 감정 분석 테스트"""
        analyzer = RuleBasedAnalyzer()
        result = analyzer.analyze("이 영상 별로예요.")
        
        assert 'negative' in result
        assert result['negative'] > 0
    
    def test_get_sentiment_label(self):
        """감정 레이블 반환 테스트"""
        analyzer = RuleBasedAnalyzer()
        label = analyzer.get_sentiment_label("최고의 영상입니다!")
        
        assert label in ['positive', 'negative', 'neutral']
    
    def test_analyze_batch(self):
        """배치 분석 테스트"""
        analyzer = RuleBasedAnalyzer()
        texts = [
            "좋은 영상이네요.",
            "별로예요.",
            "보통이에요."
        ]
        
        results = analyzer.analyze_batch(texts)
        assert len(results) == len(texts)
        assert all('positive' in r for r in results)


if __name__ == "__main__":
    pytest.main([__file__])

