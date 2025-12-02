"""
규칙 기반 감정 분석 모듈
한국어 키워드 기반 감정 분석
"""

from typing import Dict, List


class RuleBasedAnalyzer:
    """규칙 기반 감정 분석 클래스"""
    
    def __init__(self):
        """감정 키워드 초기화"""
        self.positive_keywords = [
            '좋아', '좋다', '최고', '훌륭', '멋져', '사랑', '행복', '기쁘', '감사',
            '만족', '추천', '완벽', '대박', '최고다', '좋은', '훌륭한'
        ]
        
        self.negative_keywords = [
            '싫어', '싫다', '나쁘', '최악', '별로', '불만', '화나', '슬프', '우울',
            '실망', '비추', '안좋', '나쁜', '최악이다', '별로다'
        ]
        
        self.neutral_keywords = [
            '보통', '그냥', '평범', '괜찮', '그저', '일반'
        ]
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        텍스트 감정 분석
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            감정 점수 딕셔너리 (positive, negative, neutral)
        """
        if not text:
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
        
        text_lower = text.lower()
        
        positive_score = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_score = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        neutral_score = sum(1 for keyword in self.neutral_keywords if keyword in text_lower)
        
        total_score = positive_score + negative_score + neutral_score
        
        if total_score == 0:
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
        
        return {
            'positive': positive_score / total_score,
            'negative': negative_score / total_score,
            'neutral': neutral_score / total_score
        }
    
    def get_sentiment_label(self, text: str) -> str:
        """
        감정 레이블 반환
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            감정 레이블 (positive, negative, neutral)
        """
        scores = self.analyze(text)
        max_label = max(scores.items(), key=lambda x: x[1])[0]
        return max_label
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        여러 텍스트 일괄 분석
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            감정 점수 리스트
        """
        return [self.analyze(text) for text in texts]
