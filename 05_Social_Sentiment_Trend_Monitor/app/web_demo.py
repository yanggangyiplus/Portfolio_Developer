"""
Streamlit 기반 실시간 감정 트렌드 모니터링 대시보드
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.collectors.youtube_collector import YouTubeCollector
from src.sentiment.rule_based_analyzer import RuleBasedAnalyzer
from src.trend.change_detection import ChangeDetector


def main():
    st.set_page_config(page_title="Social Sentiment Trend Monitor", layout="wide")
    
    st.title("Social Sentiment Trend Monitor")
    st.markdown("실시간 감정 분석 및 트렌드 모니터링 서비스")
    
    # 사이드바 설정
    st.sidebar.header("설정")
    search_query = st.sidebar.text_input("검색어", value="")
    max_videos = st.sidebar.slider("최대 비디오 수", 1, 20, 10)
    detection_method = st.sidebar.selectbox(
        "변화점 탐지 방법",
        ["simple", "cusum", "zscore"]
    )
    
    if st.sidebar.button("데이터 수집 시작"):
        if search_query:
            with st.spinner("데이터 수집 중..."):
                collector = YouTubeCollector()
                data = collector.collect_data(search_query, max_videos)
                
                if data:
                    st.session_state['collected_data'] = data
                    st.success(f"{len(data)}개의 데이터를 수집했습니다.")
                else:
                    st.error("데이터 수집에 실패했습니다.")
    
    # 데이터 분석
    if 'collected_data' in st.session_state:
        data = st.session_state['collected_data']
        
        # 감정 분석
        analyzer = RuleBasedAnalyzer()
        sentiments = []
        for item in data:
            sentiment = analyzer.get_sentiment_label(item['comment_text'])
            sentiments.append({
                'text': item['comment_text'],
                'sentiment': sentiment,
                'video_title': item['video_title'],
                'published_at': item['published_at']
            })
        
        df = pd.DataFrame(sentiments)
        
        # 감정 분포 시각화
        st.subheader("감정 분포")
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="감정 분포"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 시계열 트렌드 분석
        st.subheader("시계열 트렌드")
        df['date'] = pd.to_datetime(df['published_at'])
        df['sentiment_score'] = df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
        
        daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean()
        
        fig2 = px.line(
            x=daily_sentiment.index,
            y=daily_sentiment.values,
            title="일별 평균 감정 점수",
            labels={'x': '날짜', 'y': '감정 점수'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # 변화점 탐지
        detector = ChangeDetector(method=detection_method)
        change_points = detector.detect(list(daily_sentiment.values))
        
        if change_points:
            st.subheader("변화점 탐지 결과")
            st.write(f"총 {len(change_points)}개의 변화점이 탐지되었습니다.")
            for cp in change_points[:5]:  # 최대 5개만 표시
                st.write(f"- 인덱스 {cp}: {list(daily_sentiment.index)[cp]}")


if __name__ == "__main__":
    main()

