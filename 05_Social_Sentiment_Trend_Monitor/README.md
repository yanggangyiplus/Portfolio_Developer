# Social Sentiment Trend Monitor

실시간 감정 분석 및 트렌드 모니터링 서비스

## 문제 정의

온라인 텍스트 기반의 감정 흐름을 실시간으로 모니터링하고 트렌드 변화를 파악하기 어려웠습니다. 기존 도구들은 단순 감정 분석에 그치며 시계열 트렌드 분석 기능이 부족했습니다.

## 해결 방법

1. **데이터 수집 시스템 구축**: YouTube API를 연동하여 댓글 및 영상 메타데이터를 수집하는 시스템을 개발했습니다.
2. **감정 분석 기능 구현**: 한국어 키워드 기반 규칙 분석기와 KoBERT/KcBERT 모델을 활용하여 감정 분석 기능을 구현했습니다.
3. **변화점 탐지 알고리즘 구현**: 시계열 데이터에서 급격한 변화가 발생하는 지점을 탐지하는 알고리즘을 구현했습니다.
4. **실시간 대시보드 개발**: Streamlit을 활용하여 실시간 모니터링이 가능한 대시보드를 구축했습니다.

## 성과

- 규칙 기반 분석기와 KoBERT/KcBERT 모델을 통한 감정 분석 시스템 구축
- 4가지 변화점 탐지 알고리즘 지원으로 다양한 시나리오 대응
- 실시간 대시보드를 통한 여론 변화 추적 가능
- 데이터 수집부터 분석, 시각화까지 End-to-End 파이프라인 구축

## 기술 스택

`Python` `FastAPI` `Streamlit` `HuggingFace Transformers` `KoBERT` `KcBERT` `YouTube API` `SQLite` `Plotly`

## 프로젝트 구조

```
05_Social_Sentiment_Trend_Monitor/
├── app/
│   └── web_demo.py         # Streamlit 대시보드
├── src/
│   ├── collectors/
│   │   └── youtube_collector.py # YouTube API 수집기
│   ├── sentiment/
│   │   └── rule_based_analyzer.py # 규칙 기반 감정 분석
│   └── trend/
│       └── change_detection.py   # 변화점 탐지 알고리즘
├── configs/
│   └── config_collector.yaml
├── docs/
│   └── ARCHITECTURE.md
└── requirements.txt
```

## 프로젝트 위치

`../../Social-sentiment-trend-monitor/`
