# News Trend Spike Monitor

뉴스 트렌드 분석 시스템 개발

## 문제 정의

뉴스 기반 트렌드 변화를 실시간으로 감지하고 분석하기 어려웠습니다. 감정 분석과 시계열 트렌드 분석을 결합한 시스템이 필요했으며, Google News API의 일일 제한으로 인해 대량 데이터 수집이 어려웠습니다.

## 해결 방법

1. **데이터 수집 최적화**: RSS 피드와 Google News API를 병행하여 API 제한을 회피하고 데이터 수집 안정성을 확보했습니다.
2. **텍스트 분석 기능 구현**: KcELECTRA 모델을 활용하여 한국어 텍스트를 분석하고 감정을 파악하는 기능을 구현했습니다.
3. **트렌드 변화 감지 알고리즘 구현**: 시계열 데이터에서 급격한 변화가 발생하는 지점을 탐지하는 알고리즘을 구현했습니다.
4. **성능 최적화**: 캐싱 전략을 적용하여 API 호출을 최소화하고 응답 속도를 향상시켰습니다.

## 성과

- KcELECTRA 모델을 통한 텍스트 분석 평균 처리 시간 54ms 달성
- 트렌드 변화 감지 F1-Score 0.80 달성
- Parquet 형식으로 데이터 저장하여 압축률 93.3% 달성
- 실시간 대시보드를 통한 트렌드 변화 추적 가능

## 기술 스택

`Python` `FastAPI` `Streamlit` `HuggingFace Transformers` `KcELECTRA` `RSS` `Google News API` `Parquet`

## 프로젝트 구조

```
07_News_Trend_Spike_Monitor/
├── app/web/
│   └── main.py             # Streamlit 대시보드
├── src/
│   ├── nlp/
│   │   └── sentiment_analyzer.py # KcELECTRA 기반 텍스트 분석
│   └── anomaly/
│       └── spike_detector.py      # 트렌드 변화 감지 알고리즘
├── configs/
│   └── config_nlp.yaml
├── docs/
│   └── ARCHITECTURE.md
└── requirements.txt
```

## 실행 방법

### 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 대시보드 실행
```bash
# 프로젝트 루트에서 실행
streamlit run app/web/main.py
```

### 감정 분석 테스트
```bash
# KcELECTRA 모델 테스트
python -c "from src.nlp.sentiment_analyzer import SentimentAnalyzer; analyzer = SentimentAnalyzer(); print(analyzer.analyze('좋은 뉴스입니다.'))"
```

### 테스트 실행
```bash
# 모든 테스트 실행
pytest tests/

# 감정 분석 테스트
pytest tests/test_sentiment_analyzer.py -v
```

## 프로젝트 위치

`../../News-trend-spike-monitor/`
