# Coffee Roasting Tracking System

센서 기반 ML 분류 모델 및 웹 대시보드

## 문제 정의

로스팅 공정이 경험에 의존적이어서 재현 가능한 프로파일을 만들기 어려웠습니다. 실시간으로 온도와 습도 변화를 기록하고 적정 배전점을 판단하는 것이 필요했습니다.

## 해결 방법

1. **센서 데이터 수집 시스템 구축**: 온도, 습도, 가열량 등 센서 데이터를 실시간으로 수집하고 저장하는 시스템을 개발했습니다.
2. **ML 모델 개발**: RandomForest와 GradientBoosting 모델을 활용하여 로스팅 단계를 분류하는 모델을 개발했습니다.
3. **로스팅 단계 자동 감지**: 규칙 기반 알고리즘과 ML 모델을 결합하여 건조/갈변/크랙 단계를 자동으로 감지했습니다.
4. **목표 배전도 도달 예측**: 현재 상태를 기반으로 목표 배전도 도달 시간을 예측하는 알고리즘을 구현했습니다.

## 성과

- 센서 데이터 기반 로스팅 단계 분류 정확도 92% 달성
- 추론 속도 1.1초에서 0.8초로 개선 (약 27% 향상)
- 실시간 모니터링 대시보드를 통한 로스팅 진행 상황 추적 가능
- SQLite 기반 프로파일 저장 및 비교 분석 기능 구현

## 기술 스택

`Python` `scikit-learn` `PyTorch` `Streamlit` `SQLite` `Pandas` `Plotly`

## 프로젝트 구조

```
01_Coffee_Roasting_Tracking_System/
├── app/
│   └── main.py              # Streamlit 대시보드
├── src/
│   ├── algorithms/
│   │   └── stage_detector.py    # 로스팅 단계 감지 알고리즘
│   ├── models/
│   │   └── sensor_classifier.py # ML 모델 (RandomForest/GradientBoosting)
│   ├── prediction/
│   │   └── roast_predictor.py  # 배전도 도달 예측
│   └── utils/
│       └── constants.py         # 상수 정의
├── configs/
│   └── config.yaml
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
streamlit run app/main.py
```

### 테스트 실행
```bash
# 모든 테스트 실행
pytest tests/

# 특정 테스트 실행
pytest tests/test_stage_detector.py -v
```

## 배포 방법

### Streamlit Cloud 배포 (권장)
1. GitHub에 프로젝트 푸시
2. [Streamlit Cloud](https://streamlit.io/cloud)에서 배포
3. Main file path: `app/main.py`

자세한 배포 방법은 `../../DEPLOYMENT_GUIDE.md`를 참조하세요.

## 프로젝트 위치

`../../Coffee-roasting-tracking-system/`
