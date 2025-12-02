# Log AI Predictor

서버 로그 기반 이상 패턴 분석 및 장애 예측 시스템

## 문제 정의

서버 장애 발생 전 미세한 패턴 변화를 실시간으로 감지하기 어려웠습니다. 로그가 수십만 줄 이상 쌓여 패턴 변화를 파악하는 것이 어려웠고, 기존 모니터링 도구로는 충분한 분석이 어려웠습니다.

## 해결 방법

1. **로그 수집 시스템 구축**: `tail -f` 기반으로 실시간 로그를 수집하고 배치 모드도 지원하도록 개발했습니다.
2. **이상 탐지 알고리즘 구현**: AutoEncoder와 Isolation Forest를 활용하여 정상 패턴과 이상 패턴을 구분하는 알고리즘을 구현했습니다.
3. **특징 추출**: 시간 윈도우별로 RPS, 에러율, 응답시간 분포 등을 계산하여 분석에 활용했습니다.
4. **장애 예측 기능**: 과거 장애 직전 패턴과 현재 패턴을 비교하여 유사도가 높을 경우 경고를 제공하는 기능을 구현했습니다.

## 성과

- 초당 수백 건의 로그를 처리할 수 있는 시스템 구축
- 이상 발생 후 수초 내 탐지 가능
- 과거 패턴과 유사도 80% 이상 시 경고 알림 제공
- Streamlit 대시보드를 통한 실시간 모니터링 구현

## 기술 스택

`Python` `PyTorch` `scikit-learn` `Streamlit` `Flask` `Pandas` `Plotly`

## 프로젝트 구조

```
03_Log_AI_Predictor/
├── app/web/
│   └── main.py             # Streamlit 대시보드
├── src/
│   ├── collector/
│   │   └── nginx_parser.py # Nginx 로그 파서
│   └── anomaly/
│       └── isolation_forest.py # 이상 탐지 알고리즘
├── configs/
│   └── config_collect.yaml
├── docs/
│   └── MODEL_COMPARISON.md
└── requirements.txt
```

## 프로젝트 위치

`../../Log-AI-Predictor/`
