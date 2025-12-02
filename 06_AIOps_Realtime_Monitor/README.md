# AIOps Realtime Monitor

실시간 스트리밍 데이터 기반 이상 탐지 및 모니터링 시스템

## 문제 정의

서비스 장애 발생 전 징후를 실시간으로 감지하기 어려웠습니다. 기존 모니터링 도구는 단순 threshold 기반으로 동작하여 미세한 변화를 감지하기 어려웠고, 로그 기반 분석은 반응 속도가 느려 장애 발생 후에야 알 수 있는 경우가 많았습니다.

## 해결 방법

1. **다양한 소스 지원**: HTTP API, WebSocket, Socket 등 다양한 소스로부터 데이터를 수집할 수 있는 시스템을 구축했습니다.
2. **이상 탐지 알고리즘 구현**: Isolation Forest와 Z-score를 활용하여 단순 threshold가 아닌 데이터 분석 기반 이상 탐지 알고리즘을 구현했습니다.
3. **메모리 효율적 처리**: Rolling Window를 활용하여 오래된 데이터를 자동으로 삭제하여 메모리 사용량을 관리했습니다.
4. **실시간 대시보드 개발**: 1초 간격으로 업데이트되는 실시간 대시보드를 구축하여 즉시 이상 징후를 파악할 수 있도록 했습니다.

## 성과

- 초당 수백 건의 데이터를 실시간으로 처리할 수 있는 시스템 구축
- HTTP 오류, 성능 이상, 리소스 이상, 보안 패턴 등 다양한 이상 탐지 가능
- 이상 발생 후 수초 내 탐지 가능
- 새로운 탐지 방법이나 수집 방식을 쉽게 추가할 수 있는 확장 가능한 구조

## 기술 스택

`Python` `Streamlit` `scikit-learn` `NumPy` `Pandas` `Plotly` `psutil`

## 프로젝트 구조

```
06_AIOps_Realtime_Monitor/
├── app/web/
│   └── dashboard.py        # Streamlit 대시보드
├── src/
│   ├── ingest/
│   │   └── http_poller.py  # HTTP API 수집기
│   └── anomaly/
│       └── iforest_detector.py # 이상 탐지 알고리즘
├── configs/
│   └── config_stream.yaml
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
streamlit run app/web/dashboard.py
```

### 데이터 소스 설정
```bash
# HTTP API 폴링 설정
# configs/config_stream.yaml 파일에서 데이터 소스 URL 설정
```

## 프로젝트 위치

`../../AIOps-Realtime-Monitor/`
