# 포트폴리오 - AI 엔지니어 / 데이터 엔지니어 / 백엔드 개발자

개인 프로젝트 포트폴리오 모음

## 프로젝트 목록

### 01. Coffee Roasting Tracking System
센서 기반 ML 분류 모델 및 웹 대시보드
- 정확도 92% 달성, 추론 속도 27% 개선
- RandomForest/GradientBoosting + ResNet18 CNN
- [상세 보기](./01_Coffee_Roasting_Tracking_System/)

### 02. AI Image Detector
딥러닝 기반 이미지 분류 시스템
- Test Accuracy 97.06% 달성 (ViT-Base)
- CNN vs ViT 비교 실험, Ablation Study 수행
- HuggingFace Spaces 배포 완료
- [상세 보기](./02_AI_Image_Detector/)

### 03. Log AI Predictor
서버 로그 기반 이상 패턴 분석 및 장애 예측 시스템
- 초당 수백 건 로그 처리, 수초 내 이상 탐지
- AutoEncoder + Isolation Forest 하이브리드
- KNN 기반 장애 예측 (유사도 80% 이상 시 경고)
- [상세 보기](./03_Log_AI_Predictor/)

### 04. Serverless RAG Assistant
AWS Serverless 기반 RAG(Retrieval-Augmented Generation) 서비스
- LangChain 기반 RAG 파이프라인 구축
- 검색 정확도 85% 달성
- Lambda 콜드스타트 2초 이내
- AWS CDK 기반 IaC, 자동 배포 파이프라인
- [상세 보기](./04_Serverless_RAG_Assistant/)

### 05. Social Sentiment Trend Monitor
실시간 감정 분석 및 트렌드 변화 탐지 서비스
- YouTube API 연동, 실시간 데이터 수집
- 4가지 변화점 탐지 알고리즘 지원
- KoBERT/KcBERT 기반 텍스트 분석
- [상세 보기](./05_Social_Sentiment_Trend_Monitor/)

### 06. AIOps Realtime Monitor
실시간 스트리밍 데이터 기반 이상 탐지 및 모니터링 시스템
- 초당 수백 건 데이터 실시간 처리
- Isolation Forest + Z-score ML 기반 탐지
- HTTP/WebSocket/Socket 다중 소스 지원
- [상세 보기](./06_AIOps_Realtime_Monitor/)

### 07. News Trend Spike Monitor
뉴스 기반 실시간 트렌드 스파이크 모니터링 시스템
- KcELECTRA 기반 감정 분석 Latency 54ms 달성
- 스파이크 감지 F1-Score 0.80 달성
- RSS + Google News API 병행, TTL 10분 캐싱
- [상세 보기](./07_News_Trend_Spike_Monitor/)

## 기술 스택 요약

- 언어: Python 3.8+
- ML/DL: PyTorch, scikit-learn
- NLP: HuggingFace Transformers (KoBERT, KcBERT, KcELECTRA), LangChain
- 웹 프레임워크: Streamlit, FastAPI, Flask
- 클라우드: AWS Lambda, API Gateway, S3, DynamoDB, AWS CDK
- 데이터베이스: SQLite, DynamoDB, PostgreSQL
- 데이터 처리: Pandas, NumPy
- 시각화: Plotly, Matplotlib

## 연락처

- 이메일: hangayeong105@gmail.com
- GitHub: [@yanggangyiplus](https://github.com/yanggangyiplus)
