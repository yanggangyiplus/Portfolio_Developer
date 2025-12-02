# 포트폴리오 실행 가이드

## 전체 프로젝트 실행 방법

### 공통 사전 준비
```bash
# Python 3.8 이상 필요
python --version

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 프로젝트별 실행 방법

#### 01. Coffee Roasting Tracking System
```bash
cd 01_Coffee_Roasting_Tracking_System
pip install -r requirements.txt
streamlit run app/main.py
```

#### 02. AI Image Detector
```bash
cd 02_AI_Image_Detector
pip install -r requirements.txt
streamlit run app/web/web_demo.py
```

#### 03. Log AI Predictor
```bash
cd 03_Log_AI_Predictor
pip install -r requirements.txt
streamlit run app/web/main.py
```

#### 04. Serverless RAG Assistant
```bash
cd 04_Serverless_RAG_Assistant
pip install -r requirements.txt

# 로컬 테스트
python -c "from src.preprocessing.chunker import DocumentChunker; chunker = DocumentChunker(); print(chunker.chunk_text('테스트 문서입니다.'))"

# AWS 배포 (CDK 설치 필요)
cd infrastructure/cdk
cdk deploy
```

#### 05. Social Sentiment Trend Monitor
```bash
cd 05_Social_Sentiment_Trend_Monitor
pip install -r requirements.txt

# YouTube API 키 설정 (선택사항)
export YOUTUBE_API_KEY="your_api_key"

streamlit run app/web_demo.py
```

#### 06. AIOps Realtime Monitor
```bash
cd 06_AIOps_Realtime_Monitor
pip install -r requirements.txt
streamlit run app/web/dashboard.py
```

#### 07. News Trend Spike Monitor
```bash
cd 07_News_Trend_Spike_Monitor
pip install -r requirements.txt
streamlit run app/web/main.py
```

## 테스트 실행

### 전체 테스트 실행
```bash
# 각 프로젝트 디렉토리에서
pytest tests/ -v

# 커버리지 포함
pytest tests/ --cov=src --cov-report=html
```

### 프로젝트별 테스트
```bash
# 프로젝트 01
cd 01_Coffee_Roasting_Tracking_System && pytest tests/

# 프로젝트 02
cd 02_AI_Image_Detector && pytest tests/

# 프로젝트 03
cd 03_Log_AI_Predictor && pytest tests/

# 프로젝트 04
cd 04_Serverless_RAG_Assistant && pytest tests/

# 프로젝트 05
cd 05_Social_Sentiment_Trend_Monitor && pytest tests/

# 프로젝트 06
cd 06_AIOps_Realtime_Monitor && pytest tests/

# 프로젝트 07
cd 07_News_Trend_Spike_Monitor && pytest tests/
```

## 주의사항

1. **의존성 충돌**: 각 프로젝트는 독립적으로 실행 가능하도록 설계되었습니다. 동시에 여러 프로젝트를 실행할 때는 별도 가상환경 사용을 권장합니다.

2. **API 키 필요**: 
   - 프로젝트 05: YouTube API 키 (선택사항, 없으면 Mock 데이터 사용)
   - 프로젝트 04: AWS 자격증명 (배포 시)

3. **포트 충돌**: Streamlit은 기본적으로 8501 포트를 사용합니다. 여러 프로젝트를 동시에 실행할 때는 포트를 변경하세요.
   ```bash
   streamlit run app/main.py --server.port 8502
   ```

4. **데이터 파일**: 일부 프로젝트는 샘플 데이터가 필요할 수 있습니다. 원본 레포지토리를 참조하세요.

## 문제 해결

### ImportError 발생 시
```bash
# 프로젝트 루트에서 실행 확인
pwd

# sys.path 확인
python -c "import sys; print(sys.path)"
```

### 의존성 설치 오류 시
```bash
# pip 업그레이드
pip install --upgrade pip

# 개별 패키지 설치
pip install package_name
```

