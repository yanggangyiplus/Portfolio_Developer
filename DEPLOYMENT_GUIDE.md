# 포트폴리오 배포 가이드

각 프로젝트를 실제로 배포하고 시각화할 수 있는 방법들을 정리했습니다.

## 배포 플랫폼 비교

| 플랫폼 | 무료 플랜 | 적합한 프로젝트 | 배포 난이도 |
|--------|----------|----------------|------------|
| Streamlit Cloud | ✅ | Streamlit 앱 (01, 02, 03, 05, 06, 07) | ⭐ 쉬움 |
| HuggingFace Spaces | ✅ | ML 모델 데모 (02) | ⭐⭐ 보통 |
| AWS Lambda | ✅ (제한적) | Serverless API (04) | ⭐⭐⭐ 어려움 |
| Vercel | ✅ | Next.js 앱 | ⭐⭐ 보통 |
| Railway | ✅ (제한적) | 전체 스택 앱 | ⭐⭐ 보통 |

## 프로젝트별 배포 방법

### 01. Coffee Roasting Tracking System

#### Streamlit Cloud 배포 (권장)

1. **GitHub 저장소 준비**
   ```bash
   cd 01_Coffee_Roasting_Tracking_System
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/coffee-roasting-tracking.git
   git push -u origin main
   ```

2. **Streamlit Cloud 배포**
   - [Streamlit Cloud](https://streamlit.io/cloud) 접속
   - GitHub 계정으로 로그인
   - "New app" 클릭
   - Repository 선택: `yourusername/coffee-roasting-tracking`
   - Main file path: `app/main.py`
   - Python version: 3.8+
   - Deploy 클릭

3. **환경 변수 설정** (필요시)
   - Streamlit Cloud 대시보드에서 Settings → Secrets
   - 필요한 API 키나 설정 추가

**배포 URL 예시**: `https://coffee-roasting-tracking.streamlit.app`

#### 로컬 실행
```bash
streamlit run app/main.py
```

---

### 02. AI Image Detector

#### HuggingFace Spaces 배포 (권장)

1. **HuggingFace Spaces 준비**
   ```bash
   # HuggingFace CLI 설치
   pip install huggingface_hub
   
   # 로그인
   huggingface-cli login
   ```

2. **Spaces 생성**
   ```bash
   # Spaces 생성
   huggingface-cli repo create ai-image-detector --type space --space_sdk streamlit
   
   # 파일 업로드
   cd 02_AI_Image_Detector
   huggingface-cli upload yourusername/ai-image-detector app/web/web_demo.py app/web/web_demo.py
   huggingface-cli upload yourusername/ai-image-detector requirements.txt requirements.txt
   ```

3. **README.md 작성** (Spaces용)
   ```markdown
   ---
   title: AI Image Detector
   emoji: 🖼️
   colorFrom: blue
   colorTo: purple
   sdk: streamlit
   sdk_version: 1.28.0
   app_file: app/web/web_demo.py
   pinned: false
   ---
   ```

**배포 URL 예시**: `https://huggingface.co/spaces/yourusername/ai-image-detector`

#### Streamlit Cloud 배포 (대안)
- GitHub에 푸시 후 Streamlit Cloud에서 배포
- Main file path: `app/web/web_demo.py`

---

### 03. Log AI Predictor

#### Streamlit Cloud 배포

1. **GitHub 저장소 준비**
   ```bash
   cd 03_Log_AI_Predictor
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/log-ai-predictor.git
   git push -u origin main
   ```

2. **Streamlit Cloud 배포**
   - Repository: `yourusername/log-ai-predictor`
   - Main file path: `app/web/main.py`
   - Deploy

**배포 URL 예시**: `https://log-ai-predictor.streamlit.app`

---

### 04. Serverless RAG Assistant

#### AWS Lambda + API Gateway 배포

1. **AWS CDK 설치 및 설정**
   ```bash
   cd 04_Serverless_RAG_Assistant/infrastructure/cdk
   
   # CDK 설치
   npm install -g aws-cdk
   
   # Python 의존성 설치
   pip install aws-cdk-lib constructs
   ```

2. **AWS 자격증명 설정**
   ```bash
   aws configure
   # AWS Access Key ID 입력
   # AWS Secret Access Key 입력
   # Default region: ap-northeast-2
   ```

3. **CDK 배포**
   ```bash
   # CDK 부트스트랩 (최초 1회)
   cdk bootstrap
   
   # 배포
   cdk deploy RagServerlessStack
   ```

4. **API Gateway 엔드포인트 확인**
   ```bash
   # 배포 후 출력된 API URL 확인
   # 예: https://xxxxx.execute-api.ap-northeast-2.amazonaws.com/prod/rag/query
   ```

5. **테스트**
   ```bash
   curl -X POST "https://your-api-url/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "RAG란 무엇인가요?", "top_k": 5}'
   ```

**배포 URL 예시**: `https://xxxxx.execute-api.ap-northeast-2.amazonaws.com/prod/rag/query`

#### 로컬 테스트 (Mock)
```bash
# Lambda 핸들러 로컬 테스트
python -c "
from src.api.query_handler import lambda_handler
event = {'body': '{\"question\": \"테스트\", \"top_k\": 5}'}
result = lambda_handler(event, None)
print(result)
"
```

---

### 05. Social Sentiment Trend Monitor

#### Streamlit Cloud 배포

1. **GitHub 저장소 준비**
   ```bash
   cd 05_Social_Sentiment_Trend_Monitor
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/social-sentiment-monitor.git
   git push -u origin main
   ```

2. **Streamlit Cloud 배포**
   - Repository: `yourusername/social-sentiment-monitor`
   - Main file path: `app/web_demo.py`
   - Secrets에 YouTube API 키 추가 (선택사항)

**배포 URL 예시**: `https://social-sentiment-monitor.streamlit.app`

---

### 06. AIOps Realtime Monitor

#### Streamlit Cloud 배포

1. **GitHub 저장소 준비**
   ```bash
   cd 06_AIOps_Realtime_Monitor
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/aiops-monitor.git
   git push -u origin main
   ```

2. **Streamlit Cloud 배포**
   - Repository: `yourusername/aiops-monitor`
   - Main file path: `app/web/dashboard.py`
   - Deploy

**배포 URL 예시**: `https://aiops-monitor.streamlit.app`

---

### 07. News Trend Spike Monitor

#### Streamlit Cloud 배포

1. **GitHub 저장소 준비**
   ```bash
   cd 07_News_Trend_Spike_Monitor
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/news-trend-monitor.git
   git push -u origin main
   ```

2. **Streamlit Cloud 배포**
   - Repository: `yourusername/news-trend-monitor`
   - Main file path: `app/web/main.py`
   - Deploy

**배포 URL 예시**: `https://news-trend-monitor.streamlit.app`

---

## 배포 전 체크리스트

### 공통 사항
- [ ] `requirements.txt` 파일이 최신 상태인지 확인
- [ ] `.gitignore`에 불필요한 파일 제외되어 있는지 확인
- [ ] README.md에 실행 방법이 명시되어 있는지 확인
- [ ] 환경 변수나 API 키가 필요한 경우 문서화

### Streamlit 앱 배포 시
- [ ] `streamlit run` 명령어로 로컬에서 정상 실행 확인
- [ ] 포트폴리오 루트가 아닌 프로젝트 루트에서 실행 가능한지 확인
- [ ] 대용량 파일이나 모델 파일이 `.gitignore`에 포함되어 있는지 확인

### AWS 배포 시
- [ ] AWS 계정 생성 및 자격증명 설정 완료
- [ ] CDK 부트스트랩 완료
- [ ] 필요한 AWS 서비스 권한 확인 (Lambda, API Gateway, S3, DynamoDB)

## 배포 후 관리

### Streamlit Cloud
- 대시보드에서 로그 확인 가능
- Settings에서 환경 변수 관리
- 자동 재배포 설정 가능 (GitHub push 시)

### HuggingFace Spaces
- Spaces 페이지에서 로그 확인
- Settings에서 하드웨어 리소스 설정
- 자동 재배포 설정 가능

### AWS Lambda
- CloudWatch에서 로그 및 메트릭 확인
- Lambda 콘솔에서 함수 설정 관리
- API Gateway에서 엔드포인트 관리

## 비용 예상

### 무료 플랜
- **Streamlit Cloud**: 무제한 앱, 무료
- **HuggingFace Spaces**: CPU 무료, GPU 제한적 무료
- **AWS Lambda**: 월 100만 요청 무료, 이후 $0.20/100만 요청
- **Vercel**: 무료 플랜 제공

### 유료 플랜 (참고)
- Streamlit Cloud Pro: $20/월
- AWS: 사용량 기반 과금
- HuggingFace Spaces GPU: 시간당 과금

## 배포 링크 추가 방법

배포 완료 후 각 프로젝트 README에 배포 링크를 추가하세요:

```markdown
## 배포 링크

- **Live Demo**: [Streamlit Cloud](https://your-app.streamlit.app)
- **GitHub**: [Repository](https://github.com/yourusername/project-name)
```

## 문제 해결

### Streamlit Cloud 배포 실패 시
1. 로그 확인: Streamlit Cloud 대시보드 → Logs
2. `requirements.txt` 확인: 모든 의존성이 명시되어 있는지 확인
3. Python 버전 확인: 3.8 이상인지 확인

### AWS 배포 실패 시
1. CloudWatch 로그 확인
2. IAM 권한 확인
3. 리전 설정 확인

### HuggingFace Spaces 배포 실패 시
1. Spaces 로그 확인
2. `requirements.txt` 확인
3. 파일 경로 확인

