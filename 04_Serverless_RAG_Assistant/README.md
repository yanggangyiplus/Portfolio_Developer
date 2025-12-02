# Serverless RAG Assistant

AWS Serverless 기반 RAG(Retrieval-Augmented Generation) 서비스

## 문제 정의

기존 문서 검색 시스템의 정확도가 낮아 사용자가 원하는 정보를 찾기 어려웠습니다. 또한 문서 업로드부터 검색 가능한 상태로 만드는 과정이 수동으로 이루어져 효율성이 떨어졌습니다.

## 해결 방법

1. **LangChain 기반 RAG 파이프라인 구축**: RetrievalQA Chain을 활용하여 문서 기반 질의응답 시스템을 구현했습니다.
2. **문서 처리 파이프라인 구축**: PDF, TXT, MD 파일을 파싱하고 전처리하는 파이프라인을 개발했습니다.
3. **벡터 기반 검색 엔진 구현**: 문서 내용을 벡터화하여 유사도 기반 검색이 가능하도록 구현했습니다.
4. **청킹 최적화**: 문서를 적절한 크기로 나누어 검색 정확도를 향상시켰습니다.
5. **Serverless 아키텍처 적용**: AWS Lambda, API Gateway, S3, DynamoDB를 활용하여 서버 관리 없이 동작하는 시스템을 구축했습니다.

## 성과

- 문서 기반 질의응답 정확도 85% 달성
- AWS Lambda 콜드스타트 2초 이내로 최적화
- S3 업로드 시 자동으로 문서 처리 및 검색 가능 상태로 변환
- AWS CDK를 통한 인프라 코드화 및 자동 배포 구현

## 기술 스택

`Python` `LangChain` `AWS Lambda` `API Gateway` `S3` `DynamoDB` `AWS CDK` `Streamlit`

## 프로젝트 구조

```
04_Serverless_RAG_Assistant/
├── src/
│   ├── api/
│   │   └── query_handler.py    # Lambda 핸들러
│   ├── rag/
│   │   └── pipeline.py          # RAG 파이프라인
│   └── preprocessing/
│       └── chunker.py           # 문서 청킹
├── infrastructure/cdk/
│   └── rag_serverless_stack.py # AWS CDK 인프라 코드
├── configs/
│   └── config_rag.yaml
├── docs/
│   └── DEPLOYMENT_AWS.md
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

### 로컬 테스트
```bash
# Lambda 핸들러 로컬 테스트
python -m src.api.query_handler

# 문서 청킹 테스트
python -c "from src.preprocessing.chunker import DocumentChunker; chunker = DocumentChunker(); print(chunker.chunk_text('테스트 문서입니다.'))"
```

### AWS 배포
```bash
# CDK 스택 배포
cd infrastructure/cdk
cdk deploy RagServerlessStack
```

### 테스트 실행
```bash
# 모든 테스트 실행
pytest tests/

# 청킹 모듈 테스트
pytest tests/test_chunker.py -v
```

## 프로젝트 위치

`../../Serverless-RAG-Assistant/`
