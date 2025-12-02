# AI Image Detector

딥러닝 기반 이미지 분류 시스템

## 문제 정의

이미지 분류 작업에서 다양한 조명 조건과 이미지 품질에 따른 성능 저하 문제가 있었습니다. 기존 시스템의 정확도를 개선하고 안정적인 서비스를 제공하는 것이 목표였습니다.

## 해결 방법

1. **모델 비교 실험**: ResNet18과 Vision Transformer 모델을 비교하여 프로젝트에 적합한 모델을 선정했습니다.
2. **이미지 전처리 개선**: Histogram Equalization을 적용하여 다양한 조명 조건에서도 안정적인 성능을 확보했습니다.
3. **클래스 불균형 해결**: Weighted Loss를 적용하여 데이터 불균형 문제를 해결했습니다.
4. **성능 최적화**: 데이터 증강, 전처리, 손실 함수 등 다양한 요소를 실험하여 최적의 조합을 찾았습니다.

## 성과

- 이미지 분류 정확도 97.06% 달성
- Baseline 대비 약 2.12%p 성능 향상
- 실시간 추론이 가능하도록 모델 최적화 완료
- HuggingFace Spaces에 배포하여 서비스 제공

## 기술 스택

`Python` `PyTorch` `Vision Transformer` `ResNet18` `Streamlit` `FastAPI`

## 프로젝트 구조

```
02_AI_Image_Detector/
├── app/web/
│   └── web_demo.py         # Streamlit 웹 데모
├── src/
│   ├── models/
│   │   ├── cnn.py          # ResNet18 CNN 모델
│   │   └── vit.py           # Vision Transformer 모델
│   ├── training/
│   │   └── train.py        # 학습 파이프라인
│   └── inference/
│       └── inference.py    # 추론 엔진
├── configs/
│   └── config_cnn.yaml
├── docs/
│   └── ABLATION_STUDY.md
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

### 웹 데모 실행
```bash
# Streamlit 웹 데모 실행
streamlit run app/web/web_demo.py
```

### 모델 학습
```bash
# CNN 모델 학습
python src/training/train.py --model cnn --config configs/config_cnn.yaml

# ViT 모델 학습
python src/training/train.py --model vit --config configs/config_cnn.yaml
```

### 테스트 실행
```bash
# 모든 테스트 실행
pytest tests/

# 모델 테스트
pytest tests/test_cnn_model.py -v
```

## 배포 방법

### HuggingFace Spaces 배포 (권장)
1. HuggingFace Spaces에 프로젝트 업로드
2. SDK: Streamlit
3. App file: `app/web/web_demo.py`

### Streamlit Cloud 배포 (대안)
1. GitHub에 프로젝트 푸시
2. [Streamlit Cloud](https://streamlit.io/cloud)에서 배포
3. Main file path: `app/web/web_demo.py`

자세한 배포 방법은 `../../DEPLOYMENT_GUIDE.md`를 참조하세요.

## 프로젝트 위치

`../../Ai-image-detector/`
