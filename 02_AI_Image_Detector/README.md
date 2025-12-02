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

## 프로젝트 위치

`../../Ai-image-detector/`
