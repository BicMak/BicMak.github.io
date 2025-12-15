---
title: Pose Estimation에서 AP 스코어 평가방법
date: 2024-12-15 14:30:00 +0900
categories: [Computer Vision, Pose Estimation]
tags: [pose estimation, evaluation metrics, oks, coco, vitpose]
math: true
---

Pose estimation 모델을 평가할 때 COCO 데이터셋 기준으로 AP (Average Precision) 스코어를 사용한다. Object detection의 IoU와 달리, pose estimation은 **OKS (Object Keypoint Similarity)** 기반으로 평가한다는 점이 핵심이다.

## OKS (Object Keypoint Similarity)

OKS는 예측된 keypoint와 ground truth 간의 유사도를 측정하는 지표다. 

$$
\text{OKS} = \frac{\sum_i \exp\left(-\frac{d_i^2}{2s^2k_i^2}\right) \delta(v_i > 0)}{\sum_i \delta(v_i > 0)}
$$

여기서:
- $d_i$ : i번째 keypoint의 예측값과 GT 사이의 유클리드 거리
- $s$ : 객체 스케일 ($\sqrt{\text{area}}$)
- $k_i$ : i번째 keypoint의 정규화 상수 (신체 부위별로 다름)
- $v_i$ : keypoint의 visibility flag (0: not labeled, 1: labeled but not visible, 2: labeled and visible)

### Keypoint별 상수 (k)

COCO 기준으로 각 keypoint마다 다른 허용 오차를 적용한다:

| Keypoint | k 값 | 설명 |
|----------|------|------|
| 눈, 코 | 0.025 | 정밀한 위치가 중요 |
| 어깨, 팔꿈치 | 0.079 | 중간 허용 오차 |
| 손목, 골반 | 0.072 | 중간 허용 오차 |
| 무릎, 발목 | 0.087 | 상대적으로 큰 허용 오차 |

작은 k 값일수록 더 정확한 위치를 요구한다.

## AP 메트릭 종류

COCO keypoint evaluation에서 사용하는 주요 지표:

### 1. AP (Average Precision)
- OKS 임계값 0.5부터 0.95까지 0.05 간격으로 측정
- 10개 임계값의 평균값
- **가장 중요한 지표**

### 2. AP50
- OKS ≥ 0.5일 때의 Average Precision
- 상대적으로 관대한 기준

### 3. AP75  
- OKS ≥ 0.75일 때의 Average Precision
- 엄격한 기준

### 4. APM / APL
- **APM**: Medium 크기 객체 (32² < area < 96²)
- **APL**: Large 크기 객체 (area > 96²)
- 객체 크기별 성능 분석에 유용

## ViTPose 성능 예시

| 모델 | AP | AP50 | AP75 | APM | APL |
|------|-----|------|------|-----|-----|
| ViTPose-H | 75.8 | 90.5 | 83.2 | 71.5 | 82.1 |
| ViTPose-L | 73.2 | 89.4 | 81.0 | 69.2 | 79.8 |

> ViTPose는 transformer 기반 architecture로 YOLO-pose보다 높은 정확도를 보이지만, 추론 속도는 느린 편이다.
{: .prompt-info }

## Heatmap 기반 Confidence Score

ViTPose는 각 keypoint마다 heatmap을 생성하고, 여기서 confidence score를 추출한다:

1. 17개 joint → 17개 heatmap 생성
2. 각 heatmap에서 최댓값 = confidence score
3. 값 범위: 0~1 (높을수록 신뢰도 높음)
```python
# Heatmap에서 keypoint 추출 (예시)
max_vals = heatmaps.max(dim=(2, 3))  # [B, 17]
keypoints_conf = max_vals.values     # confidence scores
```

## IoU vs OKS 차이점

| 구분 | IoU | OKS |
|------|-----|-----|
| 용도 | Object Detection | Pose Estimation |
| 측정 대상 | Bounding box 겹침 | Keypoint 거리 |
| 스케일 보정 | 없음 | 있음 (s 파라미터) |
| 부위별 가중치 | 없음 | 있음 (k 파라미터) |

**핵심**: Pose estimation에서 AP50, AP75는 IoU가 아닌 **OKS 임계값**을 의미한다.

## 실무 적용 시 고려사항

1. **엣지 디바이스 배포**: AP는 높지만 FPS가 낮으면 실시간 처리 불가
2. **APM vs APL**: 원거리 검출이 필요하면 APM 성능 중요
3. **AP75**: 정밀한 위치 추정이 필요한 경우 (의료, 스포츠 분석 등) 중점 확인

> Hailo NPU나 edge device에 배포할 때는 quantization 후 AP 하락폭도 함께 측정해야 한다.
