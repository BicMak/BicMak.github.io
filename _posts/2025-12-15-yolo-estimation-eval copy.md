---
title: Object detection에서 AP 스코어 평가방법
date: 2024-12-15 14:30:00 +0900
categories: [Deeplearning Evaluation, Object detection]
tags: [Object detection, evaluation metrics, yolo]
math: true
---

# Object Detection 평가 지표

Object Detection 모델의 성능을 평가하기 위한 주요 지표들을 정리

![alt text](/assets/img/2024-12-15-pose-estimation/image.png)
## 1. Precision (정밀도)
### 정의
분류 평가 지표로, 모델이 탐지한 바운딩 박스 중에서 실제로 객체를 올바르게 분류한 비율을 나타냅니다.

### 특징
- **Precision만 높은 경우**: 감지는 잘 못하지만, 한번 탐지하면 분류 정확도가 높음
- **YOLO의 예시**: 배경을 물체로 잘못 감지하는 경우가 많다면 Precision이 낮게 됨

### 평가 요소
Precision은 두 가지 관점으로 평가됩니다:
1. 바운딩 박스의 objectness는 Positive지만, classification을 제대로 하는지
2. 바운딩 박스가 objectness 자체를 제대로 감지하는지

### 수식
```
Precision = TP / (TP + FP)
```

---

## 2. Sensitivity (Recall, 재현율)

### 정의
감지 평가 지표로, 실제 객체 중에서 모델이 얼마나 잘 감지하는지를 나타냅니다.

### 특징
- Object Detection에서는 바운딩 박스를 생성할 수 있는지 여부를 평가
- **Recall만 높은 경우**: 감지는 잘 하지만, 분류 성능이 떨어짐

### 수식
```
Recall = TP / (TP + FN)
```

---

## 3. F1 Score

### 정의
Precision과 Recall의 조화평균으로, 두 지표 간의 균형을 평가합니다.

### 특징
- **1에 가까울수록**: 두 지표의 밸런스가 좋음
- **0에 가까울수록**: 한쪽 지표만 극단적으로 높음

### 수식
```
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## 4. mAP (mean Average Precision)
![alt text](/assets/img/2024-12-15-pose-estimation/image-1.png)
### 정의
Average Precision을 활용하여 모델의 종합적인 성능을 나타내는 지표입니다.

### 계산 방법
1. 클래스별로 confidence threshold를 변경하면서 Precision-Recall 커브를 생성
2. 각 클래스의 AP(Area Under Curve) 계산
3. 전체 클래스의 AP 평균을 구하여 mAP 산출

### 중요 포인트
- **IoU 고정**: 하나의 mAP 값은 특정 IoU threshold에서 측정
- 일반적으로 IoU=0.5를 기준으로 사용 (mAP@0.5 또는 mAP50)

### mAP50-95
```
mAP50-95 = IoU 0.50부터 0.95까지 0.05 간격으로 측정한 mAP의 평균
```

- IoU threshold: [0.50, 0.55, 0.60, ..., 0.90, 0.95] (총 10개)
- 각 IoU threshold에서 mAP 계산
- 10개 mAP 값의 평균

---

## 시각화
![alt text](/assets/img/2024-12-15-pose-estimation/image-2.png)
### Precision-Recall Curve
![PR Curve](이미지 경로)

- X축: Recall
- Y축: Precision
- 커브 아래 면적(AUC)이 AP(Average Precision)

