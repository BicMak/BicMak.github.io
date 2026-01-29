---
title: Quantization scale,zero
date: 2026-01-28 20:30:00 +0900
categories: [Quantization]
tags: [Quantization, pytorch]
math: true
---


## 1. 선형양자화 개요

![alt text](/assets/img/2026-01-02-FP2Int/image.png)

- 양자화를 표현 하는 나타내려고 하는 데이터셋의 range를 균일한 간격으로 구간을 만들어서 숫자를 압축 시키는 방법
- 수학적으로 구현이 쉽고 모든 구간에서 발생하는 최대 오차가 일정하기 때문에 노이즈 예측이 쉬움 (int8 기준 이론상 49.9 dB 까지 QSNR을 구현가능함)
$$
\text{QSNR}= 6.02* \text{bits} + 1.76
$$

- fp32를 가장 가까운 구간에 반올림 하는 과정에서 QSNR을 관리하는것이 중요함


## 2 캘리브레이션

- calibration은 static 양자화를 하기 위해서 각 가중치와 출력정보를 양자화할 S,Zero를 구하는 방법

### 2.1 Minmax 
- 데이터 혹은 가중치 셋에서 min,max값을 기준으로 S,Z 산출
- outlier가 적고 데이터가 균등하게 분포되어있다면 사용하기 유리함

```python
#asymmetric quant paras
scale = (max_val - min_val) / float(qmax - qmin)
scale.clamp_(self.eps)
zero_point = qmin - torch.round(min_val / scale)
zero_point.clamp_(qmin, qmax)
        
```

### 2.2 Percentile 
- outlier가 있다면 outlier를 무시하고 S,Z를 산출 하는 방법
- Outlier를 희생하는 대신에 대다수의 데이터 분포를 세밀하게 가져갈 수 있음

```python
#min/max 산출 과정에서 별도로 클리핑을 해서 구현해주면

cur_max = torch.quantile(v, selfpercentile_alpha, dim=1)
cur_min = torch.quantile(v, 1.0 - selfpercentile_alpha, dim=1)

if self.max_val is None:
    self.max_val = cur_max
else:
    self.max_val = self.max_val + \
        self.percentile_sigma * (cur_max - self.max_val)

if self.min_val is None:
    self.min_val = cur_min
else:
    self.min_val = self.min_val + \
        self.percentile_sigma * (cur_min - self.min_val)

```


### 2.3 Omse 
- 하지만 Percentile이 유리하지만 outlier를 제거하는 비중을 수작업으로 찾기는 어렵기 때문에 양자화 MSE로스를 활용
- 구간별로 모든값을 탐색하기 때문에 percentile이나 minmax 대비 속도가 느림

```python
for keep_ratio in keep_ratios:
    new_max = max_val * keep_ratio
    new_min = min_val * keep_ratio
    if self.symmetric:
        new_max = torch.max(-new_min, new_max)
        new_scale = new_max / (float(qmax - qmin) / 2)
        new_scale = torch.clamp(new_scale, min=eps)  # 최소값 보장
        new_zero_point = torch.zeros_like(new_max, dtype=torch.int64)
    else:
        new_scale = (new_max - new_min) / float(qmax - qmin)
        new_scale = torch.clamp(new_scale, min=eps)  # 최소값 보장
        new_zero_point = qmin - torch.round(new_min / new_scale)
        new_zero_point = new_zero_point.clamp(qmin, qmax)
    # Quantize & Dequantize
    if self.calibration_mode == 'layer_wise':
        inputs_q = ((all_data / new_scale + new_zero_point).round().clamp(
            qmin, qmax) - new_zero_point) * new_scale
    else:
        # channel_wise
        new_scale_expanded = new_scale.unsqueeze(1)
        new_zero_point_expanded = new_zero_point.unsqueeze(1)
        inputs_q = ((all_data / new_scale_expanded + new_zero_point_expanded).round().clamp(
            qmin, qmax) - new_zero_point_expanded) * new_scale_expanded
    # L2 loss
    score = lp_loss(all_data, inputs_q, p=1.0, reduction='all')

```

### 2.3 KL divergence 
- 양자화 전/후의 데이터셋에서 히스토그램을 산출하여 매칭여부를 확인하는 방법
$$D_{KL}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \left( \frac{P(x)}{Q(x)} \right)$$

- 가장 확실하지만 연산속도가 느리다는 문제가 있음


출처 : https://hanlab.mit.edu/courses/2024-fall-65940

