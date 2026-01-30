---
title: Torch memory 관리 팁
date: 2026-01-29 20:30:00 +0900
categories: [Quantization]
tags: [Quantization, pytorch]
math: true
---


# Torch memory 관리 팁

## Torch 레이어의 메모리 구조

- 기본적인 딥러닝 유닛들은 auto-grad 자동미분을 고려해서 중간에 가중치를 저장하는 구조
- 즉 레이어를 학습을 한다고 하면 아래와 같이 변수가 필요함
    1. 가중치 : 모델이 최종적으로 학습해야 되는 값
    2. 출력값/활성화 값 :  **학습 모드**에서는 미분(Chain Rule)을 계산하기 위해서 저장
    3. 미분값 : `backward()`가 호출될 때 생성되어 가중치의 `.grad` 속성에 저장
    
- 표로 정리하면 아래와같이 정리가됨        
        
| **층 종류** | **보관 항목 (학습 시)** | **이유** |
| --- | --- | --- |
| **연산 레이어** (Linear, Conv 등) | **가중치**,  **출력값**, **미분값** | 가중치 자체를 저장해야 하고, 역전파  때 가중치를 업데이트하기 위해 출력값(또는 입력값)이 필요하기  때문입니다. ⚙️ |
| **활성화 층** (ReLU, Sigmoid 등) | **입력값** (또는  출력값) | 층 자체에 학습할 가중치는 없지만, **이전 층으로 미분값을 전달(Chain Rule)**하기 위해 계산의 재료가 되는 값이 필요하기 때문입니다. ⚡ |

## Torch에서 메모리 절약 방법

- 즉 Tensor를 하나 만든다고 해서 numpy처럼 해당값만 들고 있는게 아니라 부대적으로 들고있는게 많은 구조가 됨. 학습에서는 필연적이지만 계산에 쓰이는 텐서를 별도로 관리하지 않으면 메모리가 누수가 될 수 있음

```python
#......
						
v = self.reshape_tensor(v)
#cur_max는 이전레이어에대한 grad 정보가 필요없으므로 detach
cur_max = v.max(axis=1).values.detach()
if self.max_val is None:
    self.max_val = cur_max
else:
    self.max_val = torch.max(cur_max, self.max_val)
    
#cur_min은 이전레이어에대한 grad 정보가 필요없으므로 detach
cur_min = v.min(axis=1).values.detach()
if self.min_val is None:
    self.min_val = cur_min
else:
    self.min_val = torch.min(cur_min, self.min_val)
if self.calibration_mode == 'layer_wise':
    self.max_val = self.max_val.max()
    self.min_val = self.min_val.min()

```

- 2번째는 별도로 추론과정에서 no_grad()를 선언해서 중간값을 저장하지 않고 전달만 하도록 해야 함

```python
#이런식으로 모델 자체를 추론해야될때는 torch.no_grad()를 쓰는게 맞음   
    qvit.set_mode('fp32')
    with torch.no_grad():
        out_qvit_fp32 = qvit.forward(x_test)

    qvit.set_mode('quantized')
    with torch.no_grad():
        out_qvit_quant = qvit.forward(x_test)
```