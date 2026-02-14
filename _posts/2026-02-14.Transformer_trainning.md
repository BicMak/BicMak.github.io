---
title: Transformer 학습이 어려운 이유
date: 2026-02-14 16:30:00 +0900
categories: [Tips]
tags: [transformer, attention, gradient, optimization]
math: true
---


# 1. Transformer 가 근본적으로 학습이 어려운 이유
---
## 1. Softmax 포화(Saturation)와 기울기 소실 

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

-  $\text{softmax}$의 함수의 정의 자체가 비선형 함수
-  $\text{softmax}$ 함수 $P_i$에 대한 입력값 $z_j$ 에대해서 미분을 하게 되면 아래와 같이 표현이 가능한데 여기서 $P_{i}$ 의 영향을 많이 받게됨

$$\frac{\partial P_i}{\partial z_j} = \begin{cases} P_i(1 - P_i) & \text{if } i = j \\ -P_i P_j & \text{if } i \neq j \end{cases}$$

- 여기서 $P_{i} = 0.9999$ 정도 라고 한다면
-  **자기 자신에 대한 영향 ($i=j$):**
    $$\frac{\partial P_i}{\partial z_i} = P_i(1 - P_i) \approx 0.999 \times (1 - 0.999) = 0.000999$$
    
- **다른 항목에 대한 영향 ($i \neq j$):**
    $$\frac{\partial P_i}{\partial z_j} = -P_i P_j \approx -0.999 \times (\text{매우 작은 값}) \approx 0$$
    
- 즉 모든 경우에 대해서 0으로 수렴해버리는 문제가 발생하게됨
- 결론을 이야기하면 확률이 hard label로 0과1에 대해서 극단적으로 값이 모여있다면 모든 미분값이 0으로 수렴되버리는 문제가 발생함

## 2. Attention의 미분문제
- Transformer의 미분에서 가장 큰 문제는 모든 Query,Key 값들이 서로 가중치 학습을 하는데 병렬적인 관계로 영향을 주게 되는 문제가 있음 


$$Y = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$


1. **$V$(Value) 경로:** 
$$ \frac{\partial L}{\partial W_v} =\frac{\partial L}{\partial \text{Y}} \cdot \frac{\partial \text{Y}}{\partial W_v}$$
    - 여기서 보게 되면 Value의 가중치의 그래디언트는 Query와 Key의 attention으로 결정이 됨
2. **$Q$(Query)와 $K$(Key) 경로:** 
	$$\frac{\partial L}{\partial W_Q} = \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial \text{Softmax}} \cdot \frac{\partial \text{Softmax}}{\partial (QK^T)} \cdot \frac{\partial (QK^T)}{\partial Q} \cdot \frac{\partial Q}{\partial W_Q}$$
	- 그러면 Query를 기준으로 보게 되면 Query도 역시 key의 값에 의해서 의존적인 문제가 있음

## 3. 헤시안 행렬(Hessian Matrix)과 곡률 
- 앞에서 이야기 한거처럼 헤시안으로 해당 레이어에서 변수간의 간섭이 어떻게 되는지 봐도 벌써부터 복잡해짐

$$\frac{\partial L}{\partial w_{i}w_{j}} = \text{Tr} \left( \left( \frac{\partial L}{\partial Q} \right)^T \frac{\partial Q}{\partial w_{i} w_{j}} \right)$$ 

