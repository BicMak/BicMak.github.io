---
title: Knowledge
date: 2026-02-01 16:30:00 +0900
categories: [Knowledge distliation]
tags: [Knowledge distliation, efficient ml]
math: true
---


## 1. knowledge distliation 이 등장하게 된 이유
- 현대의 머신러닝의 모델은 점점 스케일링 법칙에 의해서 가중치가 늘어나는 방향으로 가고있음
- 대부분의 하드웨어는 저장공간과 연산자원에 한계가 있기 때문에 딥러닝 기반 모델을 배포하는데 문제가 생김
- 큰 스케일의 모델의 특성을 작은 모델에게 교육을 시켜서 온디바이스 환경에 대응하기 위해서 등장

## 2. Knowledge 
- 지식증류법에 지식은 학생이 선생모델로 부터 배워야될 규칙을 의미함
> 대표적인 지식 증류법에는
> 1. Response based : 최종 출력값을 기준
> 2. Feature based : 모델의 중간 출력값에 대한 매칭을 기준으로함
> 3. Relation-based : 비교하려는 레이어에서 상대적인 관계성

### 2.1 Response-based distilation
![alt text](/assets/img/2026-02-01-Knowledge%20distliation/image.png)

- 최종 결과값만 비교하는 방식으로 가장 비교가 쉬움
$$L_{ResD}(z_t, z_s) = L_R(z_t, z_s)$$
- 최종 확률값에 대해서만 결과를 비교함, 대표적으로 L2-loss와 KL-divergence로 계산함


#### L2 Loss(MSE)
$$L_{L2}(z_t, z_s) = \frac{1}{N} \sum_{i=1}^{N} (z_t^{(i)} - z_s^{(i)})^2$$
- 모든 클래스를 동등하게 보고 계산
- 각 개별 클래스에 대한 특성은 학습하지 않음

#### KL divergence
$$L_{KL}(p_t || p_s) = \sum_{i=1}^{C} p_t^{(i)} \cdot \log \frac{p_t^{(i)}}{p_s^{(i)}}$$
- 각 구간별 비교하여 분포자체가 일치하는지 검증

#### Temperature Scaling과 Soft Targets


**일반 Softmax (T=1):**
$$p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

**Temperature Softmax:**
$$p_i^T = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

- 여기서 $T$는 temperature parameter
- Teacher model의 확률 값은 T가 높을수록 상대적이 차이가 줄어들어서 학생모델에게 암묵적인 지식을 전수함



### 2.2 Feature-based
![alt text](/assets/img/2026-02-01-Knowledge%20distliation/image-1.png)

- 학생 모델이 선생모델의 활성화 레이어의 특성을 익혀서 매칭시키는 방법
- 중간과정도 학습시키는 방법이라고 보면됨
- KL divergence와 코사인 유사도로 loss 검증

$$L_{feat} = \sum_{i=1}^{M} \|f_t^{(i)} - f_s^{(i)}\|_2^2$$

#### cosine similarity
$$L_{cos} = 1 - \frac{f_t \cdot f_s}{\|f_t\|_2 \cdot \|f_s\|_2}$$


### 2.3 Relation-based
![alt text](/assets/img/2026-02-01-Knowledge%20distliation/image-2.png)

- feature based로 계산하려고 하면 teacher와 student사이에 레이어의 차원이 맞아야 되는 문제가 발생함
- 그래서 각 모델마다 상대적인 관계만 추론하는 Huber loss가 등장하게됨
- 정규화를 통해 절대값이 아닌 상대적 비율만 비교

$$L_{RKD} = L_{RKD-D} + L_{RKD-A}$$

**Distance-wise (Huber loss):**
$$L_{RKD-D} = \sum_{(i,j) \in \mathcal{P}} \ell_\delta \left( \psi(d_t^{ij}) - \psi(d_s^{ij}) \right)$$

**Angle-wise (Huber loss):**
$$L_{RKD-A} = \sum_{(i,j,k) \in \mathcal{T}} \ell_\delta \left( \cos\theta_t^{ijk} - \cos\theta_s^{ijk} \right)$$

여기서:
- $\ell_\delta$: Huber loss (outlier에 robust)
- $\psi$: 정규화 함수 (평균으로 나눔)
- $d^{ij}$: 샘플 i, j 간의 거리
- $\cos\theta^{ijk}$: 샘플 i, j, k로 만든 각도