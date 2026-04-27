---
title: Optimization of CLIP Distillation for Mobile
date: 2026-03-11 16:30:00 +0900
categories: [Deeplearning, Knowledge Distillation]
tags: [clip, distillation, mobile, knowledge-distillation]
math: true
---

## 1. Current Status

### 1.1 LPCV 2026 Competition Constraints

| **Category**   | **Requirement**                                       |
| -------------- | ----------------------------------------------------- |
| **Task**       | Image-to-Text Retrieval                               |
| **Metric**     | Recall@10                                             |
| **Latency**    | Image + Text < 35ms (on Qualcomm XR2 Gen 2)           |
| **Format**     | ONNX → QAI Hub Compilation                            |
| **Parameters** | No explicit limit (indirectly constrained by latency) |

### 1.2 Baseline Performance

- **Recall@10 (Submission):** 74.96%
- **Top-1 Accuracy (ImageNet-mini, CLIP mode):** 0.4596
- **Student Latency:** 16.0ms (Current margin: 19ms)
- **Teacher (ViT-B/32) Top-1:** 0.635
- **Teacher-Student Gap:** ~17%p
### 1.3 Architecture
- **Teacher:** CLIP ViT-B/32 (512-dim, frozen)
- **Student (Vision):** ConvNeXtV2-Tiny (28M) → Conv1x1 (768 to 512) → 3×Blocks (512) → GAP → 512-dim
- **Student (Text):** ViT-B/32 pruned (8/12 layers, 512-dim)
    

### 1.4 Current Loss Configuration

$$L_{Total} = L_{Affinity} + \beta \cdot L_{MSE}$$

- **Affinity Loss:** Cross-Entropy (CE) between student similarity matrix and softmaxed teacher similarity matrix.
    
- **MSE Loss:** $\|v^S - v^T\|^2$ (with $\beta=100$)
    

---

## 2. Problem Diagnosis

### 2.1 Embedding Dimension Mismatch (Critical Bottleneck)

Distilling into a mismatched dimension prevents a pretrained student from effectively inheriting the teacher's latent space. While original papers successfully use projections, they often train students from scratch. For pretrained students, forced alignment via projection often leads to representation collapse.

### 2.2 Suboptimal Training Strategy

The current setup relies only on two losses. Modern SOTA (State-of-the-Art) methods utilize 3–5 complementary objectives. Integrating **ICL (Interactive Contrastive Learning)** and **CRD (Contrastive Relational Distillation)** is essential. A more effective strategy involves pre-aligning encoders using Cosine Similarity before joint distillation.

### 2.3 Computational Constraints

CLIP's performance scales with batch size and data volume (Scaling Law). OpenAI utilized batches of ~32,000 samples. Small-batch distillation in personal environments often fails to capture the global distribution required for robust retrieval.

---

## 3. Proposed Loss Functions

### 3.1 Contrastive Relational Distillation (CRD)

CRD compares softmax probability distributions, allowing the student to learn relative relationships between samples rather than just hard labels.

$$L_{i2t} = KL\left( \text{log\_softmax}\left(\frac{s_{i2t}}{T}\right) \| \text{softmax}\left(\frac{t_{i2t}}{T}\right) \right)$$

$$L_{CRD} = \frac{1}{2}(L_{i2t} + L_{t2i}) \cdot T^2$$

### 3.2 Interactive Contrastive Learning (ICL)

ICL directly validates student embeddings against teacher embeddings across modalities (Student Image ↔ Teacher Text).

$$L_{ICL, I \to T} = -\log \frac{\exp(v_k^S \cdot s_k^T / \tau)}{\sum_{b=1}^{|B|} \exp(v_k^S \cdot s_b^T / \tau)}$$

_Effect: Enforces a one-to-one alignment with the teacher's high-quality feature space._

### 3.3 Student Self-Contrastive Loss ($L_{itc}$)

Maintains the student's inherent image-text alignment to prevent degradation during distillation.

$$L_{itc} = \frac{1}{2}(CE(s_{i2t}, labels) + CE(s_{t2i}, labels))$$

_Note: Personal observation suggests this might be redundant if MSE is already well-tuned._

---

## 4. Internal Test Results & Conclusion

### 4.1 Pitfalls of Pretrained Student Distillation

Forcing a pretrained student to align with a teacher often disrupts its pre-existing feature distribution, leading to a drop in accuracy. Scratch distillation is often more stable for these frameworks, but computationally expensive.

### 4.2 Optimized Roadmap

For resource-constrained environments, the most effective path is:

1. **Dimension Matching:** Initialize the student with the same embedding dimension as the teacher to avoid noisy projections.
    
2. **Pre-alignment:** Perform an initial stage of alignment using **Cosine Similarity + MSE** for each encoder independently.
    
3. **Low-LR Refinement:** Finalize with a multi-loss distillation (FD + ICL + CRD) using a very low Learning Rate.
    

**Final Verdict:** Without massive compute, brute-force distillation from scratch is inefficient. Success lies in structural alignment (2.1) and staged pre-alignment (2.2) rather than complex loss combinations alone.

## 5. Reference

1. **TinyCLIP** (ICCV 2023) — Affinity Mimicking + Weight Inheritance
2. **CLIP-KD** (CVPR 2024) — Interactive Contrastive Learning, 
3. **MobileCLIP** (CVPR 2024) —  Structured Pruning + Multi-loss Framework
4. **ConvNeXt V2** (ICCV 2023) — ConvNeXt Archtecture


---
