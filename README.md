# Multimodal Scientific Question-Answering on ScienceQA


> **Columbia COMS4705 Natural Language Processing - Final Project**
>
> *Authors:* Reneth Raj Simon, Sreeram Raghammudi, Vishal Menon  
> *Institution:* Columbia University, Department of Computer Science

## Overview
This project investigates effective and resource-efficient strategies for solving multimodal scientific questions using the **ScienceQA** benchmark. We explore and compare three classes of models: standard large generative baselines, custom-built discriminative hybrid architectures, and novel Chain-of-Thought (CoT) fine-tuning pipelines.

---

## Key Results

| Approach | Best Model | Overall Acc | Parameters | Training Time |
|----------|-----------|-------------|------------|---------------|
| **Decoupled CoT** | Qwen3-8B (LoRA) + BLIP | **78.0%** | 8B (LoRA: ~32M) | 3 hours |
| **Hybrid Architectures** | hybrid_v2 | **67.4%** | 141M | 1.5 hours |
| **Standard Baselines** | Qwen2.5-1.5B + GIT | **59.0%** | 1.5B | 5 hours |
| **Instruction-Tuned CoT** | UnifiedQA CoT | **48.2%** | 223M | 4 hours |

---

## Dataset

**ScienceQA** ([derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA))

- Training: 6,218 examples
- Validation: 1,726 examples  
- Test: 2,017 examples

Each example contains: question, context (optional), image (optional), choices, answer, rationale

---

## Models

### Baseline Models

| Model | Overall Acc | Parameters | Architecture |
|-------|-------------|------------|--------------|
| Qwen2.5-1.5B + GIT | 59.0% | 1.5B | Caption pipeline |
| UnifiedQA (fine-tuned) | 57.2% | 60M | Generative QA |
| BERT (fine-tuned) | 42.8% | 110M | Text classifier |
| LLaVA-v1.6-Mistral-7B | 40.2% | 7.3B | Vision-Language |
| BLIP-VQA | 2.7% | 220M | Vision-Language |

### Custom Hybrid Architectures

| Model | Overall Acc | Parameters | Components |
|-------|-------------|------------|------------|
| hybrid_v2 | 67.4% | 141M | LSTM + Cross-Attn + Attention Pooling |
| cross_attn | 65.4% | 138M | Cross-Attention |
| baseline_cls | 65.6% | 135M | CLS Pooling |
| lstm_attention | 64.6% | 139M | BiLSTM + Attention Pooling |
| baseline_attention | 62.0% | 135M | Attention Pooling |

### Instruction-Tuned Chain-of-Thought

| Model | Overall Acc | Parameters | Method |
|-------|-------------|------------|--------|
| UnifiedQA CoT (ViT captioning) | 48.2% | 223M | Direct CoT supervision |

### Decoupled Instruction Tuning

| Model | Overall Acc | Parameters | Method |
|-------|-------------|------------|--------|
| Qwen3-8B (LoRA) + BLIP | 78.0% | 8B (LoRA: ~32M) | Caption + LoRA + Structured CoT |

---

## Detailed Results

### Overall Performance by Subject and Context

| Model | Overall | NAT | SOC | LAN | TXT | IMG | TXT+IMG | G1-6 | G7-12 |
|-------|---------|-----|-----|-----|-----|-----|---------|------|-------|
| **Qwen3-8B (LoRA) + BLIP** | **78.0** | 76.9 | 74.4 | 84.1 | 87.1 | 79.2 | 67.3 | 82.0 | 69.2 |
| hybrid_v2 | 67.4 | 70.2 | 65.1 | 63.6 | 79.6 | 61.1 | 66.7 | 67.3 | 67.6 |
| Qwen2.5-1.5B + GIT | 59.0 | 60.6 | 33.3 | 69.6 | 93.3 | 22.7 | 38.4 | 57.8 | 62.1 |
| UnifiedQA (fine-tuned) | 57.2 | 56.3 | 58.7 | 58.0 | 63.9 | — | — | 59.9 | 52.4 |
| UnifiedQA CoT | 48.2 | 41.6 | 40.4 | 36.9 | 47.7 | 37.8 | 40.1 | 45.5 | 47.5 |
| BERT | 42.8 | 43.9 | 32.6 | 48.6 | 42.8 | — | — | 43.6 | 41.2 |
| LLaVA-v1.6-Mistral-7B | 40.2 | 41.7 | 26.4 | 48.4 | 43.9 | 25.9 | 41.2 | 40.3 | 40.0 |

**Legend:**
- NAT: Natural Science | SOC: Social Science | LAN: Language Science
- TXT: Text-only | IMG: Image-only | TXT+IMG: Both | G1-6: Elementary | G7-12: Middle/High School

---

## Training Configuration

### Baseline Models

| Model | Batch Size | Learning Rate | Epochs | Time |
|-------|------------|---------------|--------|------|
| unifiedqa-t5-small | 8 | 5×10⁻⁵ | 3 | 1.5h |
| bert-base-uncased | 4 | 5×10⁻⁵ | 3 | 1.5h |
| Qwen2.5-1.5B-Instruct | 4 | 5×10⁻⁵ | 3 | 5.0h |
| llava-v1.6-mistral-7b | 1 | 5×10⁻⁵ | 3 | 12.0h |
| git-base-textvqa | 8 | 5×10⁻⁵ | 3 | 3.0h |

### Instruction-Tuned CoT

| Model | Batch Size | Learning Rate | Epochs | Time |
|-------|------------|---------------|--------|------|
| UnifiedQA CoT | 4 | 5×10⁻⁵ | 5 | 4.0h |

### Decoupled Instruction Tuning

| Model | Batch Size | Learning Rate | Epochs | Time |
|-------|------------|---------------|--------|------|
| Qwen3-8B (LoRA, r=32) | 128 | 3×10⁻⁵ | 5 | 3.0h |

### Custom Hybrid Architectures

| Model | Batch Size | Learning Rate | Epochs | Time |
|-------|------------|---------------|--------|------|
| baseline_cls | 32 | 2×10⁻⁵ | 10 | 1.0h |
| baseline_attention | 32 | 2×10⁻⁵ | 10 | 1.0h |
| lstm_attention | 32 | 2×10⁻⁵ | 10 | 1.0h |
| cross_attn | 32 | 2×10⁻⁵ | 10 | 1.0h |
| hybrid | 32 | 2×10⁻⁵ | 10 | 1.0h |
| hybrid_v2 | 32 | 2×10⁻⁵ | 15 | 1.5h |

---

## Installation

```bash
git clone https://github.com/Sreeram-Ragha/NLP-Final-Project-COMS4705.git
cd NLP-Final-Project-COMS4705

```


## Acknowledgments

- **Course:** COMS4705 Natural Language Processing, Columbia University
- **TA Mentor:** Chaitya Shah
- **Dataset:** ScienceQA ([Lu et al., 2022](https://arxiv.org/abs/2209.09513))

---

## Contact

- Reneth Raj Simon - rs4761@columbia.edu
- Sreeram Raghammudi - sr4314@columbia.edu  
- Vishal Menon - vm2820@columbia.edu
