# ModelShield

**Code and datasets for our paper: [ModelShield: Adaptive and Robust Watermark Against Model Extraction Attacks](https://arxiv.org/abs/2405.02365)**

---

## Overview

ModelShield provides a framework for generating, embedding, and verifying watermarks in language models to protect intellectual property against model extraction attacks. This repository includes code for:
- Watermark generation and verification.
- Training imitation models with watermarked data and Generating imitation model’s output.
- Supporting datasets for experimentation.

---

## Dependencies

The environment setup is required only during the model training phase. Please refer to the [requirements file](https://github.com/amaoku/ModelShield/blob/master/Imitation_Model_training/train/requirements.txt) for the necessary dependencies.

---

## Stages

### 1. **Watermark Generation**
We utilize system-level instructions to guide watermark generation in language models, ensuring seamless integration and high robustness. (need API-KEY for different LMaaS)

### 2. **Imitation Model Training**
Fine-tune imitation models with watermarked data to simulate model extraction attacks. 

We base our training and fine-tuning on the [BELLE GitHub project](https://github.com/LianjiaTech/BELLE). Key features include:
- **Full fine-tuning** and **LoRA fine-tuning** support.
- Flexibility to integrate your own fine-tuning methods.
- Configuration options available in the `config` directory (e.g., model base, fine-tuning epochs, batch size, learning rate, and LoRA usage).

### 3. **Watermark Verification**
We provide two methods for verifying embedded watermarks:
1. **Rapid Verification**: Quickly detect the presence of a watermark based on texts.
2. **Detailed Verification**: Need comparison with legimate model and base model.
   
---

## Datasets

The following datasets are used in our experiments:
- **HC3**: A dataset for language model imitation analysis.
- **WILD**: A dataset for evaluating robustness in diverse scenarios.

---
## Usage
1. 1 Generate watermarked data from the victim model (See readme in Watermark Generation)
2. 2 Simulate the model extraction attack (See readme in Imitation Model training)
3. 3 Verify the watermark (See readme in Watermark Verification)

---
## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{modelshield,
  title={Adaptive and robust watermark against model extraction attack},
  author={Pang, Kaiyi and Qi, Tao and Wu, Chuhan and Bai, Minhao and Jiang, Minghu and Huang, Yongfeng},
  journal={arXiv preprint arXiv:2405.02365},
  year={2024}
}
