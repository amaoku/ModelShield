
# ModelShield

Code and datasets for our paper on language model IP protection watermark：ModelShield: Adaptive and Robust Watermark against Model Extraction Attack

## Dependencies
Only the model training phase needs to take the environment into consideration. [ Requirement](https://github.com/amaoku/ModelShield/blob/master/Imitation_Model_training/train/requirements.txt)

## Watermark Generation
We use system instructions to guide the generation of watermarks in language models.

## Imitation Model training
fine-tuning imitation model with watermarked data. We base our model training and fine-tuning on the [GitHub project](https://github.com/LianjiaTech/BELLE). Both full fine-tuning and LoRA fine-tuning are supported, and you can also choose your own fine-tuning method.

## Watermark Verification
We offer two methods for watermark verification:
1. Rapid Verification
2. Detailed Verification

## Dataset
- HC3
- WILD
