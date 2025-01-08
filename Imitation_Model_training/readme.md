# README

## Overview

This repository is designed to facilitate fine-tuning language models and performing inference to generate outputs embedded with watermarks. The main functionality is organized under the `train` directory, which includes scripts and configurations necessary for both fine-tuning and watermarking-based generation tasks.

## Features

1. **Fine-Tuning Models**  
   The fine-tuning process is highly configurable and can be adjusted to suit various experimental needs. The configurations for fine-tuning are managed through the `config` directory, where you can set parameters such as:
   - **model_name_or_path**: Specify the pre-trained language model to use as the foundation for fine-tuning.
   - **num_epochss**: Define the total training cycles.
   - **batch_size**: Adjust the size of batches for training.
   - **learning_rate**: Customize the learning rate for optimization.
   - **LoRA CONFIGS**: Enable or disable the usage of Low-Rank Adaptation for parameter-efficient fine-tuning (see lora_config_llama.json).

   The output of the fine-tuning process will be a trained model saved in the specified directory.

2. **Generating Watermarked Text**  
   The fine-tuned model can be used for inference to generate text outputs with embedded watermarks. To perform this task:
   - Specify the path to the fine-tuned model.
   - Provide the input `query` dataset for the model to generate text responses.

   The generated outputs will include the desired watermarked content.

## Directory Structure

- **`train/`**: Contains scripts for fine-tuning models and generating watermarked outputs.
- **`configs/`**: Includes configuration files for fine-tuning settings such as model architecture, training parameters, and output specifications.

## How to Use

### Fine-Tuning a Model
1. Navigate to the `configs/` directory and edit the appropriate configuration file:
   - Specify the base model.
   - Adjust training parameters like epochs, batch size, learning rate, and LoRA usage.
2. Run the fine-tuning script from the `train/` directory:
   ```bash
   python train.py --model_config_file/<your_config_file>.json 
