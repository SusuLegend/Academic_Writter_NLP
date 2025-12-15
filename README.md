# Academic LoRA Model for Academic Writing

This project provides a complete pipeline for training, fine-tuning, evaluating, and deploying a LoRA-adapted language model for academic writing style. It includes tools for dataset preparation, model training, evaluation, and a Gradio-based user interface for interactive text generation.

## Project Structure

```
NLP/
├── demo.ipynb                        # Gradio UI and model inference demo
├── model_evaluation_and_finetuning.ipynb # Training, evaluation, and style control
├── requirements.txt                   # Python dependencies
├── academic_LoRa_model/
│   ├── corpus.txt                     # Academic writing corpus
│   ├── dataset_utils.py               # Dataset preparation utilities
│   ├── evaluation.ipynb               # Model evaluation notebook
│   ├── finetuning.ipynb               # LoRA fine-tuning notebook
│   ├── model_evaluation_scores.csv    # Evaluation results
│   ├── model_utils.py                 # Model loading and utility functions
│   ├── style_controller.ipynb         # Style control and analysis
│   ├── token_freq.json                # Token frequency statistics
│   └── lora_academic_model/
│       ├── adapter_config.json        # LoRA adapter config
│       ├── adapter_model.safetensors  # LoRA adapter weights
│       ├── merges.txt                 # Tokenizer merges
│       ├── special_tokens_map.json    # Tokenizer special tokens
│       ├── tokenizer_config.json      # Tokenizer config
│       ├── tokenizer.json             # Tokenizer
│       ├── vocab.json                 # Tokenizer vocab
│       └── checkpoint-170/            # Model checkpoint (latest)
│           └── ...                    # All model and optimizer files
└── .gitignore, .python-version, README.md
```

## Features

- **LoRA Adapter Training**: Efficient fine-tuning of large language models for academic style using LoRA adapters.
- **Dataset Preparation**: Utilities for processing and analyzing academic corpora.
- **Evaluation & Style Control**: Notebooks for evaluating model outputs and controlling writing style.
- **Interactive UI**: Gradio-based notebook for generating academic text continuations.
- **Reproducibility**: All code and checkpoints are versioned for reproducible experiments.

## Quickstart

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Model Inference Demo

Open `demo.ipynb` in Jupyter and run all cells. This notebook loads the LoRA-adapted model and launches a Gradio UI for interactive text generation.

### 3. Training & Evaluation

- **Fine-tuning**: Use `academic_LoRa_model/finetuning.ipynb` to train or continue training the LoRA adapter on your academic corpus.
- **Evaluation**: Use `academic_LoRa_model/evaluation.ipynb` and `model_evaluation_and_finetuning.ipynb` for model evaluation, style control, and scoring.

### 4. Model Checkpoints & Adapters

The latest LoRA adapter and tokenizer files are in `academic_LoRa_model/lora_academic_model/` and its subfolders (e.g., `checkpoint-170/`).

## Usage Example (Python)

```python
from academic_LoRa_model.model_utils import load_lora_model

# Load the LoRA-adapted model
model, tokenizer = load_lora_model(
	base_model_name="facebook/opt-1.3b",  # or your base model
	adapter_dir="academic_LoRa_model/lora_academic_model/checkpoint-170"
)

# Generate academic text
input_text = "In recent years, the field of machine learning has"
output = model.generate(tokenizer(input_text, return_tensors="pt").input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Gradio UI

Run all cells in `demo.ipynb` to launch a web UI for interactive academic text generation. You can input a prompt and receive a model-generated academic continuation.

## Notebooks Overview

- **demo.ipynb**: Gradio UI, model loading, and inference
- **model_evaluation_and_finetuning.ipynb**: LoRA training, evaluation, and style control
- **academic_LoRa_model/evaluation.ipynb**: Model evaluation and scoring
- **academic_LoRa_model/finetuning.ipynb**: LoRA adapter fine-tuning
- **academic_LoRa_model/style_controller.ipynb**: Style analysis and control

## File/Folder Descriptions

- `academic_LoRa_model/`: Main code, corpus, and utilities for LoRA training and evaluation
- `lora_academic_model/`: Contains LoRA adapter weights, tokenizer, and checkpoints
- `model_utils.py`: Functions for loading and using the LoRA-adapted model
- `dataset_utils.py`: Dataset loading and preprocessing utilities
- `token_freq.json`: Token frequency statistics for analysis

## Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- peft (LoRA)
- Gradio
- Jupyter
- (See `requirements.txt` for full list)

## Troubleshooting

- **CUDA/Memory Issues**: Use a GPU with sufficient memory (16GB+ recommended for OPT-1.3B).
- **Model Loading Errors**: Ensure all adapter and tokenizer files are present in `lora_academic_model/` and checkpoint folders.
- **Gradio UI Not Launching**: Check Python and Gradio installation, and run all cells in `demo.ipynb`.

## Citation

If you use this project, please cite the repository or relevant papers for LoRA and Hugging Face Transformers.

---
For questions or contributions, please open an issue or pull request.
