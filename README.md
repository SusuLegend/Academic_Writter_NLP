# Academic Writing LoRA Model

A fine-tuned language model using Low-Rank Adaptation (LoRA) for generating academic-style text continuations. This project demonstrates model evaluation, fine-tuning, and deployment using modern NLP techniques.

## 🎯 Overview

This repository contains:
- **Fine-tuned LoRA Model**: Adapted for academic writing style and formal text generation
- **Interactive Demo**: Gradio web interface for testing the model
- **Training Pipeline**: Complete evaluation and fine-tuning workflow
- **Model Evaluation**: Statistical analysis and performance metrics

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Training](#training)
- [Requirements](#requirements)

## ✨ Features

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using PEFT
- **Academic Text Generation**: Specialized in formal academic writing style
- **Interactive Demo**: User-friendly Gradio interface
- **GPU Support**: Automatic CUDA detection with CPU fallback
- **Model Evaluation**: Comprehensive evaluation metrics and statistical analysis
- **Modular Design**: Separate notebooks for training and deployment

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster inference)

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd NLP
```

2. Install dependencies:
```bash
pip install gradio transformers accelerate peft torch
```

3. (Optional) Upgrade transformers for latest features:
```bash
pip install --upgrade transformers
```

## 🎬 Quick Start

### Launch the Demo

1. Open `demo.ipynb` in Jupyter or VS Code
2. Run all cells sequentially
3. The Gradio interface will launch automatically
4. Access the web UI at the provided local URL (typically `http://127.0.0.1:7860`)

### Example Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load model
model_name = "./lora_academic_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model = PeftModel.from_pretrained(base_model, model_name).to(device)

# Generate text
prompt = "The main objective of this research is to"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📖 Usage

### Demo Interface

The Gradio demo (`demo.ipynb`) provides:
- **Text Input**: Enter your academic writing prompt
- **Generated Output**: View model-generated continuations
- **Example Prompts**: Pre-configured academic writing examples
- **Real-time Inference**: Instant text generation

Example prompts:
- "The main objective of this study is to"
- "Previous research has shown that"
- "In conclusion, the results demonstrate"

### Customization

Adjust generation parameters in the code:
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=150,      # Length of generated text
    temperature=0.7,         # Creativity (0.1-1.0)
    top_p=0.9,              # Nucleus sampling
    do_sample=True          # Enable sampling
)
```

## 🧠 Model Details

### Architecture

- **Base Model**: GPT-2 / GPT-Neo / OPT (as configured in training)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Adapter Rank**: Configured in `adapter_config.json`
- **Target Modules**: Query and Value projection layers
- **Task**: Causal Language Modeling (next-token prediction)

### LoRA Configuration

```json
{
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.1
}
```

### Performance

- **Domain**: Academic writing and formal text
- **Training Data**: Academic papers and scholarly texts
- **Inference Speed**: ~10-50 tokens/second (GPU) | ~2-5 tokens/second (CPU)
- **Model Size**: ~5MB (adapter only) + base model

## 📁 Project Structure

```
NLP/
├── demo.ipynb                              # Interactive Gradio demo
├── model_evaluation_and_finetuning.ipynb  # Training and evaluation pipeline
├── README.md                               # This file
├── .gitignore                             # Git ignore rules
└── lora_academic_model/                   # Fine-tuned LoRA adapter
    ├── adapter_config.json                # LoRA configuration
    ├── adapter_model.safetensors          # Trained adapter weights
    ├── tokenizer_config.json              # Tokenizer settings
    ├── tokenizer.json                     # Tokenizer vocabulary
    ├── vocab.json                         # Vocabulary mappings
    ├── merges.txt                         # BPE merge rules
    ├── special_tokens_map.json            # Special token definitions
    ├── README.md                          # Model card
    └── checkpoint-34/                     # Training checkpoint
        ├── adapter_model.safetensors      # Checkpoint weights
        ├── optimizer.pt                   # Optimizer state
        ├── scheduler.pt                   # LR scheduler state
        ├── trainer_state.json             # Training metrics
        └── rng_state.pth                  # Random state for reproducibility
```

## 🏋️ Training

### Evaluation and Fine-tuning Pipeline

Open `model_evaluation_and_finetuning.ipynb` to:

1. **Model Evaluation**: Compare multiple pre-trained models
   - GPT-2, GPT-Neo, OPT, DialoGPT
   - Perplexity, BLEU, and custom metrics

2. **Data Preparation**: Load and preprocess academic text corpus

3. **LoRA Fine-tuning**: Train adapter layers
   - Configure hyperparameters
   - Monitor training loss
   - Save checkpoints

4. **Statistical Analysis**: 
   - MLE (Maximum Likelihood Estimation)
   - Bootstrap sampling
   - Confidence intervals

### Training Parameters

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,                    # Rank
    lora_alpha=16,          # Scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

## 📦 Requirements

### Core Dependencies

```
transformers>=4.30.0
torch>=2.0.0
peft>=0.4.0
accelerate>=0.20.0
gradio>=3.35.0
```

### Optional (for training)

```
datasets>=2.12.0
evaluate>=0.4.0
scikit-learn>=1.0.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce `max_new_tokens` parameter
- Use CPU mode: `device = "cpu"`
- Close other GPU applications

**Slow Inference**
- Enable GPU if available
- Reduce batch size
- Use smaller `max_new_tokens`

**Module Not Found**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify correct Python environment is active

## 📊 Model Performance

The model has been evaluated on academic writing tasks with the following characteristics:
- **Coherence**: Maintains academic tone and formal structure
- **Fluency**: Generates grammatically correct sentences
- **Relevance**: Stays on-topic with given prompts
- **Style**: Preserves scholarly writing conventions

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Expand training dataset with more academic domains
- Experiment with different base models
- Add more evaluation metrics
- Improve demo UI features

## 📄 License

This project is for educational and research purposes. Please check base model licenses for commercial use restrictions.

## 🙏 Acknowledgments

- **Hugging Face**: Transformers and PEFT libraries
- **Gradio**: Interactive ML demos
- **LoRA**: Parameter-efficient fine-tuning methodology

## 📬 Contact

For questions or feedback, please open an issue in the repository.

---

**Last Updated**: December 2025
