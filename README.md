# BANKING77 Intent Classification (Unsloth + Qwen2.5)

## 1. Overview

This project fine-tunes `Qwen2.5-0.5B-Instruct` for BANKING77 intent classification.

Main pipeline:

1. Download the original BANKING77 data.
2. Clean and split data into train/validation/test.
3. Fine-tune with Unsloth + LoRA.
4. Run single-message inference and return one intent label.

## 2. Project Structure

```text
.
|-- banking77_data/
|   |-- category.json
|   |-- train.csv
|   |-- test.csv
|   |-- train.json
|   `-- test.json
|-- configs/
|   |-- train.yaml
|   `-- inference.yaml
|-- outputs/
|   `-- intent_qwen05b/
|-- sample_data/
|   |-- train.csv
|   |-- val.csv
|   |-- test.csv
|   `-- cleaning_log.txt
|-- scripts/
|   |-- 1_download_data.py
|   |-- 2_preprocess_data.py
|   |-- 3_eda_data.ipynb
|   |-- train.py
|   `-- inference.py
|-- requirements.txt
`-- README.md
```

## 3. System Requirements

1. Python 3.10.
2. NVIDIA GPU + CUDA driver (recommended for both training and inference with Unsloth).
3. Windows, Linux, or WSL.

Notes:

1. `scripts/train.py` requires CUDA GPU.
2. `scripts/inference.py` is currently implemented with Unsloth backend (GPU-oriented).

## 4. Environment Setup

### 4.1 Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 4.2 Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4.3 Install CUDA PyTorch (Important for GPU)

For Windows with NVIDIA GPU:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade torch==2.10.0+cu126 torchvision==0.25.0+cu126 torchaudio==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

Quick GPU check:

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## 5. Data Preparation

### Step 1: Download Original BANKING77 Data

```powershell
python scripts/1_download_data.py
```

Outputs:

1. `banking77_data/train.csv`
2. `banking77_data/test.csv`
3. `banking77_data/category.json`
4. `banking77_data/train.json`
5. `banking77_data/test.json`

### Step 2: Clean and Split Data

```powershell
python scripts/2_preprocess_data.py
```

Outputs:

1. `sample_data/train.csv`
2. `sample_data/val.csv`
3. `sample_data/test.csv`
4. `sample_data/cleaning_log.txt`

Default split ratio:

1. Train: 80%
2. Validation: 10%
3. Test: 10%

## 6. Training

Run training with default config:

```powershell
python scripts/train.py --config configs/train.yaml
```

Key settings in `configs/train.yaml`:

1. Base model: `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit`
2. Max sequence length: `256`
3. LoRA rank `r`: `16`
4. Epochs: `2`
5. Learning rate: `2e-4`

Training outputs in `outputs/intent_qwen05b`:

1. `adapter_model.safetensors`
2. `adapter_config.json`
3. `tokenizer.json`
4. `tokenizer_config.json`
5. `label_mapping.json`
6. `test_metrics.json`

## 7. Inference

### 7.1 Single Message from Command Line

```powershell
python scripts/inference.py --config configs/inference.yaml --message "I was charged extra or hidden fee on my purchase"
```

### 7.2 Interactive Mode

```powershell
python scripts/inference.py
```

Then type a message in terminal. The script returns exactly one label.

## 8. Troubleshooting (Windows + GPU)

### Error 1: `NotImplementedError: Unsloth cannot find any torch accelerator`

Cause:

1. CPU-only torch is installed (`torch+cpu`) instead of CUDA torch (`torch+cuXXX`).

Fix:

1. Reinstall CUDA torch as shown in section 4.3.
2. Confirm `torch.cuda.is_available()` is `True`.

### Error 2: `include file 'tccdefs.h' not found` during inference/training on Windows

Common cause:

1. Triton TinyCC compiles kernels through Unicode/UNC paths.
2. Project path contains non-ASCII characters.

Recommended fixes:

1. Move the project to an ASCII-only path, for example `C:\NLP-INDUSTRY-Lab2`.
2. Use Visual Studio Build Tools (MSVC) to reduce TinyCC issues on Windows.
3. If you recreate `venv`, re-check the full GPU stack.

Note:

1. The current local environment was patched to bypass TinyCC path issues.
2. Any local patch inside `venv` can be lost after recreating the environment.

## 9. Config Quick Guide

`configs/train.yaml`:

1. Set data paths, hyperparameters, and prompt template.
2. Set `output_dir` for saved checkpoints.

`configs/inference.yaml`:

1. Set `checkpoint_dir` to your trained output.
2. Tune generation and postprocessing (`fuzzy_cutoff`).

## 10. Quick Commands (Copy/Paste)

```powershell
# 1) Activate environment
.\venv\Scripts\activate

# 2) Download and preprocess data
python scripts/1_download_data.py
python scripts/2_preprocess_data.py

# 3) Train
python scripts/train.py --config configs/train.yaml

# 4) Inference
python scripts/inference.py --config configs/inference.yaml --message "Where is my refund?"
```
