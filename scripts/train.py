from __future__ import annotations

import argparse
import difflib
import inspect
import json
import random
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTConfig, SFTTrainer


def suppress_known_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
        category=FutureWarning,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BANKING77 intent model with Unsloth")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config file",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(project_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    required_keys = ["data", "model", "training", "evaluation", "prompt"]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise KeyError(f"Missing config sections: {missing}")

    return config


def load_split(csv_path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in (text_col, label_col) if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {missing}")

    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[(df[text_col] != "") & (df[label_col] != "")].reset_index(drop=True)
    return df


def build_prompt(message: str, config: dict[str, Any], label: str | None = None) -> str:
    system = str(config["prompt"].get("system", "You are an intent classifier."))
    input_prefix = str(config["prompt"].get("input_prefix", "Message:"))
    output_prefix = str(config["prompt"].get("output_prefix", "Label:"))

    message = re.sub(r"\s+", " ", message).strip()
    prompt = f"{system}\n{input_prefix} {message}\n{output_prefix}"
    if label is not None:
        return f"{prompt} {label}"
    return prompt


def build_sft_dataset(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    config: dict[str, Any],
) -> Dataset:
    formatted = [
        build_prompt(message=row[text_col], config=config, label=row[label_col])
        for _, row in df.iterrows()
    ]
    return Dataset.from_dict({"text": formatted})


def normalize_label(label: str) -> str:
    label = label.strip().lower()
    label = re.sub(r"[\s\-/]+", "_", label)
    label = re.sub(r"[^a-z0-9_]", "", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label


def canonicalize_prediction(prediction: str, canonical_labels: dict[str, str]) -> str:
    first_line = prediction.strip().splitlines()[0] if prediction.strip() else ""
    first_line = first_line.strip(" \t\n\r\"'`.,:;!?()[]{}")

    candidates = [first_line]
    if ":" in first_line:
        candidates.append(first_line.split(":", 1)[-1].strip())
    candidates.extend(re.findall(r"[A-Za-z0-9_\-/?]+", first_line))

    for candidate in candidates:
        normalized = normalize_label(candidate)
        if not normalized:
            continue

        if normalized in canonical_labels:
            return canonical_labels[normalized]

        containment = [
            key for key in canonical_labels if key in normalized or normalized in key
        ]
        if containment:
            best_key = max(
                containment,
                key=lambda key: (difflib.SequenceMatcher(a=normalized, b=key).ratio(), len(key)),
            )
            return canonical_labels[best_key]

        close = difflib.get_close_matches(normalized, list(canonical_labels.keys()), n=1, cutoff=0.72)
        if close:
            return canonical_labels[close[0]]

    normalized_first = normalize_label(first_line)
    if normalized_first and canonical_labels:
        best_key = max(
            canonical_labels,
            key=lambda key: difflib.SequenceMatcher(a=normalized_first, b=key).ratio(),
        )
        return canonical_labels[best_key]

    if canonical_labels:
        return next(iter(canonical_labels.values()))

    return first_line


def predict_label(
    model,
    tokenizer,
    message: str,
    config: dict[str, Any],
    canonical_labels: dict[str, str],
) -> str:
    prompt = build_prompt(message=message, config=config, label=None)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(config["model"].get("max_seq_length", 256)),
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(config["evaluation"].get("max_new_tokens", 16)),
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()
    return canonicalize_prediction(generated, canonical_labels)


def clear_generation_max_length(model) -> None:
    # Keep only max_new_tokens control to avoid noisy max_length precedence warnings.
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and getattr(generation_config, "max_length", None) is not None:
        generation_config.max_length = None


def evaluate(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    max_samples = int(config["evaluation"].get("max_samples", -1))
    eval_df = test_df if max_samples <= 0 else test_df.head(max_samples).copy()

    labels = sorted(test_df[label_col].unique().tolist())
    canonical = {normalize_label(label): label for label in labels}

    y_true: list[str] = []
    y_pred: list[str] = []

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating"):
        true_label = row[label_col]
        pred_label = predict_label(
            model=model,
            tokenizer=tokenizer,
            message=row[text_col],
            config=config,
            canonical_labels=canonical,
        )
        y_true.append(true_label)
        y_pred.append(pred_label)

    accuracy = accuracy_score(y_true, y_pred)
    metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_eval_samples": len(eval_df),
        "accuracy": float(accuracy),
    }

    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Saved test metrics to: {metrics_path}")


def main() -> None:
    suppress_known_warnings()
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    config_path = resolve_path(project_dir, args.config)
    config = load_config(config_path)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Unsloth fine-tuning. Use Colab/Kaggle/local GPU.")

    data_cfg = config["data"]
    text_col = str(data_cfg.get("text_column", "text"))
    label_col = str(data_cfg.get("label_column", "category"))

    train_df = load_split(resolve_path(project_dir, data_cfg["train_csv"]), text_col, label_col)
    val_df = load_split(resolve_path(project_dir, data_cfg["val_csv"]), text_col, label_col)
    test_df = load_split(resolve_path(project_dir, data_cfg["test_csv"]), text_col, label_col)

    label_list = sorted(set(train_df[label_col]) | set(val_df[label_col]) | set(test_df[label_col]))

    train_dataset = build_sft_dataset(train_df, text_col, label_col, config)
    val_dataset = build_sft_dataset(val_df, text_col, label_col, config)

    model_cfg = config["model"]
    training_cfg = config["training"]

    max_seq_length = int(model_cfg.get("max_seq_length", 256))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_cfg["base_model"]),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(model_cfg.get("lora_r", 16)),
        target_modules=list(model_cfg.get("target_modules", [])),
        lora_alpha=int(model_cfg.get("lora_alpha", 16)),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.0)),
        bias=str(model_cfg.get("bias", "none")),
        use_gradient_checkpointing=model_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=seed,
        use_rslora=bool(model_cfg.get("use_rslora", False)),
        loftq_config=None,
    )

    output_dir = resolve_path(project_dir, training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_strategy = str(
        training_cfg.get("eval_strategy", training_cfg.get("evaluation_strategy", "epoch"))
    )
    training_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 8)),
        "per_device_eval_batch_size": int(training_cfg.get("per_device_eval_batch_size", 8)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 4)),
        "learning_rate": float(training_cfg.get("learning_rate", 2e-4)),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 2)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.03)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.01)),
        "logging_steps": int(training_cfg.get("logging_steps", 20)),
        "save_strategy": str(training_cfg.get("save_strategy", "epoch")),
        "optim": str(training_cfg.get("optim", "adamw_8bit")),
        "lr_scheduler_type": str(training_cfg.get("lr_scheduler_type", "linear")),
        "max_grad_norm": float(training_cfg.get("max_grad_norm", 1.0)),
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "report_to": str(training_cfg.get("report_to", "none")),
        "seed": seed,
    }

    training_arg_params = inspect.signature(SFTConfig.__init__).parameters
    if "eval_strategy" in training_arg_params:
        training_kwargs["eval_strategy"] = eval_strategy
    else:
        training_kwargs["evaluation_strategy"] = eval_strategy

    if "dataset_text_field" in training_arg_params:
        training_kwargs["dataset_text_field"] = "text"
    if "max_length" in training_arg_params:
        training_kwargs["max_length"] = max_seq_length
    elif "max_seq_length" in training_arg_params:
        training_kwargs["max_seq_length"] = max_seq_length
    if "packing" in training_arg_params:
        training_kwargs["packing"] = bool(model_cfg.get("packing", False))

    training_args = SFTConfig(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "args": training_args,
    }

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = max_seq_length
    if "packing" in trainer_params:
        trainer_kwargs["packing"] = bool(model_cfg.get("packing", False))

    trainer = SFTTrainer(
        **trainer_kwargs,
    )

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Unique labels: {len(label_list)}")

    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    label_mapping = {
        "label2id": {label: idx for idx, label in enumerate(label_list)},
        "id2label": {str(idx): label for idx, label in enumerate(label_list)},
    }

    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)

    with open(output_dir / "train_config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)

    if bool(config["evaluation"].get("enabled", True)):
        FastLanguageModel.for_inference(model)
        clear_generation_max_length(model)
        evaluate(
            model=model,
            tokenizer=tokenizer,
            test_df=test_df,
            text_col=text_col,
            label_col=label_col,
            config=config,
            output_dir=output_dir,
        )

    print(f"Saved checkpoint and artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
