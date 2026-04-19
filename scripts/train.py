from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


PROJECT_DIR = Path(__file__).resolve().parent.parent


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def clean_dataframe(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    cleaned = df.loc[:, [text_col, label_col]].copy()
    cleaned[text_col] = cleaned[text_col].astype("string").str.strip().str.replace('"', "", regex=False)
    cleaned[label_col] = cleaned[label_col].astype("string").str.strip().str.replace('"', "", regex=False)
    cleaned = cleaned.dropna(subset=[text_col, label_col])
    cleaned = cleaned[(cleaned[text_col] != "") & (cleaned[label_col] != "")]
    return cleaned.reset_index(drop=True)


def load_dataset_csv(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    df = pd.read_csv(path)
    required = {text_col, label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    return clean_dataframe(df, text_col, label_col)


def resolve_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None:
        return None

    value = dtype_name.strip().lower()
    if value in {"auto", "none", ""}:
        return None
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32

    raise ValueError(f"Unsupported dtype value: {dtype_name}")


def build_texts(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    template: str,
    eos_token: str,
) -> Dataset:
    def _format_batch(batch: dict[str, list[str]]) -> dict[str, list[str]]:
        texts: list[str] = []
        for message, label in zip(batch[text_col], batch[label_col]):
            texts.append(template.format(message=message, label=label).strip() + eos_token)
        return {"text": texts}

    dataset = Dataset.from_pandas(df[[text_col, label_col]], preserve_index=False)
    return dataset.map(_format_batch, batched=True, remove_columns=[text_col, label_col])


def score_label_candidates(
    model,
    tokenizer,
    prompts: list[str],
    label_candidates: list[str],
    label_chunk_size: int,
    device: torch.device,
) -> list[str]:
    if not prompts:
        return []

    prompt_batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    prompt_batch = {k: v.to(device) for k, v in prompt_batch.items()}
    prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
    prompt_rows = [
        prompt_batch["input_ids"][i, : prompt_lengths[i]].tolist()
        for i in range(len(prompts))
    ]

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer is missing eos_token_id.")

    label_token_rows = [
        tokenizer(label, add_special_tokens=False)["input_ids"] + [eos_id]
        for label in label_candidates
    ]
    scores = torch.empty((len(prompts), len(label_candidates)), device=device)

    for label_start in range(0, len(label_candidates), label_chunk_size):
        chunk_labels = label_token_rows[label_start : label_start + label_chunk_size]
        sequences: list[dict[str, list[int]]] = []
        seq_prompt_lengths: list[int] = []
        seq_lengths: list[int] = []

        for prompt_tokens, prompt_len in zip(prompt_rows, prompt_lengths):
            for label_tokens in chunk_labels:
                seq = prompt_tokens + label_tokens
                sequences.append({"input_ids": seq})
                seq_prompt_lengths.append(prompt_len)
                seq_lengths.append(len(seq))

        batch = tokenizer.pad(sequences, padding=True, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.inference_mode():
            logits = model(**batch).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = batch["input_ids"][:, 1:].unsqueeze(-1)
            token_scores = log_probs.gather(-1, target_ids).squeeze(-1)

            mask = torch.zeros_like(token_scores, dtype=torch.bool)
            for row, (prompt_len, seq_len) in enumerate(zip(seq_prompt_lengths, seq_lengths)):
                mask[row, prompt_len - 1 : seq_len - 1] = True

            seq_scores = (token_scores * mask).sum(dim=1)

        scores[:, label_start : label_start + len(chunk_labels)] = seq_scores.view(len(prompts), -1)

    best_indices = scores.argmax(dim=1).tolist()
    return [label_candidates[idx] for idx in best_indices]


def evaluate_csv(
    model,
    tokenizer,
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    inference_template: str,
    label_candidates: list[str],
    batch_size: int,
    label_chunk_size: int,
    device: torch.device,
) -> dict[str, Any]:
    messages = df[text_col].tolist()
    true_labels = df[label_col].tolist()
    predicted_labels: list[str] = []

    for start in range(0, len(messages), batch_size):
        batch_messages = messages[start : start + batch_size]
        prompts = [inference_template.format(message=message, label="").strip() for message in batch_messages]
        batch_predictions = score_label_candidates(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            label_candidates=label_candidates,
            label_chunk_size=label_chunk_size,
            device=device,
        )
        predicted_labels.extend(batch_predictions)

    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, zero_division=0)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "num_examples": len(df),
    }


def main() -> None:
    config_path = resolve_path(sys.argv[1]) if len(sys.argv) > 1 else resolve_path("configs/train.yaml")
    config = load_yaml(config_path)

    paths_cfg = config["paths"]
    data_cfg = config["data"]
    prompt_cfg = config["prompt"]
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    training_cfg = config["training"]

    train_path = resolve_path(paths_cfg["train_csv"])
    val_path = resolve_path(paths_cfg["val_csv"])
    test_path = resolve_path(paths_cfg["test_csv"])
    output_dir = resolve_path(paths_cfg["output_dir"])
    checkpoint_dir = resolve_path(paths_cfg["checkpoint_dir"])
    metadata_path = resolve_path(paths_cfg["metadata_file"])
    metrics_path = resolve_path(paths_cfg["metrics_file"])

    text_col = data_cfg["text_column"]
    label_col = data_cfg["label_column"]

    train_df = load_dataset_csv(train_path, text_col, label_col)
    if val_path.exists():
        val_df = load_dataset_csv(val_path, text_col, label_col)
    else:
        val_df = pd.DataFrame(columns=[text_col, label_col])
    test_df = load_dataset_csv(test_path, text_col, label_col)

    label_candidates = sorted(train_df[label_col].astype(str).unique().tolist())

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires a CUDA-capable GPU for Unsloth.")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dtype = resolve_dtype(model_cfg.get("dtype"))
    load_in_4bit = bool(model_cfg.get("load_in_4bit", True))
    max_seq_length = int(model_cfg["max_seq_length"])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg["r"]),
        target_modules=list(lora_cfg["target_modules"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        bias=str(lora_cfg["bias"]),
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=int(lora_cfg["random_state"]),
        use_rslora=bool(lora_cfg["use_rslora"]),
        loftq_config=lora_cfg.get("loftq_config"),
    )

    eos_token = tokenizer.eos_token or ""
    train_dataset = build_texts(
        df=train_df,
        text_col=text_col,
        label_col=label_col,
        template=prompt_cfg["train_template"],
        eos_token=eos_token,
    )
    eval_dataset = build_texts(
        df=val_df if len(val_df) else test_df,
        text_col=text_col,
        label_col=label_col,
        template=prompt_cfg["train_template"],
        eos_token=eos_token,
    )

    fp16_cfg = training_cfg.get("fp16", "auto")
    bf16_cfg = training_cfg.get("bf16", "auto")
    use_bf16 = torch.cuda.is_bf16_supported() if str(bf16_cfg).lower() == "auto" else bool(bf16_cfg)
    use_fp16 = (not use_bf16) if str(fp16_cfg).lower() == "auto" else bool(fp16_cfg)

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(training_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training_cfg["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(training_cfg["gradient_accumulation_steps"]),
        "warmup_steps": int(training_cfg["warmup_steps"]),
        "learning_rate": float(training_cfg["learning_rate"]),
        "weight_decay": float(training_cfg["weight_decay"]),
        "lr_scheduler_type": str(training_cfg["lr_scheduler_type"]),
        "optim": str(training_cfg["optimizer"]),
        "logging_steps": int(training_cfg["logging_steps"]),
        "evaluation_strategy": str(training_cfg["evaluation_strategy"]),
        "save_strategy": str(training_cfg["save_strategy"]),
        "save_steps": int(training_cfg["save_steps"]),
        "eval_steps": int(training_cfg["eval_steps"]),
        "save_total_limit": int(training_cfg["save_total_limit"]),
        "load_best_model_at_end": bool(training_cfg["load_best_model_at_end"]),
        "metric_for_best_model": str(training_cfg["metric_for_best_model"]),
        "greater_is_better": bool(training_cfg["greater_is_better"]),
        "report_to": training_cfg.get("report_to", "none"),
        "seed": int(training_cfg["seed"]),
        "fp16": use_fp16,
        "bf16": use_bf16,
    }

    max_steps = int(training_cfg.get("max_steps", -1))
    if max_steps > 0:
        training_kwargs["max_steps"] = max_steps
    else:
        training_kwargs["num_train_epochs"] = float(training_cfg.get("num_train_epochs", 1.0))

    training_args = TrainingArguments(**training_kwargs)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=int(training_cfg["dataset_num_proc"]),
        packing=bool(training_cfg.get("packing", False)),
        args=training_args,
    )

    print("Starting training...")
    train_result = trainer.train()
    print(train_result)

    print("Saving checkpoint...")
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    metadata = {
        "model_name": model_cfg["name"],
        "checkpoint_dir": str(checkpoint_dir),
        "text_column": text_col,
        "label_column": label_col,
        "label_candidates": label_candidates,
        "train_template": prompt_cfg["train_template"],
        "inference_template": prompt_cfg["inference_template"],
        "max_seq_length": max_seq_length,
        "load_in_4bit": load_in_4bit,
        "lora": {
            "r": int(lora_cfg["r"]),
            "target_modules": list(lora_cfg["target_modules"]),
            "lora_alpha": int(lora_cfg["lora_alpha"]),
            "lora_dropout": float(lora_cfg["lora_dropout"]),
            "bias": str(lora_cfg["bias"]),
        },
    }
    save_json(metadata_path, metadata)

    FastLanguageModel.for_inference(model)
    test_metrics = evaluate_csv(
        model=model,
        tokenizer=tokenizer,
        df=test_df,
        text_col=text_col,
        label_col=label_col,
        inference_template=prompt_cfg["inference_template"],
        label_candidates=label_candidates,
        batch_size=int(training_cfg["per_device_eval_batch_size"]),
        label_chunk_size=int(prompt_cfg["label_chunk_size"]),
        device=torch.device("cuda"),
    )
    save_json(metrics_path, test_metrics)

    print("\nTest accuracy:", f"{test_metrics['accuracy'] * 100:.2f}%")
    print("\nClassification report:")
    print(test_metrics["classification_report"])
    print(f"\nCheckpoint saved to: {checkpoint_dir}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Test metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
