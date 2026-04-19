from __future__ import annotations

import argparse
import difflib
import json
import re
import warnings
from pathlib import Path
from typing import Any

import torch
import yaml
from unsloth import FastLanguageModel


def suppress_known_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
        category=FutureWarning,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BANKING77 intent inference from a fine-tuned checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to inference config file",
    )
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="Input message to classify",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path_value: str, config_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    config_relative = (config_path.parent / path).resolve()
    project_relative = (config_path.parent.parent / path).resolve()

    if config_relative.exists() or not project_relative.exists():
        return config_relative
    return project_relative


def normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    normalized = re.sub(r"[\s\-/]+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def build_prompt(message: str, prompt_cfg: dict[str, Any]) -> str:
    system = str(prompt_cfg.get("system", "You are an intent classifier."))
    input_prefix = str(prompt_cfg.get("input_prefix", "Message:"))
    output_prefix = str(prompt_cfg.get("output_prefix", "Label:"))
    message = re.sub(r"\s+", " ", message).strip()
    return f"{system}\n{input_prefix} {message}\n{output_prefix}"


def canonicalize_prediction(prediction: str, canonical_labels: dict[str, str], fuzzy_cutoff: float) -> str:
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

        close = difflib.get_close_matches(
            normalized,
            list(canonical_labels.keys()),
            n=1,
            cutoff=fuzzy_cutoff,
        )
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


def clear_generation_max_length(model) -> None:
    # Keep only max_new_tokens control to avoid noisy max_length precedence warnings.
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and getattr(generation_config, "max_length", None) is not None:
        generation_config.max_length = None


class IntentClassification:
    def __init__(self, model_path: str):
        self.config_path = Path(model_path).resolve()
        self.config = load_yaml(self.config_path)

        model_cfg = self.config.get("model", {})
        generation_cfg = self.config.get("generation", {})
        postprocess_cfg = self.config.get("postprocess", {})

        if "checkpoint_dir" not in model_cfg:
            raise KeyError("Missing model.checkpoint_dir in inference config")

        self.checkpoint_dir = resolve_path(str(model_cfg["checkpoint_dir"]), self.config_path)
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")

        train_snapshot_value = model_cfg.get("train_config_snapshot", "")
        prompt_cfg = self.config.get("prompt", {})
        if train_snapshot_value:
            train_snapshot_path = resolve_path(str(train_snapshot_value), self.config_path)
            if train_snapshot_path.exists():
                train_snapshot = load_yaml(train_snapshot_path)
                if not prompt_cfg and isinstance(train_snapshot.get("prompt"), dict):
                    prompt_cfg = train_snapshot["prompt"]

        self.prompt_cfg = prompt_cfg
        self.max_seq_length = int(model_cfg.get("max_seq_length", 256))
        self.load_in_4bit = bool(model_cfg.get("load_in_4bit", True))
        self.max_new_tokens = int(generation_cfg.get("max_new_tokens", 12))
        self.do_sample = bool(generation_cfg.get("do_sample", False))
        self.temperature = float(generation_cfg.get("temperature", 0.0))
        self.fuzzy_cutoff = float(postprocess_cfg.get("fuzzy_cutoff", 0.72))

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.checkpoint_dir),
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        FastLanguageModel.for_inference(self.model)
        clear_generation_max_length(self.model)

        label_mapping_value = model_cfg.get("label_mapping_path", "")
        if label_mapping_value:
            label_mapping_path = resolve_path(str(label_mapping_value), self.config_path)
        else:
            label_mapping_path = self.checkpoint_dir / "label_mapping.json"

        if not label_mapping_path.exists():
            raise FileNotFoundError(f"Label mapping file not found: {label_mapping_path}")

        with open(label_mapping_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)

        if "label2id" not in label_mapping:
            raise KeyError(f"Invalid label mapping format in: {label_mapping_path}")

        self.canonical_labels = {
            normalize_label(label): label for label in label_mapping["label2id"].keys()
        }

    def __call__(self, message: str) -> str:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")

        prompt = build_prompt(message=message, prompt_cfg=self.prompt_cfg)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()

        return canonicalize_prediction(
            prediction=generated,
            canonical_labels=self.canonical_labels,
            fuzzy_cutoff=self.fuzzy_cutoff,
        )


def main() -> None:
    suppress_known_warnings()
    args = parse_args()

    classifier = IntentClassification(args.config)

    message = args.message
    if message is None:
        message = input("Enter message: ").strip()

    predicted_label = classifier(message)
    print(predicted_label)


if __name__ == "__main__":
    main()
