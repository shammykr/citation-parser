from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from training.dataset import CitationNERDataset, load_jsonl


@dataclass
class TrainConfig:
    train_path: Path
    val_path: Path
    labels_path: Path
    model_name: str
    output_dir: Path
    batch_size: int
    epochs: int
    learning_rate: float
    seed: int


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Fine-tune SciBERT on citation BIO tags")
    p.add_argument("--train-path", type=Path, default=Path("data/processed/train.jsonl"))
    p.add_argument("--val-path", type=Path, default=Path("data/processed/val.jsonl"))
    p.add_argument("--labels-path", type=Path, default=Path("data/processed/labels.json"))
    p.add_argument("--model-name", type=str, default="allenai/scibert_scivocab_uncased")
    p.add_argument("--output-dir", type=Path, default=Path("model/saved_model"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return TrainConfig(
        args.train_path,
        args.val_path,
        args.labels_path,
        args.model_name,
        args.output_dir,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.seed,
    )


def compute_metrics_builder(label_list: List[str]):
    def compute_metrics(predictions_output: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        predictions, labels = predictions_output
        preds = np.argmax(predictions, axis=2)
        true_predictions = []
        true_labels = []
        for pred, lab in zip(preds, labels):
            cur_preds = []
            cur_labs = []
            for p, l in zip(pred, lab):
                if l == -100:
                    continue
                cur_preds.append(label_list[p])
                cur_labs.append(label_list[l])
            true_predictions.append(cur_preds)
            true_labels.append(cur_labs)
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    return compute_metrics


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    with cfg.labels_path.open(encoding="utf-8") as f:
        labels: List[str] = json.load(f)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    train_records = load_jsonl(cfg.train_path)
    val_records = load_jsonl(cfg.val_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds = CitationNERDataset(train_records, tokenizer, label2id)
    val_ds = CitationNERDataset(val_records, tokenizer, label2id)

    config = AutoConfig.from_pretrained(
        cfg.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModelForTokenClassification.from_pretrained(cfg.model_name, config=config)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        # metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(labels),
    )

    trainer.train()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))


if __name__ == "__main__":
    main()
