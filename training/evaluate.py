from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from seqeval.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

from training.dataset import CitationNERDataset, load_jsonl


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate citation parser model")
    p.add_argument("--val-path", type=Path, default=Path("data/processed/val.jsonl"))
    p.add_argument("--labels-path", type=Path, default=Path("data/processed/labels.json"))
    p.add_argument("--model-dir", type=Path, default=Path("model/saved_model"))
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.labels_path.open(encoding="utf-8") as f:
        labels: List[str] = json.load(f)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    val_records = load_jsonl(args.val_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    ds = CitationNERDataset(val_records, tokenizer, label2id)
    loader = DataLoader(ds, batch_size=args.batch_size)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            labels_tensor = batch["labels"].numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            for p, l in zip(preds, labels_tensor):
                cur_p = []
                cur_l = []
                for pi, li in zip(p, l):
                    if li == -100:
                        continue
                    cur_p.append(id2label[pi])
                    cur_l.append(id2label[li])
                all_preds.append(cur_p)
                all_labels.append(cur_l)

    print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    main()
