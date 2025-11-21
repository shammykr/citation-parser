from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from backend.utils import bio_to_metadata, metadata_to_bibtex


class CitationParserModel:
    """Wrapper around a fine‑tuned SciBERT token classification model."""

    def __init__(self, model_dir: str | Path, labels_path: str | Path) -> None:
        model_dir = Path(model_dir)
        labels_path = Path(labels_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        import json

        with labels_path.open(encoding="utf-8") as f:
            labels = json.load(f)
        self.id2label = {i: lab for i, lab in enumerate(labels)}

    def tokenize(self, text: str) -> List[str]:
        import re

        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def predict(self, citation_text: str) -> Dict:
        tokens = self.tokenize(citation_text)
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()

        word_ids = self.tokenizer(tokens, is_split_into_words=True).word_ids()
        labels: List[str] = []
        for idx, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid < len(tokens):
                label_id = preds[idx]
                labels.append(self.id2label.get(label_id, "O"))
        labels = labels[: len(tokens)]

        metadata = bio_to_metadata(tokens, labels)
        bibtex = metadata_to_bibtex(metadata)
        return {
            "tokens": tokens,
            "labels": labels,
            "metadata": metadata,
            "bibtex": bibtex,
        }
