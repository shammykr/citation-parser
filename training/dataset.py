from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            records.append(json.loads(line))
    return records


class CitationNERDataset(Dataset):
    """Dataset that aligns word-level BIO labels to subword tokens."""

    def __init__(
        self,
        records: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        label2id: Dict[str, int],
        max_length: int = 256,
        label_all_tokens: bool = True,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        tokens = record["tokens"]
        labels = record["labels"]
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=False,
        )
        word_ids = encoding.word_ids()
        previous_word_idx = None
        label_ids: List[int] = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
                continue
            label = labels[word_idx]
            if word_idx != previous_word_idx:
                label_ids.append(self.label2id[label])
            else:
                if self.label_all_tokens and label.startswith("B-"):
                    label = label.replace("B-", "I-")
                label_ids.append(self.label2id[label])
            previous_word_idx = word_idx
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(label_ids)
        return encoding
