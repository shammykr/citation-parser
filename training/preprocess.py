from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class PreprocessConfig:
    raw_path: Path
    output_dir: Path
    val_ratio: float
    seed: int


def parse_args() -> PreprocessConfig:
    p = argparse.ArgumentParser(description="Preprocess annotated citation data")
    p.add_argument("--raw-path", type=Path, default=Path("data/raw/annotated_citations.json"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return PreprocessConfig(args.raw_path, args.output_dir, args.val_ratio, args.seed)


def load_records(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def split_records(records: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rnd = random.Random(seed)
    shuffled = records[:]
    rnd.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def write_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def write_label_file(records: List[Dict], path: Path) -> None:
    labels = set()
    for rec in records:
        labels.update(rec["labels"])
    ordered = sorted(labels)
    path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    records = load_records(cfg.raw_path)
    train, val = split_records(records, cfg.val_ratio, cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, cfg.output_dir / "train.jsonl")
    write_jsonl(val, cfg.output_dir / "val.jsonl")
    write_label_file(records, cfg.output_dir / "labels.json")
    print(f"Wrote {len(train)} train and {len(val)} val samples.")


if __name__ == "__main__":
    main()
