from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from backend.model import CitationParserModel
from backend.utils import bio_to_metadata, metadata_to_bibtex

app = FastAPI(title="Citation Parser API", version="1.0.0")


class ParseRequest(BaseModel):
    citation_text: str


class ParseResponse(BaseModel):
    tokens: List[str]
    labels: List[str]
    metadata: Dict[str, Any]
    bibtex: str


MODEL: CitationParserModel | None = None


@app.on_event("startup")
def load_model() -> None:
    global MODEL
    model_dir = Path("model/saved_model")
    labels_path = Path("data/processed/labels.json")
    if model_dir.exists() and labels_path.exists():
        MODEL = CitationParserModel(model_dir, labels_path)
    else:
        MODEL = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/parse", response_model=ParseResponse)
def parse(req: ParseRequest) -> ParseResponse:
    if MODEL is None:
        # simple fallback: all tokens labeled O
        tokens = req.citation_text.split()
        labels = ["O"] * len(tokens)
        metadata = bio_to_metadata(tokens, labels)
        bibtex = metadata_to_bibtex(metadata)
        return ParseResponse(tokens=tokens, labels=labels, metadata=metadata, bibtex=bibtex)
    result = MODEL.predict(req.citation_text)
    return ParseResponse(
        tokens=result["tokens"],
        labels=result["labels"],
        metadata=result["metadata"],
        bibtex=result["bibtex"],
    )
