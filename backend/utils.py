from __future__ import annotations

from typing import Dict, List


ENTITY_FIELDS = [
    "AUTHOR",
    "TITLE",
    "JOURNAL",
    "CONFERENCE",
    "YEAR",
    "VOLUME",
    "ISSUE",
    "PAGES",
    "DOI",
]


def bio_to_spans(tokens: List[str], labels: List[str]) -> Dict[str, List[str]]:
    """Convert BIO tags to field → list of text spans."""
    field_spans: Dict[str, List[List[str]]] = {f: [] for f in ENTITY_FIELDS}
    current_field: str | None = None
    current_tokens: List[str] = []

    def flush():
        nonlocal current_field, current_tokens
        if current_field and current_tokens:
            field_spans[current_field].append(current_tokens)
        current_field = None
        current_tokens = []

    for tok, lab in zip(tokens, labels):
        if lab == "O" or not lab:
            flush()
            continue
        if "-" not in lab:
            flush()
            continue
        prefix, field = lab.split("-", 1)
        if prefix == "B":
            flush()
            current_field = field
            current_tokens = [tok]
        elif prefix == "I":
            if current_field == field:
                current_tokens.append(tok)
            else:
                flush()
                current_field = field
                current_tokens = [tok]
    flush()

    merged: Dict[str, List[str]] = {}
    for field, spans in field_spans.items():
        merged[field] = [" ".join(span).replace(" ,", ",").replace(" .", ".") for span in spans]
    return merged


def spans_to_metadata(span_dict: Dict[str, List[str]]) -> Dict[str, str | List[str]]:
    authors = span_dict.get("AUTHOR", [])
    metadata: Dict[str, str | List[str]] = {
        "authors": authors,
        "title": " ".join(span_dict.get("TITLE", [])) or "",
        "journal": " ".join(span_dict.get("JOURNAL", [])) or "",
        "conference": " ".join(span_dict.get("CONFERENCE", [])) or "",
        "year": " ".join(span_dict.get("YEAR", [])) or "",
        "volume": " ".join(span_dict.get("VOLUME", [])) or "",
        "issue": " ".join(span_dict.get("ISSUE", [])) or "",
        "pages": " ".join(span_dict.get("PAGES", [])) or "",
        "doi": " ".join(span_dict.get("DOI", [])) or "",
    }
    return metadata


def bio_to_metadata(tokens: List[str], labels: List[str]) -> Dict[str, str | List[str]]:
    spans = bio_to_spans(tokens, labels)
    return spans_to_metadata(spans)


def metadata_to_bibtex(metadata: Dict[str, str | List[str]]) -> str:
    authors = metadata.get("authors") or []
    if isinstance(authors, list):
        author_str = " and ".join(authors)
    else:
        author_str = str(authors)

    year = metadata.get("year", "")
    if isinstance(year, list):
        year = " ".join(year)

    key_base = ""
    if authors:
        key_base += str(authors[0]).split()[0].lower()
    key = f"{key_base}{year}".strip() or "citation"

    lines = [
        f"@article{{{key},",
        f"  author = {{{author_str}}},",
    ]
    for field, bib_key in [
        ("title", "title"),
        ("journal", "journal"),
        ("conference", "booktitle"),
        ("year", "year"),
        ("volume", "volume"),
        ("issue", "number"),
        ("pages", "pages"),
        ("doi", "doi"),
    ]:
        value = metadata.get(field)
        if not value:
            continue
        if isinstance(value, list):
            value = " ".join(value)
        lines.append(f"  {bib_key} = {{{value}}},")
    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines.append("}")
    return "\n".join(lines)
