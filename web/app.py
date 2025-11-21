from __future__ import annotations

import json
from typing import Any, Dict, List

import requests
import streamlit as st


API_URL = "http://localhost:8000/parse"


def call_backend(citation_text: str) -> Dict[str, Any]:
    resp = requests.post(API_URL, json={"citation_text": citation_text})
    resp.raise_for_status()
    return resp.json()


def label_to_color(label: str) -> str:
    if label.startswith("B-AUTHOR") or label.startswith("I-AUTHOR"):
        return "#ffef99"
    if label.startswith("B-TITLE") or label.startswith("I-TITLE"):
        return "#b3d9ff"
    if label.startswith("B-JOURNAL") or label.startswith("I-JOURNAL"):
        return "#ffcccc"
    if label.startswith("B-CONFERENCE") or label.startswith("I-CONFERENCE"):
        return "#c2f0c2"
    if label.startswith("B-YEAR") or label.startswith("I-YEAR"):
        return "#ffe0b3"
    if label.startswith("B-PAGES") or label.startswith("I-PAGES"):
        return "#ffcce6"
    if label.startswith("B-DOI") or label.startswith("I-DOI"):
        return "#d1c4e9"
    return "#eeeeee"


def render_tokens(tokens: List[str], labels: List[str]) -> None:
    st.markdown(
        '<style>.token{display:inline-block;padding:2px 4px;margin:2px;border-radius:4px;font-size:0.85rem}</style>',
        unsafe_allow_html=True,
    )
    html = "<div>"
    for tok, lab in zip(tokens, labels):
        color = label_to_color(lab)
        field = lab.split("-", 1)[-1] if "-" in lab else "O"
        html += f'<span class="token" style="background-color:{color}">{tok} <sub>{field}</sub></span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Citation Parser", page_icon="📚")
    st.title("📚 Citation Parser (SciBERT)")

    st.write("Paste an academic citation and extract structured metadata + BibTeX.")

    with st.form("citation-form"):
        citation = st.text_area(
            "Citation Text",
            value="Smith J., Doe A. (2019). Deep Learning for Science. "
                  "Journal of AI Research, 45(3), 120-135. doi:10.1000/jair.2019.123",
            height=150,
        )
        submitted = st.form_submit_button("Parse Citation")

    if not submitted:
        return

    if not citation.strip():
        st.error("Citation text is required.")
        return

    with st.spinner("Calling backend..."):
        try:
            result = call_backend(citation)
        except requests.RequestException as exc:
            st.error(f"Backend error: {exc}")
            return

    metadata = result["metadata"]
    bibtex = result["bibtex"]
    tokens = result["tokens"]
    labels = result["labels"]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Metadata")
        st.json(metadata)
        st.download_button(
            "Download JSON",
            data=json.dumps(metadata, indent=2),
            file_name="citation.json",
            mime="application/json",
        )
    with col2:
        st.subheader("BibTeX")
        st.code(bibtex, language="bib")

    st.subheader("Token Highlighting")
    render_tokens(tokens, labels)


if __name__ == "__main__":
    main()
