📚 Citation Parser using Transformers
A Machine Learning Pipeline for Structured Citation Component Extraction

🚀 Overview
This project implements an end-to-end Transformer-based Named Entity Recognition (NER) system for extracting structured metadata from XML-annotated citation strings.
It includes:
Large-scale dataset preprocessing (700MB+)
BIO-tag generation from XML
Fine-tuning SciBERT/BERT for sequence labeling
Evaluation using seqeval (precision/recall/F1)
Inference pipeline that converts predictions → structured JSON
Optional Web UI for user-friendly citation testing

This project demonstrates a complete ML engineering workflow:
data processing → dataset creation → model training → evaluation → deployment.

✨ Features
🔧 Dataset Processing
Stream-based CSV loader (handles very large files)
XML annotation parsing for:
authors
titles
journals
years
volumes
issues
pages
DOI/URL
Tokenization & BIO-label generation

🤖 Model Training (Transformers)

Fine-tuning SciBERT / BERT base
HuggingFace Trainer pipeline
Training arguments fully configurable (epochs, batch size, eval strategy)

📈 Evaluation

span-level metrics (F1, precision, recall)
confusion matrix utilities
error analysis helper scripts

⚙️ Inference Pipeline
Converts model token predictions into structured citation components:
{
  "authors": [...],
  "title": "...",
  "journal": "...",
  "year": "...",
  "volume": "...",
  "issue": "...",
  "pages": "...",
  "doi": "..."
}

🌐 Optional Web UI
A simple frontend to paste citations and see parsed output.

📁 Project Structure
Citation_Parser/
│
├── training/
│   ├── train.py
│   ├── preprocess.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── labels.txt
│   └── config.json
│
├── tools/
│   └── convert_csv_to_biojson.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── examples/
│
├── backend/
├── web/
└── README.md


🛠 Installation
1. Create virtual environment
python -m venv .venv

2. Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

3. Install dependencies
pip install -r requirements.txt

📑 Dataset Conversion
The dataset consists of citation strings containing XML annotations:
<author><family>Doe</family><given>J.</given></author>
<title>Deep Learning for Citation Parsing</title>
<container-title>Journal of AI</container-title>
<issued><year>2020</year></issued>
<URL>https://doi.org/10.xxxx/yyyy</URL>

Convert CSV → BIO JSON:
python tools/csv_to_biojson.py

Then preprocess into model-ready format:
python training/preprocess.py

🎯 Training
python -m training.train

Model checkpoints appear under:
models/checkpoints/

🧪 Evaluation
python training/evaluate.py

Outputs:
F1 score
precision
recall
per-label metrics

📤 Inference Example
from backend.inference import CitationParser

parser = CitationParser("models/checkpoints/best_model")
result = parser.parse("<annotated citation here>")
print(result)


⚠️ Current Limitations
Important:
The model is trained only on XML-annotated citations.
Therefore:
✔ Works well for:
Citations containing XML tags like:
<title>...</title>
<author>...</author>
<year>2020</year>

❌ Does not generalize to:
raw APA
raw MLA
Chicago
IEEE
book chapter formats
Springer LNCS (without tags)

Reason:
The dataset includes tags, not raw citations.
So the model learns annotation patterns, not citation structure.

⛔ The current version is a citation-annotation NER model, not a full citation parser.

🔮 Future Work
To become a full citation parser, next steps include:
Training on raw citations with human-annotated BIO labels

Supporting more fields:
editors
booktitle
series
publisher
location
ISBN
Adding LLM-based parsing (GPT-4/5)
Integrating GROBID for hybrid parsing
Style-agnostic extraction (APA, MLA, Chicago, IEEE)

⭐ Why This Project Is Valuable
This project demonstrates:
Large-scale dataset handling (700MB+)
Text parsing and annotation pipelines
BIO tag generation
Fine-tuning large Transformer models
Evaluation with seqeval
API and UI design
End-to-end ML engineering
Excellent for roles in:
Natural Language Processing
Machine Learning Engineering
AI Research Engineering
Data Science

🙏 Acknowledgements
HuggingFace Transformers
AllenAI SciBERT
seqeval
Original XML-annotated citation dataset
