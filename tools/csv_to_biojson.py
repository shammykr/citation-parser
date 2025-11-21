import csv
import json
import re
import argparse
from html import unescape

# =========================
# 1. TAG → LABEL MAPPING
# =========================
TAG_LABEL_MAP = {
    "family":   "AUTHOR",
    "given":    "AUTHOR",
    "year":     "YEAR",
    "title":    "TITLE",
    "container-title": "JOURNAL",
    "volume":   "VOLUME",
    "issue":    "ISSUE",
    "page":     "PAGES",
    "URL":      "DOI",   # because your sample uses DOI inside <URL>
}

# =========================
# 2. TOKENIZER
# =========================
def tokenize(text):
    # Basic tokenizer: splits words, punctuation, and URL fragments
    tokens = re.findall(r"https?://\S+|[\w\-]+|[.,()\"“”‘’:;!?]", text)
    return tokens

# =========================
# 3. SIMPLE TAG PARSER
# =========================
TAG_REGEX = re.compile(r"<(/?)([\w\-]+)>([^<]*)")

def parse_annotated_string(s):
    """
    INPUT:
        <author><family>Ritchie</family>, <given>E.</given>...</author>
    OUTPUT:
        tokens: [...]
        labels: [...]
    """
    s = unescape(s)  # reduce HTML escapes

    tokens = []
    labels = []

    active_label = None  # current BIO tag

    # Scan through annotated markup
    pos = 0
    while pos < len(s):
        match = TAG_REGEX.search(s, pos)
        if not match:
            raw_text = s[pos:].strip()
            if raw_text:
                for t in tokenize(raw_text):
                    tokens.append(t)
                    labels.append("O")
            break

        start, end = match.span()
        text_before = s[pos:start]

        # Add tokens before tag
        if text_before.strip():
            for t in tokenize(text_before):
                tokens.append(t)
                labels.append("O")

        is_closing = match.group(1) == "/"
        tag = match.group(2)
        inner_text = match.group(3).strip()

        if not is_closing:  
            # Opening tag
            if tag in TAG_LABEL_MAP:
                active_label = TAG_LABEL_MAP[tag]

            # If inner text exists: label it right away
            if inner_text:
                toks = tokenize(inner_text)
                if active_label:
                    tokens.extend(toks)
                    labels.extend(
                        ["B-"+active_label] + ["I-"+active_label]*(len(toks)-1)
                    )
                else:
                    tokens.extend(toks)
                    labels.extend(["O"]*len(toks))

        else:
            # Closing tag ends the active label
            active_label = None

        pos = end

    return tokens, labels

# =========================
# 4. MAIN CONVERSION LOGIC
# =========================
def process_csv(input_path, output_path):
    output_data = []

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader):
            annotated = row.get("citationStringAnnotated", "")
            if not annotated.strip():
                continue

            try:
                tokens, labels = parse_annotated_string(annotated)
                if tokens:
                    output_data.append({
                        "tokens": tokens,
                        "labels": labels
                    })
            except Exception as e:
                print(f"[WARN] Parsing error at row {row_idx}: {e}")

            # Console progress every 10k rows
            if row_idx % 10000 == 0:
                print(f"Processed {row_idx} rows...")

    print(f"Writing output: {output_path}")
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(output_data, out, indent=2)

    print(f"Done! Total records: {len(output_data)}")


# =========================
# 5. CLI ENTRYPOINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to 700MB CSV input")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    process_csv(args.input, args.output)
