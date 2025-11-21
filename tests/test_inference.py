from backend.utils import bio_to_metadata, metadata_to_bibtex


def test_bio_decoding_roundtrip():
    tokens = ["Smith", "J.", ",", "Doe", "A.", "(", "2019", ")", ".", "Deep", "Learning"]
    labels = [
        "B-AUTHOR",
        "I-AUTHOR",
        "O",
        "B-AUTHOR",
        "I-AUTHOR",
        "O",
        "B-YEAR",
        "O",
        "O",
        "B-TITLE",
        "I-TITLE",
    ]
    metadata = bio_to_metadata(tokens, labels)
    assert metadata["authors"] == ["Smith J.", "Doe A."]
    assert metadata["year"] == "2019"
    assert metadata["title"] == "Deep Learning"
    bibtex = metadata_to_bibtex(metadata)
    assert "author = {Smith J. and Doe A.}" in bibtex
    assert "year = {2019}" in bibtex
