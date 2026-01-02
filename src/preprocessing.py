import re
import csv
from typing import List, Dict, Generator

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys
import csv

csv.field_size_limit(sys.maxsize)

# One-time downloads (run once)
# Comment these after first run
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def normalize_text(text: str) -> List[str]:
    """
    Clean and normalize raw text.
    Output: list of normalized tokens.
    """
    if not text:
        return []

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep letters only
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords + lemmatize
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 2
    ]

    return tokens


def read_documents(
    csv_path: str, chunk_size: int = 1000
) -> Generator[Dict, None, None]:
    """
    Stream documents from CSV to avoid memory blowups.
    Yields dict: {id, title, text}
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) == chunk_size:
                for doc in batch:
                    yield doc
                batch = []
        # remaining
        for doc in batch:
            yield doc


def preprocess_corpus(csv_path: str) -> Generator[Dict, None, None]:
    """
    Main preprocessing pipeline.
    Yields:
      {
        "id": <doc_id>,
        "title": <title>,
        "tokens": <list of tokens>
      }
    """
    for doc in read_documents(csv_path):
        tokens = normalize_text(doc.get("text", ""))
        if not tokens:
            continue

        yield {
            "id": doc.get("id"),
            "title": doc.get("title"),
            "tokens": tokens,
        }


if __name__ == "__main__":
    # Quick sanity check (DO NOT PROCESS FULL CORPUS HERE)
    from itertools import islice

    DATA_PATH = "data/sampled_wikipedia.csv"


    for item in islice(preprocess_corpus(DATA_PATH), 3):
        print(item["id"], item["title"])
        print(item["tokens"][:20])
        print("-" * 40)
