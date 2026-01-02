from collections import defaultdict
from src.preprocessing import preprocess_corpus


def build_inverted_index(csv_path, limit=None):
    index = defaultdict(dict)
    doc_lengths = {}
    total_docs = 0

    for i, doc in enumerate(preprocess_corpus(csv_path)):
        doc_id = doc["id"]
        tokens = doc["tokens"]

        doc_lengths[doc_id] = len(tokens)
        total_docs += 1

        for token in tokens:
            if doc_id in index[token]:
                index[token][doc_id] += 1
        else:
            index[token][doc_id] = 1

        if limit and i >= limit:
            break

    return index, doc_lengths, total_docs
if __name__ == "__main__":
    index, doc_lengths, N = build_inverted_index("data/sampled_wikipedia.csv", limit=1000)
    print("radio" in index)
    print("radio doc count:", len(index["radio"]))
    print("Sample doc length:", list(doc_lengths.items())[:3])
    print("Total docs:", N)
