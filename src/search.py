import csv
import os

from src.preprocessing import normalize_text
from src.indexer import build_inverted_index
from src.bm25 import BM25
from src.cache import save, load, exists



# # Cache paths
INDEX_PATH = "data/index.pkl"
DOC_LEN_PATH = "data/doc_lengths.pkl"
N_PATH = "data/N.pkl"


class SearchEngine:
    def __init__(self, csv_path, limit=None):
        self.csv_path = csv_path

        # -------- LOAD OR BUILD INDEX --------
        if exists(INDEX_PATH) and exists(DOC_LEN_PATH) and exists(N_PATH):
            self.index = load(INDEX_PATH)
            self.doc_lengths = load(DOC_LEN_PATH)
            self.N = load(N_PATH)
        else:
            self.index, self.doc_lengths, self.N = build_inverted_index(
                csv_path, limit=limit
            )
            save(self.index, INDEX_PATH)
            save(self.doc_lengths, DOC_LEN_PATH)
            save(self.N, N_PATH)

        # -------- LOAD DOCUMENTS --------
        self.docs = self._load_docs()

        # -------- INIT BM25 --------
        self.bm25 = BM25(self.index, self.doc_lengths, self.N)

    def _load_docs(self):
        """
        Load full documents (id -> title + text)
        Needed for semantic reranking.
        """
        docs = {}
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                docs[row["id"]] = {
                    "title": row["title"],
                    "text": row["text"]
                }
        return docs

    def search(self, query, rerank_k=20):
        """
        Returns BM25 candidates for semantic reranking.
        """
        # Preprocess query
        query_tokens = normalize_text(query)

        # BM25 scoring
        scores = self.bm25.score(query_tokens)

        # Top-K lexical candidates
        ranked = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )[:rerank_k]

        # Prepare candidates with raw text
        candidates = []
        for doc_id, score in ranked:
            doc = self.docs.get(doc_id)
            if not doc:
                continue

            candidates.append({
                "doc_id": doc_id,
                "title": doc["title"],
                "text": doc["text"],
                "bm25_score": score
            })

        return candidates


# =========================
# TEST: BM25 + SEMANTIC
# =========================
if __name__ == "__main__":
    from semantic import SemanticReranker

    engine = SearchEngine("data/sampled_wikipedia.csv", limit=5000)
    reranker = SemanticReranker()

    query = "radio station mexico"

    candidates = engine.search(query, rerank_k=20)
    results = reranker.rerank(query, candidates, top_k=5)

    for r in results:
        print(r["semantic_score"], "-", r["title"])
