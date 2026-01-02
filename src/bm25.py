import math
from src.preprocessing import normalize_text


class BM25:
    def __init__(self, inverted_index, doc_lengths, total_docs, k1=1.5, b=0.75):
        self.index = inverted_index
        self.doc_lengths = doc_lengths
        self.N = total_docs
        self.k1 = k1
        self.b = b
        self.avg_doc_len = sum(doc_lengths.values()) / total_docs

    def idf(self, term):
        df = len(self.index.get(term, {}))
        if df == 0:
            return 0.0
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query_tokens):
        scores = {}

        for term in query_tokens:
            if term not in self.index:
                continue

            idf = self.idf(term)

            for doc_id, tf in self.index[term].items():
                dl = self.doc_lengths[doc_id]

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (dl / self.avg_doc_len)
                )

                score = idf * (numerator / denominator)
                scores[doc_id] = scores.get(doc_id, 0) + score

        return scores
if __name__ == "__main__":
    from indexer import build_inverted_index

    index, doc_lengths, N = build_inverted_index("data/sampled_wikipedia.csv", limit=2000)

    bm25 = BM25(index, doc_lengths, N)

    query = "radio station mexico"
    query_tokens = normalize_text(query)

    scores = bm25.score(query_tokens)
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

    print(top_docs)
