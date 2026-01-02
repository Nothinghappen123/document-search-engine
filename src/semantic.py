from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticReranker:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def rerank(self, query, candidates, top_k=5):
        """
        candidates: list of dicts with keys
        {doc_id, title, text, bm25_score}
        """
        if not candidates:
            return []

        # Encode query
        query_embedding = self.model.encode([query])

        # Encode candidate documents
        doc_texts = [doc["text"] for doc in candidates]
        doc_embeddings = self.model.encode(doc_texts)

        # Cosine similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Attach semantic score
        for i, sim in enumerate(similarities):
            candidates[i]["semantic_score"] = float(sim)

        # Sort by semantic score
        candidates.sort(key=lambda x: x["semantic_score"], reverse=True)

        return candidates[:top_k]
