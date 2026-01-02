from fastapi import FastAPI
from src.search import SearchEngine
from src.semantic import SemanticReranker


app = FastAPI(
    title="Hybrid Document Search Engine",
    description="BM25 + Transformer-based Semantic Reranking",
    version="1.0"
)

# Initialize once (cached index will be reused)
engine = SearchEngine("data/sampled_wikipedia.csv", limit=5000)
reranker = SemanticReranker()


@app.get("/search")
def search(query: str, top_k: int = 5):
    """
    Search endpoint.
    """
    candidates = engine.search(query, rerank_k=20)
    results = reranker.rerank(query, candidates, top_k=top_k)

    return [
        {
            "title": r["title"],
            "bm25_score": r["bm25_score"],
            "semantic_score": r["semantic_score"]
        }
        for r in results
    ]
