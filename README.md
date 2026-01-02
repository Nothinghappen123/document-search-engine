Hybrid Document Search Engine (BM25 + Semantic Reranking)
Overview

This project implements a hybrid document retrieval system that efficiently retrieves and ranks relevant documents from a large text corpus using a combination of lexical search (BM25) and semantic similarity (transformer-based embeddings).

The system is designed as a two-stage retrieval pipeline, similar to the retrieval layer used in modern enterprise search systems and Retrieval-Augmented Generation (RAG) architectures.

This project focuses on document retrieval, not answer generation.

Problem Statement

Searching large text corpora using only keyword-based methods often fails to capture semantic meaning, while purely embedding-based search is computationally expensive at scale.

This project addresses the problem of efficient and accurate document retrieval by:

Using BM25 for fast lexical candidate retrieval

Applying semantic reranking using transformer embeddings to improve relevance

Exposing the retrieval pipeline via a FastAPI service

System Architecture

The system follows a hybrid two-stage retrieval architecture:

1. Lexical Retrieval (BM25)

Builds an inverted index over the document corpus

Retrieves top-k candidate documents based on keyword overlap

Optimized for speed and scalability

2. Semantic Reranking

Uses a transformer-based sentence embedding model (all-MiniLM-L6-v2)

Computes semantic similarity between query and candidate documents

Reranks BM25 results to improve contextual relevance

3. API Layer

FastAPI-based backend

Provides a /search endpoint for querying the system

Returns ranked documents with lexical and semantic scores

Technologies Used

Python

FastAPI

BM25 (Information Retrieval)

Sentence-Transformers (all-MiniLM-L6-v2)

scikit-learn

NLTK

Streamlit (local demo UI)

Dataset

Wikipedia Plaintext Dataset (sampled)

Each document consists of:

Title

Full article text

⚠️ Dataset is not included in this repository due to size constraints.

Dataset Setup

Download a Wikipedia plaintext dataset (for example from Kaggle) and place it in:

data/


The system builds indexes and cached artifacts locally during execution.

How to Run Locally
1. Clone the repository
git clone https://github.com/<your-username>/document-search-engine.git
cd document-search-engine

2. Create virtual environment
python -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Start backend API
uvicorn src.app:app

5. (Optional) Start local UI
streamlit run ui.py

Example Query

Query:

radio station mexico


Top Results:

XHVG-FM (Baja California)

XHCJX-FM

XHTRR-FM

Each result includes:

BM25 score (lexical relevance)

Semantic similarity score (contextual relevance)

What This Project Is (and Is Not)
This project IS:

A document retrieval system

A search engine over a fixed corpus

The retrieval layer used in RAG systems

Relevant to enterprise search, NLP, and GenAI pipelines

This project is NOT:

Google Search

A chatbot

A question-answering system

A web crawler

Future Improvements

Integrate Retrieval-Augmented Generation (RAG) for answer generation

Support PDF and domain-specific document ingestion

Add document chunking and snippet highlighting

Deploy API to cloud platform

Key Learning Outcomes

Implemented inverted indexing and BM25 ranking

Applied transformer-based semantic reranking

Built a production-style FastAPI service

Designed a scalable and explainable retrieval pipeline

License

This project is for educational and portfolio purposes.
