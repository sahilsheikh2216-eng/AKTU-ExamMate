"""
AKTU ExamMate — LLM Module (Optional / Not Required for Local Pipeline)

The core pipeline operates entirely locally and deterministically using:
- Deterministic regex and page segmenters for question extraction
- Local sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) for embeddings
Zero external API calls are made to prevent costs, rate limits, and latency.
"""
