# Resume bullets (draft)

1. Built a **local multimodal Document QA** system (Gemma 3 4B QAT GGUF + PaddleOCR + FAISS) with a fair **text-only vs multimodal RAG** design that shares retrieval and feeds only top-k page images to the VLM.

2. Implemented **page-level and hierarchical (coarse-to-fine) retrieval** and a config-driven evaluation harness (ANLS/EM/F1, recall@k, latency, RSS/VRAM) across **5–100 page** document buckets with CSV/JSON/Markdown reporting.

3. Benchmarked **quantized on-device inference practicality** (TTFT, tokens/sec, memory) for text vs image+text, documenting deployability trade-offs for private laptop/desktop DocQA.
