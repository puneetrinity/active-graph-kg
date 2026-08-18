# Search scoring modes

Memory's active retrieval contract is authenticated `POST /search`. It supports vector and hybrid retrieval;
callers must interpret ranking only through that response contract and its documented mode.

Generic grounded Q&A and production score explanation are unavailable for launch:

- `POST /ask` → HTTP 410 `MEMORY_GROUNDED_QA_UNAVAILABLE`;
- `POST /ask/stream` → the same no-work HTTP 410;
- `POST /debug/search_explain` → the same no-work HTTP 410.

Those compatibility routes do not parse a body, authenticate, search, rerank, construct an LLM provider, or expose
internal scoring diagnostics. Historical Q&A thresholds such as `ASK_SIM_THRESHOLD`,
`RRF_LOW_SIM_THRESHOLD`, and `RAW_LOW_SIM_THRESHOLD` are not active API configuration.

## Direct search

Use the request's `use_hybrid` flag to select the supported retrieval mode:

```bash
curl -X POST http://localhost:8000/search \\
  -H 'Authorization: Bearer <token>' \\
  -H 'Content-Type: application/json' \\
  -d '{"query":"machine learning","top_k":10,"use_hybrid":true}'
```

Keep direct-search evaluation in `evaluation/weighted_search_eval.py`. Any future operator explanation tool or
recruiter assistant requires a separate product decision, typed score semantics, bounded execution, and a new
activation review.
