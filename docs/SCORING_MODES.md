# Scoring Modes

Active Graph KG supports two primary scoring modes for hybrid search: **RRF (Reciprocal Rank Fusion)** and **Cosine similarity** (weighted fusion). Each mode has different score scales and requires different threshold tuning.

## Overview

| Mode | Score Range | Typical Top Scores | Use Case |
|------|-------------|-------------------|----------|
| **RRF** | 0.01-0.04 | ~0.02-0.035 | Production default, rank-based fusion |
| **Cosine** | 0.0-1.0 | ~0.3-0.9 | Research/tuning, similarity-based fusion |

## RRF Mode (Reciprocal Rank Fusion)

RRF fuses vector similarity and BM25 text search using reciprocal rank scores rather than raw similarity values. This produces more stable, scale-independent scores.

### Configuration

```bash
# RRF Mode Environment Variables
export HYBRID_RRF_ENABLED=true
export RRF_LOW_SIM_THRESHOLD=0.01      # Extremely low threshold (gate out garbage)
export ASK_SIM_THRESHOLD=0.01          # /ask endpoint threshold (aligned with RRF scale)
export HYBRID_RRF_K=60                 # RRF K parameter (controls rank decay)

# Optional: Reranker settings
export HYBRID_RERANKER_BASE=20         # Base candidate count for reranking
export HYBRID_RERANKER_BOOST=45        # Additional candidates for low-confidence queries
export HYBRID_ADAPTIVE_THRESHOLD=0.55  # Confidence threshold for boost
export MAX_RERANK_BUDGET_MS=250        # Maximum reranking time budget
```

### Score Characteristics

- **Range**: 0.01 to ~0.04 (rarely exceeds 0.05)
- **Interpretation**: Rank-based, not similarity-based
  - 0.03+ = strong match (top-3 in both vector and BM25)
  - 0.02-0.03 = good match (top-10 in both rankings)
  - 0.01-0.02 = moderate match (present in both rankings)
  - <0.01 = weak/absent (rejected by gating)

### Advantages

- **Scale-independent**: Scores are consistent regardless of embedding model changes
- **Stable thresholds**: 0.01 threshold works across different datasets
- **Better fusion**: Rank-based fusion avoids score magnitude mismatch between vector and BM25
- **Production-ready**: Requires minimal tuning

### Reranking Behavior

In RRF mode, the rerank skip logic (`RERANK_SKIP_TOPSIM`) is **disabled** because RRF scores (0.01-0.04) never reach typical cosine skip thresholds (0.80+). All candidates are reranked when `use_reranker=true`.

## Cosine Mode (Weighted Fusion)

Cosine mode uses raw cosine similarity scores combined with BM25 scores via weighted fusion. This produces human-interpretable similarity values but requires careful threshold tuning.

### Configuration

```bash
# Cosine Mode Environment Variables
export HYBRID_RRF_ENABLED=false
export RAW_LOW_SIM_THRESHOLD=0.15      # Extremely low threshold for cosine similarity
export ASK_SIM_THRESHOLD=0.20          # /ask endpoint threshold (cosine scale)

# Optional: Reranker settings
export RERANK_SKIP_TOPSIM=0.80         # Skip reranking if top score >= 0.80 (high confidence)
```

### Score Characteristics

- **Range**: 0.0 to 1.0 (theoretical maximum, rarely exceeds 0.95)
- **Interpretation**: Similarity-based
  - 0.80+ = very strong match (near-duplicate, semantic match)
  - 0.60-0.80 = strong match (clear relevance)
  - 0.40-0.60 = moderate match (related content)
  - 0.20-0.40 = weak match (tangentially related)
  - <0.20 = very weak match (rejected by gating)

### Advantages

- **Interpretable**: Scores represent actual similarity percentages
- **Fine-grained tuning**: Can set precise thresholds for different quality levels
- **Research-friendly**: Easier to analyze and compare results

### Disadvantages

- **Threshold sensitivity**: Requires dataset-specific tuning
- **Embedding drift**: Scores change when embedding model is updated
- **Scale mismatch**: BM25 and vector scores may have different magnitudes

### Reranking Behavior

In cosine mode, the rerank skip logic is **enabled**. If the top candidate's score is ≥ `RERANK_SKIP_TOPSIM` (default 0.80), reranking is skipped to save latency, assuming high confidence.

## Metadata Exposure

Both `/ask` and `/debug/search_explain` endpoints expose scoring mode metadata:

### /ask Response

```json
{
  "answer": "...",
  "citations": [...],
  "confidence": 0.85,
  "metadata": {
    "gating_score": 0.033,           // Top similarity score used for gating
    "gating_score_type": "rrf_fused", // "rrf_fused" or "cosine"
    "top_similarity": 0.033,
    "cited_nodes": 3,
    "searched_nodes": 12
  }
}
```

### /debug/search_explain Response

```json
{
  "query": "machine learning frameworks",
  "mode": "hybrid",
  "score_type": "rrf_fused",         // Overall scoring mode
  "score_range": "0.01-0.04 (low)",  // Expected range for this mode
  "results": [
    {
      "node_id": "abc123",
      "similarity": 0.0328,
      "score_type": "rrf_fused",     // Per-result score type
      "classes": ["Job"],
      "snippet": "..."
    }
  ],
  "scoring_notes": {
    "rrf_fused": "RRF scores range 0.01-0.04 (rank-based fusion of vector+BM25)",
    "weighted_fusion": "Weighted scores range 0.0-1.0 (linear combination of vector+BM25)",
    "cosine": "Cosine similarity range 0.0-1.0 (vector-only)"
  }
}
```

## Threshold Tuning Guide

### RRF Mode Tuning

1. **Start with defaults**: `RRF_LOW_SIM_THRESHOLD=0.01`, `ASK_SIM_THRESHOLD=0.01`
2. **Monitor rejection rates**: Check `/ask` metadata for `reason: "extremely_low_similarity"`
3. **Adjust conservatively**:
   - If too many good queries rejected: lower to 0.008
   - If too much garbage getting through: raise to 0.012
4. **Don't overthink it**: RRF thresholds are remarkably stable across datasets

### Cosine Mode Tuning

1. **Baseline with /debug/search_explain**: Run representative queries and check score distributions
2. **Set `RAW_LOW_SIM_THRESHOLD`**: Reject clearly irrelevant results (typically 0.10-0.20)
3. **Set `ASK_SIM_THRESHOLD`**: Higher than raw threshold for LLM generation (typically 0.15-0.25)
4. **Validate with edge cases**: Test boundary queries to ensure thresholds are appropriate
5. **Re-tune after model changes**: Cosine scores shift when embedding models change

## Observability

### Key Metrics to Track

1. **Scoring distribution**:
   - `metadata.gating_score` histogram by `gating_score_type`
   - P50, P95, P99 scores for both modes

2. **Rejection rates**:
   - Count of `reason: "extremely_low_similarity"` by mode
   - Track as percentage of total queries

3. **Cited nodes distribution**:
   - `metadata.cited_nodes` histogram
   - Zero-citation rate (answers without sources)

4. **Latency by mode**:
   - RRF vs cosine search latency
   - Reranking time when applied

### Debugging Low-Quality Results

```bash
# Check score distribution
curl -s -X POST http://localhost:8000/debug/search_explain \
  -H "Content-Type: application/json" \
  -d '{"query":"your query here","use_hybrid":true,"top_k":20}' \
  | jq '{score_type, score_range, top_5_scores: [.results[0:5][].similarity]}'

# Check embedding health
curl -s http://localhost:8000/debug/embed_info | jq

# Check if RRF is actually enabled
curl -s http://localhost:8000/health | jq .env_config
```

## Migration Between Modes

### From Cosine to RRF

1. Update environment variables (see RRF configuration above)
2. **No re-embedding required** - uses same vector embeddings
3. Lower thresholds by ~10-15x (0.15 → 0.01)
4. Expect more consistent results across different query types

### From RRF to Cosine

1. Update environment variables (see Cosine configuration above)
2. Raise thresholds by ~10-15x (0.01 → 0.15)
3. Monitor score distributions carefully
4. May need dataset-specific threshold tuning

## Production Recommendations

### Default Configuration (RRF Mode)

```bash
# RRF Mode - Production Defaults
export HYBRID_RRF_ENABLED=true
export RRF_LOW_SIM_THRESHOLD=0.01
export ASK_SIM_THRESHOLD=0.01
export HYBRID_RRF_K=60
export HYBRID_RERANKER_BASE=20
export HYBRID_RERANKER_BOOST=45
export HYBRID_ADAPTIVE_THRESHOLD=0.55
export MAX_RERANK_BUDGET_MS=250
```

### Research Configuration (Cosine Mode)

```bash
# Cosine Mode - Research/Tuning
export HYBRID_RRF_ENABLED=false
export RAW_LOW_SIM_THRESHOLD=0.15
export ASK_SIM_THRESHOLD=0.20
export RERANK_SKIP_TOPSIM=0.80
```

### Monitoring Dashboard

Track these metrics in your observability platform:

```
# Grafana/Prometheus examples
activekg_gating_score{mode="rrf_fused"} - histogram
activekg_gating_score{mode="cosine"} - histogram
activekg_rejection_rate{reason="extremely_low_similarity",mode="rrf_fused"} - gauge
activekg_cited_nodes{mode="rrf_fused"} - histogram
activekg_search_latency_ms{mode="hybrid",reranked="true"} - histogram
```

## References

- [RRF Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - Original Reciprocal Rank Fusion algorithm
- `activekg/graph/repository.py:555-650` - RRF implementation
- `activekg/api/main.py:1620-1800` - Gating and metadata logic
- `tests/test_scoring_modes.py` - Comprehensive test coverage
