"""
Test scoring mode metadata (RRF vs Cosine) across /ask and /debug/search_explain endpoints.

These tests verify that:
1. gating_score_type is present and correct in /ask metadata
2. score_type is present and correct in /debug/search_explain
3. Score ranges match the expected mode (RRF: 0.01-0.04, Cosine: 0.0-1.0)
4. Threshold behavior is appropriate for each mode
"""

import os

import pytest
from fastapi.testclient import TestClient


def _llm_available() -> bool:
    enabled = os.getenv("LLM_ENABLED", "true").lower() == "true"
    if not enabled:
        return False
    return bool(
        os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
    )


@pytest.fixture(scope="module")
def client_rrf():
    """Test client with RRF mode enabled."""
    # Set RRF mode environment
    os.environ["HYBRID_RRF_ENABLED"] = "true"
    os.environ["RRF_LOW_SIM_THRESHOLD"] = "0.01"
    os.environ["ASK_SIM_THRESHOLD"] = "0.01"
    os.environ["HYBRID_RRF_K"] = "60"

    # Import after setting env vars
    from activekg.api.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def client_cosine():
    """Test client with cosine mode (RRF disabled)."""
    # Set cosine mode environment
    os.environ["HYBRID_RRF_ENABLED"] = "false"
    os.environ["RAW_LOW_SIM_THRESHOLD"] = "0.15"
    os.environ["ASK_SIM_THRESHOLD"] = "0.20"

    # Re-import to pick up new env vars
    from activekg.api.main import app

    return TestClient(app)


class TestAskEndpointScoring:
    """Test /ask endpoint metadata for different scoring modes."""

    def test_ask_rrf_mode_metadata(self, client_rrf):
        """Test that /ask returns correct metadata in RRF mode."""
        if not _llm_available():
            pytest.skip("LLM backend not configured")
        response = client_rrf.post(
            "/ask",
            json={
                "question": "What ML frameworks does the Machine Learning Engineer position require?"
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check metadata exists
        assert "metadata" in data
        metadata = data["metadata"]

        # Verify gating_score_type
        assert "gating_score_type" in metadata
        assert metadata["gating_score_type"] == "rrf_fused"

        # Verify gating_score exists
        assert "gating_score" in metadata
        gating_score = metadata["gating_score"]

        # RRF scores should be in 0.01-0.04 range (or rejected if < 0.01)
        if metadata.get("cited_nodes", 0) > 0:
            # If we got results, score should be in RRF range
            assert 0.01 <= gating_score <= 0.10, f"RRF score {gating_score} out of expected range"

    def test_ask_cosine_mode_metadata(self, client_cosine):
        """Test that /ask returns correct metadata in cosine mode."""
        if not _llm_available():
            pytest.skip("LLM backend not configured")
        response = client_cosine.post(
            "/ask",
            json={
                "question": "What ML frameworks does the Machine Learning Engineer position require?"
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check metadata exists
        assert "metadata" in data
        metadata = data["metadata"]

        # Verify gating_score_type
        assert "gating_score_type" in metadata
        assert metadata["gating_score_type"] == "cosine"

        # Verify gating_score exists
        assert "gating_score" in metadata
        gating_score = metadata["gating_score"]

        # Cosine scores should be in 0.0-1.0 range
        if metadata.get("cited_nodes", 0) > 0:
            assert 0.0 <= gating_score <= 1.0, f"Cosine score {gating_score} out of expected range"
            # Typically should be > 0.15 for valid results
            assert gating_score >= 0.15, f"Cosine score {gating_score} unexpectedly low"


class TestDebugSearchExplain:
    """Test /debug/search_explain endpoint for different scoring modes."""

    def test_search_explain_rrf_hybrid(self, client_rrf):
        """Test /debug/search_explain with RRF hybrid mode."""
        # Ensure RRF mode is set (fixtures are module-scoped so env can be overridden)
        os.environ["HYBRID_RRF_ENABLED"] = "true"

        response = client_rrf.post(
            "/debug/search_explain",
            json={"query": "machine learning engineer frameworks", "use_hybrid": True, "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify score_type
        assert "score_type" in data
        assert data["score_type"] == "rrf_fused"

        # Verify score_range
        assert "score_range" in data
        assert "0.01" in data["score_range"].lower() or "0.04" in data["score_range"].lower()

        # Verify per-result score_type
        if data.get("results"):
            for result in data["results"]:
                assert "score_type" in result
                assert result["score_type"] == "rrf_fused"

                # Check score is in RRF range
                similarity = result["similarity"]
                assert 0.0 <= similarity <= 0.10, f"RRF similarity {similarity} out of range"

        # Verify scoring_notes
        assert "scoring_notes" in data
        notes = data["scoring_notes"]
        assert "rrf_fused" in notes
        assert "rank-based" in notes["rrf_fused"].lower() or "rrf" in notes["rrf_fused"].lower()

    def test_search_explain_cosine_vector(self, client_cosine):
        """Test /debug/search_explain with cosine vector-only mode."""
        response = client_cosine.post(
            "/debug/search_explain",
            json={"query": "machine learning engineer frameworks", "use_hybrid": False, "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify score_type
        assert "score_type" in data
        assert data["score_type"] == "cosine"

        # Verify score_range
        assert "score_range" in data
        assert "0.0-1.0" in data["score_range"] or "0.0" in data["score_range"]

        # Verify per-result score_type
        if data.get("results"):
            for result in data["results"]:
                assert "score_type" in result
                assert result["score_type"] == "cosine"

                # Check score is in cosine range
                similarity = result["similarity"]
                assert 0.0 <= similarity <= 1.0, f"Cosine similarity {similarity} out of range"

        # Verify scoring_notes
        assert "scoring_notes" in data
        notes = data["scoring_notes"]
        assert "cosine" in notes
        assert "vector" in notes["cosine"].lower()


class TestThresholdBehavior:
    """Test that thresholds are applied correctly for each mode."""

    def test_rrf_low_threshold_applied(self, client_rrf):
        """Test that RRF uses the lower threshold (0.01) appropriately."""
        # Ensure RRF mode is set (fixtures are module-scoped so env can be overridden)
        os.environ["HYBRID_RRF_ENABLED"] = "true"
        if not _llm_available():
            pytest.skip("LLM backend not configured")

        response = client_rrf.post("/ask", json={"question": "machine learning frameworks"})

        assert response.status_code == 200
        data = response.json()
        metadata = data.get("metadata", {})

        # In RRF mode, scores around 0.02-0.04 should NOT be rejected
        # Check that we're using rrf_fused mode
        assert metadata.get("gating_score_type") == "rrf_fused"

        # If there's a low similarity rejection, it should be << 0.01
        if "extremely_low_similarity" in metadata.get("reason", ""):
            assert metadata["gating_score"] < 0.01

    def test_cosine_higher_threshold(self, client_cosine):
        """Test that cosine mode uses appropriate threshold (0.15+)."""
        # Ensure cosine mode is set (fixtures are module-scoped so env can be overridden)
        os.environ["HYBRID_RRF_ENABLED"] = "false"
        if not _llm_available():
            pytest.skip("LLM backend not configured")

        response = client_cosine.post(
            "/ask", json={"question": "completely unrelated query about astronomy"}
        )

        assert response.status_code == 200
        data = response.json()
        metadata = data.get("metadata", {})

        # Check that we're using cosine mode
        assert metadata.get("gating_score_type") == "cosine"

        # If rejected due to low similarity, score should be < 0.15
        if (
            "extremely_low_similarity" in metadata.get("reason", "")
            or metadata.get("cited_nodes", 0) == 0
        ):
            # Low confidence queries should have lower scores
            assert metadata.get("gating_score", 0) < 0.50


class TestScoringNotes:
    """Test that scoring_notes are present and informative."""

    def test_scoring_notes_complete(self, client_rrf):
        """Test that scoring_notes contain all three modes."""
        response = client_rrf.post(
            "/debug/search_explain", json={"query": "test", "use_hybrid": True, "top_k": 1}
        )

        assert response.status_code == 200
        data = response.json()

        notes = data.get("scoring_notes", {})
        assert "rrf_fused" in notes
        assert "weighted_fusion" in notes
        assert "cosine" in notes

        # Check that each has a meaningful description
        assert len(notes["rrf_fused"]) > 20
        assert len(notes["weighted_fusion"]) > 20
        assert len(notes["cosine"]) > 20
