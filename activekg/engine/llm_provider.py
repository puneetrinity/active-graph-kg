"""
LLM Provider for grounded Q&A with citations.

Supports OpenAI-compatible APIs (OpenAI, Azure, local models via LiteLLM).
"""

import os
import time
from collections.abc import Iterable
from typing import Any

from activekg.common.logger import get_enhanced_logger
from activekg.common.metrics import metrics

logger = get_enhanced_logger(__name__)


class LLMProvider:
    """Wrapper for LLM inference with grounding support."""

    def __init__(
        self,
        backend: str = "groq",
        model: str = "mixtral-8x7b-32768",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        """Initialize LLM provider.

        Args:
            backend: "openai", "groq", or "litellm" (for multi-provider support)
            model: Model name
                - OpenAI: "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"
                - Groq: "mixtral-8x7b-32768", "llama2-70b-4096" (ultra-fast inference)
                - LiteLLM: "claude-3-sonnet", "gemini-pro", etc.
            api_key: API key (defaults to env OPENAI_API_KEY, GROQ_API_KEY, or ANTHROPIC_API_KEY)
            base_url: Custom base URL (for Azure, local models, etc.)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Max tokens to generate
        """
        self.backend = backend
        self.model = model

        # Set API key from env if not provided (can be None)
        self.api_key: str | None
        if api_key:
            self.api_key = api_key
        elif backend == "groq":
            self.api_key = os.getenv("GROQ_API_KEY")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

        # Set base URL (can be None)
        self.base_url: str | None
        if base_url:
            self.base_url = base_url
        elif backend == "groq":
            self.base_url = "https://api.groq.com/openai/v1"  # Groq's OpenAI-compatible endpoint
        else:
            self.base_url = None

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.metrics = metrics  # Use global metrics instance

        if self.backend in ["openai", "groq"]:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                logger.info(
                    "LLM provider initialized",
                    extra_fields={
                        "backend": backend,
                        "model": model,
                        "base_url": base_url or "default",
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
            except ImportError:
                raise ImportError(
                    "OpenAI library not installed. Install with: pip install openai"
                ) from None
        elif self.backend == "litellm":
            try:
                import litellm

                self.litellm = litellm
                logger.info(
                    "LLM provider initialized (litellm)",
                    extra_fields={"backend": backend, "model": model},
                )
            except ImportError:
                raise ImportError(
                    "LiteLLM library not installed. Install with: pip install litellm"
                ) from None
        else:
            raise ValueError(f"Unsupported backend: {backend}. Use 'openai', 'groq', or 'litellm'.")

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_message: str | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text completion with metrics tracking.

        Args:
            prompt: User prompt
            max_tokens: Override default max_tokens
            temperature: Override default temperature
            system_message: Optional system message (for instruction following)
            stop: Stop sequences to end generation early (e.g., ["\n\n", "Citations:"])

        Returns:
            Generated text
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        # Default stop sequences for concise, citation-focused answers
        if stop is None:
            stop = ["\n\n\n", "\n---", "\nNote:", "\nDisclaimer:"]

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        # Track metrics
        start_time = time.time()
        status = "success"

        try:
            if self.backend in ["openai", "groq"]:
                # Both OpenAI and Groq use the same SDK (OpenAI-compatible)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )

                # Track token usage
                if hasattr(response, "usage") and response.usage:
                    self.metrics.increment_counter(
                        "llm_tokens_total",
                        value=float(response.usage.prompt_tokens),
                        labels={"backend": self.backend, "model": self.model, "type": "input"},
                    )

                    self.metrics.increment_counter(
                        "llm_tokens_total",
                        value=float(response.usage.completion_tokens),
                        labels={"backend": self.backend, "model": self.model, "type": "output"},
                    )

                return response.choices[0].message.content or ""

            elif self.backend == "litellm":
                response = self.litellm.completion(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                # Track token usage for litellm
                if hasattr(response, "usage") and response.usage:
                    self.metrics.increment_counter(
                        "llm_tokens_total",
                        value=float(response["usage"]["prompt_tokens"]),
                        labels={"backend": self.backend, "model": self.model, "type": "input"},
                    )

                    self.metrics.increment_counter(
                        "llm_tokens_total",
                        value=float(response["usage"]["completion_tokens"]),
                        labels={"backend": self.backend, "model": self.model, "type": "output"},
                    )

                return response.choices[0].message.content or ""

            # This should never happen since __init__ validates backend
            raise RuntimeError(f"Unsupported backend: {self.backend}")

        except Exception as e:
            status = "error"
            logger.error(
                "LLM generation failed",
                extra_fields={"error": str(e), "model": self.model, "prompt_length": len(prompt)},
            )
            raise
        finally:
            # Track latency and request count
            latency_ms = (time.time() - start_time) * 1000

            self.metrics.record_histogram(
                "llm_latency_ms", latency_ms, labels={"backend": self.backend, "model": self.model}
            )

            self.metrics.increment_counter(
                "llm_requests_total",
                labels={"backend": self.backend, "model": self.model, "status": status},
            )

    def generate_with_retry(self, prompt: str, max_retries: int = 3, **kwargs) -> str:
        """Generate with automatic retry on failure.

        Args:
            prompt: User prompt
            max_retries: Max retry attempts
            **kwargs: Additional args passed to generate()

        Returns:
            Generated text
        """
        import time

        last_error = None
        for attempt in range(max_retries):
            try:
                return self.generate(prompt, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM generation failed (attempt {attempt + 1}/{max_retries})",
                    extra_fields={"error": str(e)},
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff

        error_msg = f"LLM generation failed after {max_retries} retries"
        logger.error(
            error_msg,
            extra_fields={"error": str(last_error), "max_retries": max_retries},
        )
        if last_error is not None:
            raise last_error
        raise RuntimeError(error_msg)

    def generate_stream(
        self,
        prompt: str,
        *,
        system_message: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> Iterable[str]:
        """Stream tokens from the LLM as they are generated.

        Yields raw text chunks (token fragments). For OpenAI/Groq backends, uses
        the OpenAI-compatible streaming API. Falls back to non-streaming if the
        backend doesn't support it.
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        # Default stop sequences for concise, citation-focused answers
        if stop is None:
            stop = ["\n\n\n", "\n---", "\nNote:", "\nDisclaimer:"]

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            if self.backend in ["openai", "groq"]:
                # OpenAI-compatible streaming
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    stop=stop,
                )
                for chunk in stream:
                    try:
                        # OpenAI streaming returns ChatCompletionChunk objects
                        delta = chunk.choices[0].delta  # type: ignore[union-attr]
                        piece = getattr(delta, "content", None)
                        if piece:
                            yield piece
                    except Exception:
                        continue
                return

            elif self.backend == "litellm":
                try:
                    for chunk in self.litellm.stream(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ):
                        if hasattr(chunk, "choices"):
                            delta = chunk.choices[0].delta
                            piece = getattr(delta, "content", None)
                            if piece:
                                yield piece
                except Exception as e:
                    logger.warning(
                        "LiteLLM streaming failed, falling back to non-streaming",
                        extra_fields={"error": str(e)},
                    )

        except Exception as e:
            logger.warning(
                "Streaming not available, falling back to single response",
                extra_fields={"error": str(e)},
            )

        # Fallback: single chunk
        text = self.generate(
            prompt, max_tokens=max_tokens, temperature=temperature, system_message=system_message
        )
        if text:
            yield text


def extract_citation_numbers(text: str) -> list[int]:
    """Extract citation numbers like [0], [1], [2] from text.

    Args:
        text: Text with citations (e.g., "According to [0], the answer is [1].")

    Returns:
        List of unique citation indices (e.g., [0, 1])
    """
    import re

    matches = re.findall(r"\[(\d+)\]", text)
    return sorted({int(m) for m in matches})


def calculate_confidence(
    answer: str, search_results: list, citation_indices: list[int], intent_type: str | None = None
) -> float:
    """Calculate confidence score for LLM answer.

    Heuristics:
    - Structured retrieval (intent_type) → high confidence (0.85-0.92)
    - Higher similarity scores → higher confidence
    - More citations used → higher confidence
    - Longer answer → slightly lower confidence (more chance for errors)

    Args:
        answer: LLM-generated answer
        search_results: List of (node, score) tuples
        citation_indices: List of citation indices used in answer
        intent_type: Optional intent type ("open_positions", "performance_issues", etc.)

    Returns:
        Confidence score (0.0-1.0)
    """
    if not search_results or not citation_indices:
        return 0.3  # Low confidence if no results or no citations

    # Structured retrieval uses SQL pattern matching → higher confidence
    # since results are pre-filtered by exact class/property matches
    if intent_type is not None:
        # Base confidence for structured queries: 0.85
        base_confidence = 0.85

        # Boost for citation coverage
        top_k = min(3, len(search_results))
        citation_coverage = len([i for i in citation_indices if i < top_k]) / top_k

        # Small answer length penalty
        length_penalty = max(0.95, 1.0 - (len(answer) / 4000))

        # Structured confidence: 0.85-0.92 range
        confidence = min(0.92, base_confidence + 0.05 * citation_coverage + 0.02 * length_penalty)

        return round(confidence, 3)

    # Semantic search relies on embedding similarity
    # Average similarity of cited nodes
    cited_scores = [search_results[i][1] for i in citation_indices if i < len(search_results)]
    avg_similarity = sum(cited_scores) / len(cited_scores) if cited_scores else 0.5

    # Citation coverage (what % of top-3 results were cited?)
    top_k = min(3, len(search_results))
    citation_coverage = len([i for i in citation_indices if i < top_k]) / top_k

    # Answer length penalty (longer answers have more chance for hallucination)
    length_penalty = max(0.8, 1.0 - (len(answer) / 2000))  # Penalty kicks in after 2K chars

    # Combined confidence
    confidence = 0.5 * avg_similarity + 0.3 * citation_coverage + 0.2 * length_penalty

    # Calibration guards to reduce overconfidence
    # 1. Cap by top similarity (if top result has low similarity, cap confidence)
    top_similarity = search_results[0][1] if search_results else 0.5
    similarity_cap = 0.5 + 0.5 * top_similarity  # Maps [0,1] similarity to [0.5,1.0] confidence cap
    confidence = min(confidence, similarity_cap)

    # 2. Penalize if first citation is not top-1 result (indicates disagreement)
    first_citation_idx = citation_indices[0] if citation_indices else None
    if first_citation_idx is not None and first_citation_idx != 0:
        # Not citing top result → less confident (cap at 0.85)
        confidence = min(confidence, 0.85)

    # 3. Overall cap at 0.95 for semantic queries
    confidence = min(confidence, 0.95)

    return round(confidence, 3)


def filter_context_by_similarity(
    search_results: list,
    similarity_threshold: float = 0.40,  # Lowered default from 0.55
    min_results: int = 2,  # Increased from 1 to ensure context
    max_results: int = 10,
) -> list:
    """Filter search results by similarity threshold for dynamic top-K.

    Only passes high-quality context to LLM, improving answer accuracy.

    Args:
        search_results: List of (node, score) tuples
        similarity_threshold: Minimum similarity to include (default: 0.55)
        min_results: Minimum results to return even if below threshold (default: 1)
        max_results: Maximum results to return (default: 10)

    Returns:
        Filtered list of (node, score) tuples
    """
    if not search_results:
        return []

    # Filter by threshold
    filtered = [(node, score) for node, score in search_results if score >= similarity_threshold]

    # Ensure at least min_results (use top scoring nodes)
    if len(filtered) < min_results:
        filtered = search_results[:min_results]

    # Cap at max_results
    return filtered[:max_results]


def build_strict_citation_prompt(context_items: list[str], question: str) -> tuple[str, str]:
    """Build prompt with strict citation requirements for better accuracy.

    Args:
        context_items: List of formatted context strings
        question: User question

    Returns:
        Tuple of (system_message, user_prompt)
    """
    system_message = (
        "You are a highly accurate assistant that answers questions using ONLY the provided context.\n\n"
        "CITATION RULES:\n"
        "1. ALWAYS cite sources using [0], [1], [2] format for EVERY factual claim\n"
        "2. ONLY cite the highest-similarity contexts that directly support your claim\n"
        "3. Prefer citing [0] over [1], [1] over [2], etc. (higher similarity = more relevant)\n"
        "4. If multiple sources equally support a claim, cite all [0][1]\n\n"
        "ACCURACY RULES:\n"
        "5. Answer directly based on the context provided\n"
        '6. If context is insufficient or ambiguous, say "I don\'t have enough information to answer this"\n'
        "7. Never make assumptions or add information not in the context\n"
        "8. Be concise and specific - focus on answering the question\n\n"
        "EXAMPLES:\n"
        "Q: What products support Bluetooth connectivity?\n"
        "Context: [0] Wireless Noise-Canceling Headphones with Bluetooth 5.0 and 30-hour battery life.\n"
        "A: The Wireless Noise-Canceling Headphones support Bluetooth connectivity, specifically Bluetooth 5.0 [0].\n\n"
        "Q: Who has experience with Python and machine learning?\n"
        "Context: [0] Senior Data Scientist with 5+ years Python, TensorFlow, PyTorch experience.\n"
        "A: The Senior Data Scientist has experience with Python and machine learning, including TensorFlow and PyTorch [0].\n\n"
        "Your responses are evaluated for answer accuracy and citation precision."
    )

    context = "\n\n".join(context_items)

    user_prompt = f"""Context from knowledge graph:
{context}

Question: {question}

Answer (with citations [0], [1], etc.):"""

    return system_message, user_prompt
