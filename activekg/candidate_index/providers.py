"""Bounded one-attempt transports; no SDK retries or import-time provider IO.

This file is also the isolated parser/embedding child entrypoint. Its top-level
imports are standard-library only so resource and network fences are installed
before importing parsers, model libraries, or any product module in the child.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

MAX_RESPONSE_BYTES = 65536
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_ORIGINAL_BYTES = 5 * 1024 * 1024
PARSER_SECONDS = 15
PARSER_MEMORY_BYTES = 256 * 1024 * 1024
ML_IMPORT_FILE_BYTES = 1024 * 1024
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


class WorkRefused(Exception):
    """Only closed reason codes cross the worker's logging/storage boundary."""

    def __init__(self, code: str, retry_ms: int = 0) -> None:
        allowed = {
            "privacy_restricted",
            "privacy_review",
            "privacy_unavailable",
            "source_missing",
            "consent_changed",
            "policy_mismatch",
            "low_confidence",
            "unsupported_format",
            "incomplete",
            "chunk_overflow",
            "provider_timeout",
            "provider_unavailable",
            "rate_limited",
            "dispatch_exhausted",
        }
        if code not in allowed or type(retry_ms) is not int or not 0 <= retry_ms <= 120000:
            raise ValueError("candidate_index_error_invalid")
        self.code = code
        self.retry_ms = retry_ms
        super().__init__(code)


def strict_json(raw: str | bytes) -> Any:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise WorkRefused("incomplete")
            value[key] = item
        return value

    def constant(_):
        raise WorkRefused("incomplete")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise WorkRefused("incomplete") from None


def remaining(deadline: float) -> float:
    seconds = deadline - time.monotonic()
    if not math.isfinite(seconds) or seconds <= 0:
        raise WorkRefused("dispatch_exhausted")
    return seconds


def retry_delay(headers: Mapping[str, str], *, wall_time: float | None = None) -> int:
    """Groq documented seconds + duration resets, bounded by the durable clock.

    https://console.groq.com/docs/rate-limits . A reset for a quota that still
    has capacity does not throttle; on 429 without usable hints back off 1s.
    Never clip a long provider wait into permission to retry too early.
    """
    delays = [1.0]
    value = headers.get("retry-after", "")
    if len(value) <= 100:
        try:
            delay = float(value)
            if math.isfinite(delay) and delay >= 0:
                delays.append(delay)
        except ValueError:
            try:
                reset = parsedate_to_datetime(value).timestamp()
                delays.append(max(0, reset - (time.time() if wall_time is None else wall_time)))
            except (ValueError, TypeError, OverflowError):
                pass
    for quota in ("requests", "tokens"):
        if headers.get("x-ratelimit-remaining-" + quota) != "0":
            continue
        value = headers.get("x-ratelimit-reset-" + quota, "")
        if len(value) <= 80 and re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?[hms])+", value):
            units = {"h": 3600, "m": 60, "s": 1}
            delays.append(
                sum(
                    float(n) * units[u]
                    for n, u in re.findall(r"([0-9]+(?:\.[0-9]+)?)([hms])", value)
                )
            )
    seconds = max(delays)
    if seconds > 120:
        raise WorkRefused("dispatch_exhausted")
    return math.ceil(seconds * 1000)


async def extract_once(
    text: str,
    model: str,
    *,
    api_key: str,
    deadline: float,
    before_send: Callable[[], Awaitable[None]],
    transport: Any = None,
) -> dict[str, Any]:
    """One HTTP request, including streamed-body timeout and socket cancellation.

    The caller has already committed a dispatch reservation. The shipped prompt
    is used unmodified, but its truncation branch is refused before dispatch.
    No response/error body is logged. Test substitution is at HTTP transport only.
    """
    import httpx

    from activekg.extraction.prompt import EXTRACTION_MAX_INPUT_CHARS, build_extraction_prompt

    if not api_key:
        raise WorkRefused("provider_unavailable")
    if not isinstance(text, str) or not text.strip() or len(text) > EXTRACTION_MAX_INPUT_CHARS:
        raise WorkRefused("incomplete")
    system, prompt = build_extraction_prompt(text)
    try:
        async with (
            asyncio.timeout(remaining(deadline)),
            httpx.AsyncClient(
                timeout=remaining(deadline),
                follow_redirects=False,
                trust_env=False,
                transport=transport,
            ) as client,
        ):
            request = client.build_request(
                "POST",
                GROQ_ENDPOINT,
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                    "stop": [],
                    "stream": False,
                },
            )
            # Last operation before socket dispatch: no sleep/model/parser after it.
            await before_send()
            remaining(deadline)
            response = await client.send(request, stream=True)
            try:
                if response.status_code == 429:
                    delay = retry_delay(response.headers)
                    if delay / 1000 >= remaining(deadline):
                        raise WorkRefused("dispatch_exhausted")
                    raise WorkRefused("rate_limited", delay)
                if response.status_code != 200:
                    raise WorkRefused("provider_unavailable", 1000)
                if (
                    response.headers.get("content-type", "").split(";")[0] != "application/json"
                    or response.headers.get("content-encoding", "identity") != "identity"
                ):
                    raise WorkRefused("incomplete")
                declared = response.headers.get("content-length")
                if declared is not None and (
                    not re.fullmatch(r"[0-9]{1,9}", declared) or int(declared) > MAX_RESPONSE_BYTES
                ):
                    raise WorkRefused("incomplete")
                raw = bytearray()
                async for chunk in response.aiter_raw():
                    if len(raw) + len(chunk) > MAX_RESPONSE_BYTES:
                        raise WorkRefused("incomplete")
                    raw.extend(chunk)
                envelope = strict_json(bytes(raw))
            finally:
                await response.aclose()
        if (
            not isinstance(envelope, dict)
            or envelope.get("model") != model
            or not isinstance(envelope.get("choices"), list)
            or len(envelope["choices"]) != 1
        ):
            raise WorkRefused("incomplete")
        choice = envelope["choices"][0]
        if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
            raise WorkRefused("incomplete")
        message = choice.get("message")
        if (
            not isinstance(message, dict)
            or message.get("role") != "assistant"
            or message.get("tool_calls")
            or message.get("function_call")
            or message.get("refusal")
            or not isinstance(message.get("content"), str)
        ):
            raise WorkRefused("incomplete")
        result = strict_json(message["content"])
        if not isinstance(result, dict):
            raise WorkRefused("incomplete")
        return result
    except (TimeoutError, httpx.TimeoutException):
        raise WorkRefused("provider_timeout") from None
    except httpx.HTTPError:
        raise WorkRefused("provider_unavailable", 1000) from None


def _child_environment() -> dict[str, str]:
    # No credentials, DSNs, proxy settings, cloud SDK config, or PYTHONPATH.
    return {
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }


async def child_work(mode: str, payload: dict[str, Any], *, deadline: float) -> Any:
    if mode in {"embed", "rerank"}:
        import tempfile

        # Parent owns cleanup, including timeout/cancellation after killing and
        # reaping the child. No candidate bytes are written to this directory.
        with tempfile.TemporaryDirectory(prefix="candidate-index-import-") as scratch:
            return await _child_work(mode, payload, deadline=deadline, scratch=scratch)
    return await _child_work(mode, payload, deadline=deadline)


async def _child_work(
    mode: str, payload: dict[str, Any], *, deadline: float, scratch: str | None = None
) -> Any:
    if mode not in {"parse", "embed", "rerank"}:
        raise WorkRefused("policy_mismatch")
    limit = MAX_TEXT_BYTES + MAX_RESPONSE_BYTES
    request = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode()
    if len(request) > 8 * 1024 * 1024:
        raise WorkRefused("incomplete")
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-B",
        str(Path(__file__).resolve()),
        "--child",
        mode,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env={
            **_child_environment(),
            **({"CANDIDATE_INDEX_IMPORT_DIR": scratch} if scratch else {}),
        },
        limit=65536,
    )
    try:
        async with asyncio.timeout(remaining(deadline)):
            child.stdin.write(request)
            await child.stdin.drain()
            child.stdin.close()
            output = bytearray()
            while block := await child.stdout.read(65536):
                if len(output) + len(block) > limit:
                    raise WorkRefused("incomplete")
                output.extend(block)
            code = await child.wait()
        if code != 0:
            raise WorkRefused("incomplete" if mode == "parse" else "provider_unavailable")
        result = strict_json(bytes(output))
        if mode in {"embed", "rerank"} and isinstance(result, dict) and "value" in result:
            if result.pop("fence", None) != {
                "network_attempts": 0,
                "processing_file_limit": [0, 0],
                "import_file_limit": ML_IMPORT_FILE_BYTES,
            }:
                raise WorkRefused("provider_unavailable")
        if not isinstance(result, dict) or set(result) not in ({"value"}, {"error"}):
            raise WorkRefused("incomplete")
        if "error" in result:
            if result["error"] not in {
                "incomplete",
                "unsupported_format",
                "provider_unavailable",
                "chunk_overflow",
            }:
                raise WorkRefused("incomplete")
            raise WorkRefused(result["error"])
        return result["value"]
    except TimeoutError:
        raise WorkRefused("provider_timeout") from None
    finally:
        # Cancellation, output overflow and broken pipes all reap the child.
        if child.returncode is None:
            try:
                child.kill()
            except ProcessLookupError:
                pass
        await child.wait()


async def parse_original(encoded: str, *, deadline: float) -> str:
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_ORIGINAL_BYTES + 2) // 3):
        raise WorkRefused("incomplete")
    result = await child_work(
        "parse",
        {"original_bytes": encoded},
        deadline=min(deadline, time.monotonic() + PARSER_SECONDS),
    )
    if not isinstance(result, str) or not result.strip() or len(result.encode()) > MAX_TEXT_BYTES:
        raise WorkRefused("incomplete")
    return result


def cached_embedding_path(model: str, revision: str, *, cache_dir: Path | None = None) -> Path:
    """Resolve an exact existing HF snapshot, never a moving revision or download."""
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)?", model) or not re.fullmatch(
        r"[0-9a-f]{40}|[0-9a-f]{64}", revision
    ):
        raise WorkRefused("policy_mismatch")
    if cache_dir is None:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_dir = Path(HF_HUB_CACHE)
    repo = model if "/" in model else "sentence-transformers/" + model
    snapshot = cache_dir / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if snapshot.is_symlink() or not snapshot.is_dir() or not (snapshot / "modules.json").is_file():
        raise WorkRefused("provider_unavailable")
    return snapshot.resolve()


async def embed_once(
    texts: list[str], model: str, revision: str, *, deadline: float
) -> list[list[float]]:
    from activekg.candidate_index.contracts import validate_vectors

    if (
        not isinstance(texts, list)
        or not 1 <= len(texts) <= 32
        or any(not isinstance(t, str) or not 1 <= len(t) <= 1200 for t in texts)
    ):
        raise WorkRefused("incomplete")
    path = cached_embedding_path(model, revision)
    vectors = await child_work(
        "embed", {"texts": texts, "path": str(path), "revision": revision}, deadline=deadline
    )
    return validate_vectors(vectors, len(texts))


async def embed_query_once(
    query: str, model: str, revision: str, *, deadline: float
) -> list[float]:
    """Same immutable vector space; a long query is refused, never silently truncated."""
    from activekg.candidate_index.contracts import validate_vectors

    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise WorkRefused("incomplete")
    path = cached_embedding_path(model, revision)
    vectors = await child_work(
        "embed", {"texts": [query], "path": str(path), "revision": revision}, deadline=deadline
    )
    return validate_vectors(vectors, 1)[0]


async def rerank_once(query: str, documents: list[str], *, deadline: float) -> list[float]:
    """The frozen graph reader's local cross-encoder, in a cancellable offline child.

    Only an already cached snapshot is used. No implicit model purchase/download,
    API credential, database handle or arbitrary model choice crosses this boundary.
    An unavailable cache is an honest ranking fallback at the caller, not an error
    body and not permission to retry or fetch a model.
    """
    if (
        not isinstance(query, str)
        or not query.strip()
        or len(query) > 2000
        or not isinstance(documents, list)
        or not 1 <= len(documents) <= 200
        or any(not isinstance(doc, str) or len(doc) > 512 for doc in documents)
    ):
        raise WorkRefused("incomplete")
    from huggingface_hub.constants import HF_HUB_CACHE

    root = Path(HF_HUB_CACHE) / "models--cross-encoder--ms-marco-MiniLM-L-6-v2"
    try:
        ref = root / "refs" / "main"
        if ref.is_symlink() or ref.stat().st_size > 64:
            raise ValueError()
        revision = ref.read_text().strip()
        path = root / "snapshots" / revision
        if (
            re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None
            or path.is_symlink()
            or not path.is_dir()
            or not (path / "config.json").is_file()
        ):
            raise ValueError()
    except (OSError, ValueError):
        raise WorkRefused("provider_unavailable") from None
    result = await child_work(
        "rerank",
        {"path": str(path.resolve()), "revision": revision, "query": query, "documents": documents},
        deadline=deadline,
    )
    if (
        not isinstance(result, list)
        or len(result) != len(documents)
        or any(type(score) not in (float, int) or not math.isfinite(score) for score in result)
    ):
        raise WorkRefused("incomplete")
    return [float(score) for score in result]


def _parse_document(encoded: Any) -> str:
    """Executed only inside the memory-limited, no-network parser process."""
    import io
    import zipfile

    if not isinstance(encoded, str) or len(encoded) > 6990508:
        raise WorkRefused("incomplete")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError:
        raise WorkRefused("incomplete") from None
    if not 1 <= len(raw) <= MAX_ORIGINAL_BYTES or base64.b64encode(raw).decode() != encoded:
        raise WorkRefused("incomplete")
    if raw.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        raise WorkRefused("unsupported_format")
    parts: list[str] = []
    size = 0

    def append(text):
        nonlocal size
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            raise WorkRefused("incomplete")
        size += len(text.encode("utf-8", errors="strict")) + 1
        if size > MAX_TEXT_BYTES:
            raise WorkRefused("incomplete")
        parts.append(text)

    if raw.startswith(b"%PDF-"):
        if not raw.rstrip().endswith(b"%%EOF"):
            raise WorkRefused("incomplete")
        import pdfplumber

        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            if not pdf.doc.is_extractable or pdf.doc.encryption or not 1 <= len(pdf.pages) <= 500:
                raise WorkRefused("incomplete")
            for page in pdf.pages:
                # No swallowed page errors, image-only pages, or partial success.
                append(page.extract_text())
    elif raw.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            names = [i.filename for i in infos]
            if (
                not 1 <= len(infos) <= 512
                or len(set(names)) != len(names)
                or not {"[Content_Types].xml", "word/document.xml"}.issubset(names)
                or any(
                    i.flag_bits & 1 or i.filename.startswith("/") or ".." in i.filename.split("/")
                    for i in infos
                )
                or sum(i.file_size for i in infos) > 16 * 1024 * 1024
                or any(i.file_size > 8 * 1024 * 1024 for i in infos)
            ):
                raise WorkRefused("incomplete")
            # Unsupported alternate content cannot be silently omitted.
            if any(n.startswith(("word/embeddings/", "word/altChunk")) for n in names):
                raise WorkRefused("unsupported_format")
            from docx import Document
            from lxml import etree

            Document(io.BytesIO(raw))  # Existing parser's structural validation.
            parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            sections = ["word/document.xml"] + sorted(
                n
                for n in names
                if re.fullmatch(r"word/(?:header\d+|footer\d+|footnotes|endnotes)\.xml", n)
            )
            for name in sections:
                section = archive.read(name)
                if b"<!DOCTYPE" in section.upper() or b"<!ENTITY" in section.upper():
                    raise WorkRefused("incomplete")
                root = etree.fromstring(section, parser=parser)
                if root.xpath(".//w:altChunk | .//w:del | .//w:ins", namespaces=ns):
                    raise WorkRefused("unsupported_format")
                # Paragraphs inside tables/text boxes are included. No paragraphs-only shortcut.
                for paragraph in root.xpath(".//w:p", namespaces=ns):
                    text = "".join(paragraph.xpath(".//w:t/text()", namespaces=ns))
                    if text.strip():
                        append(text)
            if not parts:
                raise WorkRefused("incomplete")
    else:
        raise WorkRefused("unsupported_format")
    return "\n".join(parts)


def _run_child(mode: str) -> None:
    import logging
    import resource
    import socket

    logging.disable(logging.CRITICAL)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if mode == "parse":
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    if mode == "parse":
        resource.setrlimit(resource.RLIMIT_AS, (PARSER_MEMORY_BYTES, PARSER_MEMORY_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU, (PARSER_SECONDS, PARSER_SECONDS))
    forbidden = 0

    def refuse(*_args, **_kwargs):
        nonlocal forbidden
        forbidden += 1
        raise WorkRefused("provider_unavailable")

    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.create_connection = refuse
    socket.getaddrinfo = refuse
    # -I discards cwd/PYTHONPATH. Only this installed product root is admitted.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        if mode in {"embed", "rerank"}:
            import os
            import stat
            import tempfile

            scratch = Path(os.environ["CANDIDATE_INDEX_IMPORT_DIR"])
            info = scratch.lstat()
            if (
                not scratch.is_absolute()
                or not stat.S_ISDIR(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o700
                or info.st_uid != os.getuid()
            ):
                raise WorkRefused("provider_unavailable")
            os.chdir(scratch)
            tempfile.tempdir = str(scratch)
            # This is a per-file kernel limit, not an aggregate disk quota.
            # Trusted imports only: stdin remains unread until the hard limit
            # is irreversibly lowered to zero below.
            resource.setrlimit(resource.RLIMIT_FSIZE, (ML_IMPORT_FILE_BYTES, ML_IMPORT_FILE_BYTES))
            import sentence_transformers  # noqa: F401
            import torch  # noqa: F401

            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
            if forbidden:
                raise WorkRefused("provider_unavailable")
        raw = sys.stdin.buffer.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise WorkRefused("incomplete")
        payload = strict_json(raw)
        if mode == "parse" and isinstance(payload, dict) and set(payload) == {"original_bytes"}:
            value = _parse_document(payload["original_bytes"])
        elif (
            mode == "embed"
            and isinstance(payload, dict)
            and set(payload) == {"path", "texts", "revision"}
        ):
            path = Path(payload["path"])
            if (
                not path.is_absolute()
                or path.name != payload["revision"]
                or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", path.name)
                or path.parent.name != "snapshots"
                or not path.is_dir()
            ):
                raise WorkRefused("provider_unavailable")
            from activekg.engine.embedding_provider import EmbeddingProvider

            # Local revision path, separate instance/process, unchanged frozen encoder.
            provider = EmbeddingProvider(backend="sentence-transformers", model_name=str(path))
            provider._ensure_model()
            # Sentence-transformers otherwise truncates long token sequences
            # silently even when the character/chunk manifest is complete.
            tokenizer = provider._model.tokenizer
            token_limit = provider._model.max_seq_length
            if (
                type(token_limit) is not int
                or token_limit <= 0
                or any(
                    len(tokenizer.encode(text, truncation=False, add_special_tokens=True))
                    > token_limit
                    for text in payload["texts"]
                )
            ):
                raise WorkRefused("chunk_overflow")
            value = provider.encode(payload["texts"]).tolist()
        elif (
            mode == "rerank"
            and isinstance(payload, dict)
            and set(payload) == {"path", "revision", "query", "documents"}
        ):
            path = Path(payload["path"])
            if (
                not path.is_absolute()
                or path.name != payload["revision"]
                or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", path.name)
                or path.parent.name != "snapshots"
                or path.parent.parent.name != "models--cross-encoder--ms-marco-MiniLM-L-6-v2"
                or not path.is_dir()
                or not isinstance(payload["query"], str)
                or not 1 <= len(payload["query"]) <= 2000
                or not isinstance(payload["documents"], list)
                or not 1 <= len(payload["documents"]) <= 200
                or any(not isinstance(doc, str) or len(doc) > 512 for doc in payload["documents"])
            ):
                raise WorkRefused("incomplete")
            import torch
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(
                str(path),
                device="cpu",
                automodel_args={
                    "device_map": None,
                    "low_cpu_mem_usage": False,
                    "dtype": torch.float32,
                },
            )
            value = model.predict(
                [[payload["query"], doc] for doc in payload["documents"]]
            ).tolist()
        else:
            raise WorkRefused("incomplete")
        if forbidden:
            raise WorkRefused("provider_unavailable")
        result = {"value": value}
        if mode in {"embed", "rerank"}:
            if resource.getrlimit(resource.RLIMIT_FSIZE) != (0, 0):
                raise WorkRefused("provider_unavailable")
            result["fence"] = {
                "network_attempts": forbidden,
                "processing_file_limit": [0, 0],
                "import_file_limit": ML_IMPORT_FILE_BYTES,
            }
    except WorkRefused as exc:
        result = {
            "error": exc.code
            if exc.code in {"incomplete", "unsupported_format", "chunk_overflow"}
            else "provider_unavailable"
        }
    except Exception:
        result = {"error": "incomplete" if mode == "parse" else "provider_unavailable"}
    sys.stdout.write(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    if (
        len(sys.argv) != 3
        or sys.argv[1] != "--child"
        or sys.argv[2] not in {"parse", "embed", "rerank"}
    ):
        raise SystemExit(2)
    _run_child(sys.argv[2])
