"""Async FastAPI routes must not run blocking database work on the event loop.

Regression (job 192, 20 Oct 2026): ingest and privacy-eligibility routes were ``async def`` but called synchronous
psycopg work directly. Each uvicorn worker then served those requests one at a time and froze every other route, so
Discover's deadlines expired while Memory was still working. Blocking calls belong in a plain ``def`` route
(FastAPI threadpool) or behind ``run_in_threadpool`` / ``asyncio.to_thread``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

API = Path(__file__).resolve().parents[1] / "activekg" / "api"
BLOCKING = re.compile(
    r"\b(_store\(|_repo\(\)\.|repository\.|repo\.|_connect\(|psycopg\.connect|\.execute\(|evaluate_many\()"
)
# Known low-traffic admin/upload routes in main.py, documented as a follow-up; do not add to this list casually.
ALLOWED = {("main.py", "upload_files"), ("main.py", "admin_refresh")}
ROUTE_METHODS = {"get", "post", "put", "patch", "delete"}


def _is_route(node: ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(d, ast.Call) and getattr(d.func, "attr", "") in ROUTE_METHODS
        for d in node.decorator_list
    )


def _offenders() -> list[str]:
    found = []
    for path in sorted(API.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.AsyncFunctionDef) or not _is_route(node):
                continue
            if (path.name, node.name) in ALLOWED:
                continue
            body = (ast.get_source_segment(source, node) or "").splitlines()[1:]
            for line in body:
                text = line.strip()
                if text.startswith("#") or "run_in_threadpool" in text or "to_thread" in text:
                    continue
                if text.startswith("lambda:") or text.startswith("repository = _repo()"):
                    continue
                if BLOCKING.search(text):
                    found.append(f"{path.name}:{node.name}: {text[:80]}")
    return found


def test_async_routes_do_not_block_the_event_loop() -> None:
    assert _offenders() == []


def test_detector_catches_a_direct_blocking_call(tmp_path: Path) -> None:
    sample = 'router=None\n@router.post("/x")\nasync def x(request):\n    return _store(payload, claims)\n'
    node = next(n for n in ast.walk(ast.parse(sample)) if isinstance(n, ast.AsyncFunctionDef))
    assert _is_route(node)
    assert BLOCKING.search("return _store(payload, claims)")
