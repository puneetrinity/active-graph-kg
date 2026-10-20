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
            offloaded_lines = {
                line
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and (
                    isinstance(call.func, ast.Name)
                    and call.func.id == "run_in_threadpool"
                    or isinstance(call.func, ast.Attribute)
                    and call.func.attr == "to_thread"
                )
                for argument in call.args
                if isinstance(argument, ast.Lambda)
                for line in range(argument.lineno, (argument.end_lineno or argument.lineno) + 1)
            }
            # Passing a function reference (to_thread(repo.read, command)) is
            # also deferred. A call expression in its arguments is not.
            offloaded_lines.update(
                line
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and (
                    isinstance(call.func, ast.Name)
                    and call.func.id == "run_in_threadpool"
                    or isinstance(call.func, ast.Attribute)
                    and call.func.attr == "to_thread"
                )
                and not any(
                    isinstance(part, ast.Call)
                    for argument in [*call.args, *(kw.value for kw in call.keywords)]
                    for part in ast.walk(argument)
                )
                for line in range(call.lineno, (call.end_lineno or call.lineno) + 1)
            )
            body = (ast.get_source_segment(source, node) or "").splitlines()[1:]
            for lineno, line in enumerate(body, start=node.lineno + 1):
                if lineno in offloaded_lines:
                    continue
                text = line.strip()
                if text.startswith("#"):
                    continue
                if text.startswith("repository = _repo()"):
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


def test_detector_distinguishes_deferred_work_from_eager_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(__import__(__name__, fromlist=["API"]), "API", tmp_path)
    path = tmp_path / "route.py"
    path.write_text(
        '@router.post("/x")\nasync def x():\n'
        '    return await run_in_threadpool(lambda: timed_work("x", 0,\n'
        "        lambda: _store(payload, claims)))\n"
    )
    assert _offenders() == []
    path.write_text(
        '@router.post("/x")\nasync def x():\n'
        "    return await run_in_threadpool(_store(payload, claims))\n"
    )
    assert len(_offenders()) == 1
