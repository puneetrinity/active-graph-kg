#!/usr/bin/env python3
"""Emit the API contract offline; production OpenAPI HTTP is intentionally retired."""

from __future__ import annotations

import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("ACTIVEKG_TEST_NO_DB", "true")
os.environ.setdefault("JWT_ENABLED", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    # The structured application logger writes its TEST_MODE notice to stdout;
    # keep the generated contract stream JSON-only without muting diagnostics.
    with redirect_stdout(sys.stderr):
        from activekg.api.main import app

    json.dump(app.openapi(), sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
