#!/usr/bin/env python3
"""Read-only schema admission for API and worker startup."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from activekg.common.schema_control import (  # noqa: E402
    SchemaControlError,
    assert_startup_schema_ready,
)


def main() -> None:
    try:
        assert_startup_schema_ready()
    except SchemaControlError as exc:
        print(f"[Schema ready] REFUSED ({type(exc).__name__})", file=sys.stderr)
        raise SystemExit(1) from exc
    print("[Schema ready] OK")


if __name__ == "__main__":
    main()
