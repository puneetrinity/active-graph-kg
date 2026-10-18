"""Explicit owner recovery CLI. Plan files are private; no model calls.

plan: --scope org_N --selection file --plan file
apply: --plan file --approve-hash HASH --max-calls N
Both require --target-id UUID and configured pinned candidate-index policy.
ACTIVEKG_MIGRATE_DSN is read from the environment, never printed.
Plan output is created exclusively with mode 0600. Apply retains it on failure;
after confirmed commit removes that exact file. Lost replies retain it for replay.
No automatic sweep or model credentials used. Keep plan files out of source control.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

from activekg.candidate_index.admission import configuration
from activekg.candidate_index.recovery import ResumeRecovery


def read_private(path: str) -> tuple[object, os.stat_result]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_size > 65536
        ):
            raise ValueError("recovery_file_permissions")
        return json.loads(stream.read(65537)), info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "apply"])
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--scope")
    parser.add_argument("--selection")
    parser.add_argument("--approve-hash")
    parser.add_argument("--max-calls", type=int)
    args = parser.parse_args()
    config = configuration()
    service = ResumeRecovery(
        os.environ["ACTIVEKG_MIGRATE_DSN"], args.target_id, config.policy, config.tunables
    )
    if args.action == "plan":
        if (
            not args.scope
            or not args.selection
            or args.approve_hash is not None
            or args.max_calls is not None
        ):
            raise ValueError("recovery_arguments")
        pairs, _ = read_private(args.selection)
        result = service.plan(args.scope, pairs)
        fd = os.open(args.plan, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(result, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        print(
            json.dumps(
                {
                    "planned": True,
                    "plan_hash": result["plan_hash"],
                    "max_calls": result["plan"]["max_calls"],
                }
            )
        )
    else:
        if (
            not args.approve_hash
            or args.max_calls is None
            or args.scope is not None
            or args.selection is not None
        ):
            raise ValueError("recovery_arguments")
        plan, info = read_private(args.plan)
        result = service.apply(plan, args.approve_hash, args.max_calls)
        current = os.stat(args.plan, follow_symlinks=False)
        if (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino):
            Path(args.plan).unlink()
        print(
            json.dumps(
                {"applied": True, "replayed": result["replayed"], "max_calls": result["max_calls"]}
            )
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(
            "Recovery refused or unconfirmed; retain the plan and investigate. No raw error printed.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
