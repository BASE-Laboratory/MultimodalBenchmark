"""Unified command-line interface (`mmbench`) for MultimodalBenchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .manifest import load_manifest, verify_cache_directory


def main_evaluate(argv: Optional[Sequence[str]] = None) -> int:
    from .evaluator import main as eval_main

    return eval_main(argv)


def main_audit(argv: Optional[Sequence[str]] = None) -> int:
    from registration.xrdct_reference import main as audit_main

    return audit_main(argv)


def main_splits(argv: Optional[Sequence[str]] = None) -> int:
    from splits.generate_splits import main as splits_main

    return splits_main(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="mmbench", description="MultimodalBenchmark CLI")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_ver = sub.add_parser("verify-manifest", help="Verify cached Zenodo files against MD5 checksums")
    p_ver.add_argument("--manifest", type=Path, default=None)
    p_ver.add_argument("--cache-dir", type=Path, default=Path("data/cache/zenodo"))
    p_ver.add_argument("--download-lightweight", action="store_true", help="Download missing files <= 2 MB")

    args, rest = parser.parse_known_args(argv)
    if args.subcommand == "verify-manifest":
        if rest:
            parser.error(f"Unrecognized arguments: {rest}")
        manifest = load_manifest(args.manifest)
        dl_limit = 2.0 if args.download_lightweight else None
        res = verify_cache_directory(args.cache_dir, manifest=manifest, download_missing_under_mb=dl_limit)
        print(json.dumps(res, indent=2))
        return 0 if res["num_mismatches"] == 0 else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
