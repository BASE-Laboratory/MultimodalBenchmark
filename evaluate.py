#!/usr/bin/env python3
"""Root CLI entrypoint (`python evaluate.py ...`) forwarding to multimodal_benchmark.evaluator."""

from multimodal_benchmark.evaluator import (
    bootstrap_ci_95,
    evaluate_bundles,
    format_markdown_leaderboard_row,
    load_bundle,
    main,
)

__all__ = [
    "bootstrap_ci_95",
    "evaluate_bundles",
    "format_markdown_leaderboard_row",
    "load_bundle",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
