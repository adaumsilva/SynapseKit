"""``synapsekit eval`` commands: report/export/compare."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from ..evaluation.dataset import EvalDataset
from ..evaluation.regression import EvalRegression
from .test import _print_regression_report


def run_eval(args: Any) -> None:
    subcommand = getattr(args, "eval_command", None)

    if subcommand == "report":
        _run_report(args)
        return

    if subcommand == "export":
        _run_export(args)
        return

    if subcommand == "compare":
        _run_compare(args)
        return

    raise SystemExit("Missing eval subcommand. Use: report, export, or compare")


def _run_report(args: Any) -> None:
    dataset = EvalDataset.from_snapshot(args.snapshot, snapshot_dir=args.snapshot_dir)
    scores = [r.score for r in dataset.records if r.score is not None]

    print(f"Snapshot: {args.snapshot}")
    print(f"Cases: {len(dataset)}")
    if scores:
        print(f"Mean score: {mean(scores):.4f}")
        threshold = getattr(args, "threshold", None)
        if threshold is not None:
            weak = [r for r in dataset.records if r.score is not None and r.score < threshold]
            print(f"Below {threshold:.2f}: {len(weak)}")
            for rec in weak:
                print(f"  - {rec.name}: {rec.score:.4f}")
    else:
        print("No score values found in snapshot.")


def _run_export(args: Any) -> None:
    dataset = EvalDataset.from_snapshot(args.snapshot, snapshot_dir=args.snapshot_dir)
    dataset = dataset.filter_score(min_score=args.min_score, max_score=args.max_score)

    output_path = dataset.export(args.output, format=args.format)
    rows = 0
    with Path(output_path).open("r", encoding="utf-8") as f:
        for _ in f:
            rows += 1

    print(f"Exported {rows} rows -> {output_path}")


def _run_compare(args: Any) -> None:
    reg = EvalRegression(store_dir=args.snapshot_dir)
    report = reg.compare(args.baseline, args.current)
    _print_regression_report(report)
