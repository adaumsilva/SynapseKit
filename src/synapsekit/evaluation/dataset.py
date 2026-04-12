"""Eval dataset helpers for filtering and exporting fine-tuning data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .regression import EvalRegression


@dataclass
class EvalRecord:
    """Single eval result record."""

    name: str
    score: float | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    input: str | None = None
    output: str | None = None
    ideal: str | None = None
    raw: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalRecord:
        return cls(
            name=str(data.get("name", "")),
            score=_to_float(data.get("score")),
            cost_usd=_to_float(data.get("cost_usd")),
            latency_ms=_to_float(data.get("latency_ms")),
            input=_to_str_or_none(data.get("input")),
            output=_to_str_or_none(data.get("output")),
            ideal=_to_str_or_none(data.get("ideal")),
            raw=data,
        )

    @property
    def assistant_target(self) -> str | None:
        """Preferred assistant target for SFT exports."""
        return self.ideal if self.ideal is not None else self.output


class EvalDataset:
    """A filterable/exportable collection of evaluation records."""

    def __init__(self, records: list[EvalRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def from_snapshot(cls, name: str, snapshot_dir: str = ".synapsekit_evals") -> EvalDataset:
        reg = EvalRegression(store_dir=snapshot_dir)
        snap = reg.load_snapshot(name)
        return cls([EvalRecord.from_dict(r) for r in snap.results])

    def filter(self, predicate: Any) -> EvalDataset:
        return EvalDataset([r for r in self.records if predicate(r)])

    def filter_score(
        self,
        *,
        min_score: float | None = None,
        max_score: float | None = None,
    ) -> EvalDataset:
        def _ok(rec: EvalRecord) -> bool:
            if rec.score is None:
                return False
            if min_score is not None and rec.score < min_score:
                return False
            return not (max_score is not None and rec.score > max_score)

        return self.filter(_ok)

    def export(
        self,
        output: str | Path,
        *,
        format: str = "openai",
        system_prompt: str | None = "You are a helpful assistant.",
    ) -> Path:
        fmt = format.lower()
        output_path = Path(output)

        if fmt in {"openai", "anthropic", "together", "jsonl"}:
            rows = self._export_supervised_rows(fmt, system_prompt=system_prompt)
        elif fmt == "dpo":
            rows = self._export_dpo_rows()
        else:
            raise ValueError(f"Unsupported export format: {format}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return output_path

    def _export_supervised_rows(
        self,
        fmt: str,
        *,
        system_prompt: str | None,
    ) -> list[dict[str, Any]]:
        self._require_io_fields()

        rows: list[dict[str, Any]] = []
        for rec in self.records:
            if rec.input is None:
                continue
            target = rec.assistant_target
            if target is None:
                continue

            if fmt == "openai":
                messages: list[dict[str, str]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.extend(
                    [
                        {"role": "user", "content": rec.input},
                        {"role": "assistant", "content": target},
                    ]
                )
                rows.append({"messages": messages})
            elif fmt == "anthropic":
                rows.append(
                    {
                        "prompt": f"Human: {rec.input}\\nAssistant:",
                        "completion": f" {target}",
                    }
                )
            else:  # together/jsonl generic
                rows.append({"prompt": rec.input, "completion": target})

        return rows

    def _export_dpo_rows(self) -> list[dict[str, Any]]:
        self._require_io_fields()

        by_input: dict[str, list[EvalRecord]] = {}
        for rec in self.records:
            if rec.input is None or rec.output is None or rec.score is None:
                continue
            by_input.setdefault(rec.input, []).append(rec)

        rows: list[dict[str, Any]] = []
        for prompt, records in by_input.items():
            if len(records) < 2:
                continue
            ordered = sorted(records, key=lambda r: r.score if r.score is not None else -1.0)
            rejected = ordered[0]
            chosen = ordered[-1]
            if chosen.output is None or rejected.output is None:
                continue
            if chosen.output == rejected.output:
                continue
            rows.append({"prompt": prompt, "chosen": chosen.output, "rejected": rejected.output})

        return rows

    def _require_io_fields(self) -> None:
        has_input = any(r.input is not None for r in self.records)
        has_output = any(r.output is not None for r in self.records)
        if not has_input or not has_output:
            raise ValueError(
                "Snapshot is missing captured input/output fields. "
                "Run evals with @eval_case(capture_io=True) and return input/output in the case result."
            )


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
