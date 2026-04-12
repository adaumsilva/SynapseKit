"""Tests for eval fine-tune flywheel primitives (issue #515)."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from synapsekit.cli.main import _add_eval_parser, _add_finetune_parser
from synapsekit.cli.test import run_test
from synapsekit.evaluation.dataset import EvalDataset
from synapsekit.evaluation.finetune import (
    FineTuneJob,
    FineTuner,
    OpenAIFineTuneAdapter,
    TogetherFineTuneAdapter,
)
from synapsekit.evaluation.regression import EvalRegression


class TestEvalCaseCaptureIO:
    def test_capture_io_fields_saved_in_snapshot(self, tmp_path):
        eval_file = tmp_path / "eval_sample.py"
        eval_file.write_text(
            """
from synapsekit.evaluation.decorators import eval_case

@eval_case(capture_io=True)
def eval_case_a():
    return {
        \"score\": 0.95,
        \"input\": \"What is the capital of France?\",
        \"output\": \"Paris\",
        \"ideal\": \"Paris\",
    }
""".strip()
        )

        args = SimpleNamespace(
            path=str(eval_file),
            threshold=0.7,
            output_format="json",
            save_snapshot="baseline",
            compare_baseline=None,
            fail_on_regression=False,
            snapshot_dir=str(tmp_path / "snaps"),
        )
        run_test(args)

        reg = EvalRegression(store_dir=args.snapshot_dir)
        snap = reg.load_snapshot("baseline")
        assert len(snap.results) == 1
        row = snap.results[0]
        assert row["input"] == "What is the capital of France?"
        assert row["output"] == "Paris"
        assert row["ideal"] == "Paris"


class TestEvalDataset:
    def test_filter_and_export_openai(self, tmp_path):
        reg = EvalRegression(store_dir=str(tmp_path))
        reg.save_snapshot(
            "baseline",
            [
                {
                    "name": "case_1",
                    "score": 0.6,
                    "input": "Q1",
                    "output": "bad",
                    "ideal": "good",
                },
                {
                    "name": "case_2",
                    "score": 0.9,
                    "input": "Q2",
                    "output": "ok",
                    "ideal": "ok",
                },
            ],
        )

        dataset = EvalDataset.from_snapshot("baseline", snapshot_dir=str(tmp_path))
        weak = dataset.filter_score(max_score=0.8)
        out = tmp_path / "openai.jsonl"
        weak.export(out, format="openai")

        line = out.read_text().strip()
        row = json.loads(line)
        assert row["messages"][1]["content"] == "Q1"
        assert row["messages"][2]["content"] == "good"

    def test_export_anthropic(self, tmp_path):
        reg = EvalRegression(store_dir=str(tmp_path))
        reg.save_snapshot(
            "baseline",
            [{"name": "case", "score": 0.7, "input": "hello", "output": "world"}],
        )

        dataset = EvalDataset.from_snapshot("baseline", snapshot_dir=str(tmp_path))
        out = tmp_path / "anthropic.jsonl"
        dataset.export(out, format="anthropic")

        row = json.loads(out.read_text().strip())
        assert row["prompt"].startswith("Human: hello")
        assert row["completion"].strip() == "world"

    def test_export_dpo_pairs_high_low(self, tmp_path):
        reg = EvalRegression(store_dir=str(tmp_path))
        reg.save_snapshot(
            "baseline",
            [
                {"name": "a", "score": 0.2, "input": "Q", "output": "bad"},
                {"name": "b", "score": 0.9, "input": "Q", "output": "good"},
            ],
        )

        dataset = EvalDataset.from_snapshot("baseline", snapshot_dir=str(tmp_path))
        out = tmp_path / "dpo.jsonl"
        dataset.export(out, format="dpo")

        row = json.loads(out.read_text().strip())
        assert row == {"prompt": "Q", "chosen": "good", "rejected": "bad"}

    def test_export_raises_without_capture_io(self, tmp_path):
        reg = EvalRegression(store_dir=str(tmp_path))
        reg.save_snapshot("baseline", [{"name": "case", "score": 0.8}])

        dataset = EvalDataset.from_snapshot("baseline", snapshot_dir=str(tmp_path))
        with pytest.raises(ValueError, match="capture_io=True"):
            dataset.export(tmp_path / "x.jsonl", format="openai")


class TestFineTuningAdapters:
    async def test_openai_adapter_submit_uses_expected_payload(self):
        calls = []

        def transport(method, url, headers, body):
            calls.append((method, url, headers, body))
            return {"id": "ftjob_1", "status": "validating_files", "model": "gpt-4o-mini"}

        adapter = OpenAIFineTuneAdapter("k", transport=transport)
        job = await adapter.submit(
            dataset="file-123", base_model="gpt-4o-mini", job_name="x", n_epochs=3
        )

        assert job.id == "ftjob_1"
        assert calls[0][0] == "POST"
        assert calls[0][1].endswith("/fine_tuning/jobs")
        assert calls[0][3]["training_file"] == "file-123"
        assert calls[0][3]["model"] == "gpt-4o-mini"

    async def test_together_adapter_submit_uses_expected_payload(self):
        calls = []

        def transport(method, url, headers, body):
            calls.append((method, url, headers, body))
            return {"id": "job_2", "status": "queued", "model": "meta-llama/Llama-3.1-8B-Instruct"}

        adapter = TogetherFineTuneAdapter("k", transport=transport)
        job = await adapter.submit(
            dataset="file-xyz",
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            job_name=None,
            n_epochs=4,
        )

        assert job.id == "job_2"
        assert calls[0][0] == "POST"
        assert calls[0][1].endswith("/fine-tuning/jobs")
        assert calls[0][3]["training_file"] == "file-xyz"

    async def test_finetuner_wait_returns_on_success(self):
        class _Adapter:
            def __init__(self):
                self.calls = 0

            async def submit(self, *, dataset, base_model, job_name, n_epochs):
                return FineTuneJob(id="j", provider="openai", status="created")

            async def status(self, job_id):
                self.calls += 1
                if self.calls < 2:
                    return FineTuneJob(id=job_id, provider="openai", status="running")
                return FineTuneJob(
                    id=job_id, provider="openai", status="succeeded", model_id="ft:model"
                )

        adapter = _Adapter()
        tuner = FineTuner(provider="openai", api_key="k", adapter=adapter)
        job = await tuner.wait("job_1", timeout_s=2, poll_interval_s=0.01)
        assert job.status == "succeeded"
        assert job.model_id == "ft:model"


class TestCLIParsers:
    def test_eval_cli_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        _add_eval_parser(subparsers)

        args = parser.parse_args(
            [
                "eval",
                "export",
                "baseline",
                "--format",
                "openai",
                "--max-score",
                "0.8",
                "--output",
                "data.jsonl",
            ]
        )
        assert args.command == "eval"
        assert args.eval_command == "export"
        assert args.snapshot == "baseline"
        assert args.max_score == 0.8

    def test_finetune_cli_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        _add_finetune_parser(subparsers)

        args = parser.parse_args(
            [
                "finetune",
                "submit",
                "dataset.jsonl",
                "--provider",
                "openai",
                "--base-model",
                "gpt-4o-mini",
            ]
        )
        assert args.command == "finetune"
        assert args.finetune_command == "submit"
        assert args.dataset == "dataset.jsonl"
