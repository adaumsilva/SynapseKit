"""EvalCI fine-tune flywheel example.

Run evals first:
    synapsekit test tests/evals --save baseline

Then:
    synapsekit eval export baseline --format openai --max-score 0.8 --output dataset.jsonl
    synapsekit finetune submit dataset.jsonl --provider openai --base-model gpt-4o-mini
"""

from __future__ import annotations

import asyncio

from synapsekit.evaluation import EvalDataset, FineTuner


async def main() -> None:
    dataset = EvalDataset.from_snapshot("baseline")
    weak = dataset.filter_score(max_score=0.8)
    weak.export("dataset.jsonl", format="openai")

    tuner = FineTuner(provider="openai", api_key="YOUR_OPENAI_API_KEY")
    job = await tuner.submit(
        dataset="dataset.jsonl",
        base_model="gpt-4o-mini",
        job_name="rag-improvement-v1",
        n_epochs=3,
    )
    print("Submitted:", job)

    final = await tuner.wait(job.id, timeout_s=7200, poll_interval_s=15)
    print("Completed:", final)


if __name__ == "__main__":
    asyncio.run(main())
