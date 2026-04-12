"""``synapsekit finetune`` commands: submit/status/wait."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from ..evaluation.finetune import FineTuner


async def _submit(args: Any) -> None:
    api_key = _resolve_api_key(args)
    tuner = FineTuner(provider=args.provider, api_key=api_key)
    job = await tuner.submit(
        dataset=args.dataset,
        base_model=args.base_model,
        job_name=args.job_name,
        n_epochs=args.n_epochs,
    )
    print(f"Job created: {job.id}")
    print(f"Status: {job.status}")
    if job.model_id:
        print(f"Model: {job.model_id}")


async def _status(args: Any) -> None:
    api_key = _resolve_api_key(args)
    tuner = FineTuner(provider=args.provider, api_key=api_key)
    job = await tuner.status(args.job_id)
    print(f"Job: {job.id}")
    print(f"Status: {job.status}")
    if job.model_id:
        print(f"Model: {job.model_id}")
    if job.error:
        print(f"Error: {job.error}")


async def _wait(args: Any) -> None:
    api_key = _resolve_api_key(args)
    tuner = FineTuner(provider=args.provider, api_key=api_key)
    job = await tuner.wait(args.job_id, timeout_s=args.timeout, poll_interval_s=args.interval)
    print(f"Job completed: {job.id}")
    print(f"Status: {job.status}")
    if job.model_id:
        print(f"Model: {job.model_id}")


def run_finetune(args: Any) -> None:
    subcommand = getattr(args, "finetune_command", None)

    if subcommand == "submit":
        asyncio.run(_submit(args))
        return
    if subcommand == "status":
        asyncio.run(_status(args))
        return
    if subcommand == "wait":
        asyncio.run(_wait(args))
        return

    raise SystemExit("Missing finetune subcommand. Use: submit, status, or wait")


def _resolve_api_key(args: Any) -> str:
    raw_api_key = getattr(args, "api_key", None)
    if raw_api_key:
        return str(raw_api_key)

    provider = str(getattr(args, "provider", "")).lower()
    if provider == "openai":
        env_key = os.getenv("OPENAI_API_KEY")
    elif provider == "together":
        env_key = os.getenv("TOGETHER_API_KEY")
    else:
        env_key = None

    if env_key:
        return env_key

    raise ValueError(
        f"Missing API key for provider '{provider}'. "
        "Pass --api-key or set the provider environment variable."
    )
