"""Fine-tuning orchestration and provider adapters."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class FineTuneJob:
    """Provider fine-tuning job state."""

    id: str
    provider: str
    status: str
    base_model: str | None = None
    model_id: str | None = None
    error: str | None = None


class FineTuneAdapter:
    """Provider adapter protocol."""

    async def submit(
        self,
        *,
        dataset: str,
        base_model: str,
        job_name: str | None,
        n_epochs: int,
    ) -> FineTuneJob:
        raise NotImplementedError

    async def status(self, job_id: str) -> FineTuneJob:
        raise NotImplementedError


Transport = Callable[[str, str, dict[str, str], dict[str, Any] | None], dict[str, Any]]


def _http_json_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any] | None,
) -> dict[str, Any]:
    data = None
    req_headers = {"Content-Type": "application/json", **headers}
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    request = urllib.request.Request(url=url, method=method, headers=req_headers, data=data)
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            payload = resp.read().decode("utf-8")
            if not payload:
                return {}
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise RuntimeError("Unexpected API response shape")
            return parsed
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Provider API request failed ({exc.code}): {details}") from exc


class OpenAIFineTuneAdapter(FineTuneAdapter):
    """OpenAI fine-tuning API adapter."""

    def __init__(
        self,
        api_key: str,
        *,
        api_base: str = "https://api.openai.com/v1",
        transport: Transport | None = None,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._transport = transport or _http_json_transport
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def submit(
        self,
        *,
        dataset: str,
        base_model: str,
        job_name: str | None,
        n_epochs: int,
    ) -> FineTuneJob:
        payload: dict[str, Any] = {
            "training_file": dataset,
            "model": base_model,
            "hyperparameters": {"n_epochs": n_epochs},
        }
        if job_name:
            payload["suffix"] = job_name

        data = await asyncio.to_thread(
            self._transport,
            "POST",
            f"{self._api_base}/fine_tuning/jobs",
            self._headers,
            payload,
        )
        return FineTuneJob(
            id=str(data.get("id", "")),
            provider="openai",
            status=str(data.get("status", "created")),
            base_model=str(data.get("model", base_model)),
            model_id=_str_or_none(data.get("fine_tuned_model")),
            error=_extract_error(data),
        )

    async def status(self, job_id: str) -> FineTuneJob:
        data = await asyncio.to_thread(
            self._transport,
            "GET",
            f"{self._api_base}/fine_tuning/jobs/{job_id}",
            self._headers,
            None,
        )
        return FineTuneJob(
            id=str(data.get("id", job_id)),
            provider="openai",
            status=str(data.get("status", "unknown")),
            base_model=_str_or_none(data.get("model")),
            model_id=_str_or_none(data.get("fine_tuned_model")),
            error=_extract_error(data),
        )


class TogetherFineTuneAdapter(FineTuneAdapter):
    """Together AI fine-tuning adapter."""

    def __init__(
        self,
        api_key: str,
        *,
        api_base: str = "https://api.together.xyz/v1",
        transport: Transport | None = None,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._transport = transport or _http_json_transport
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def submit(
        self,
        *,
        dataset: str,
        base_model: str,
        job_name: str | None,
        n_epochs: int,
    ) -> FineTuneJob:
        payload: dict[str, Any] = {
            "training_file": dataset,
            "model": base_model,
            "n_epochs": n_epochs,
        }
        if job_name:
            payload["suffix"] = job_name

        data = await asyncio.to_thread(
            self._transport,
            "POST",
            f"{self._api_base}/fine-tuning/jobs",
            self._headers,
            payload,
        )
        return FineTuneJob(
            id=str(data.get("id", "")),
            provider="together",
            status=str(data.get("status", "created")),
            base_model=_str_or_none(data.get("model")) or base_model,
            model_id=_str_or_none(data.get("output_name"))
            or _str_or_none(data.get("model_output_name")),
            error=_extract_error(data),
        )

    async def status(self, job_id: str) -> FineTuneJob:
        data = await asyncio.to_thread(
            self._transport,
            "GET",
            f"{self._api_base}/fine-tuning/jobs/{job_id}",
            self._headers,
            None,
        )
        return FineTuneJob(
            id=str(data.get("id", job_id)),
            provider="together",
            status=str(data.get("status", "unknown")),
            base_model=_str_or_none(data.get("model")),
            model_id=_str_or_none(data.get("output_name"))
            or _str_or_none(data.get("model_output_name")),
            error=_extract_error(data),
        )


class FineTuner:
    """High-level fine-tuning orchestration."""

    TERMINAL_SUCCESS = {"succeeded", "completed", "success"}
    TERMINAL_FAILURE = {"failed", "cancelled", "canceled", "error"}

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        adapter: FineTuneAdapter | None = None,
    ) -> None:
        self.provider = provider.lower().strip()
        self._adapter = adapter or _build_default_adapter(self.provider, api_key)

    async def submit(
        self,
        *,
        dataset: str,
        base_model: str,
        job_name: str | None = None,
        n_epochs: int = 3,
    ) -> FineTuneJob:
        if n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")
        return await self._adapter.submit(
            dataset=dataset,
            base_model=base_model,
            job_name=job_name,
            n_epochs=n_epochs,
        )

    async def status(self, job_id: str) -> FineTuneJob:
        return await self._adapter.status(job_id)

    async def wait(
        self,
        job_id: str,
        *,
        timeout_s: float = 3600,
        poll_interval_s: float = 10,
    ) -> FineTuneJob:
        start = time.monotonic()

        while True:
            job = await self.status(job_id)
            status = job.status.lower()
            if status in self.TERMINAL_SUCCESS:
                return job
            if status in self.TERMINAL_FAILURE:
                message = job.error or f"Fine-tune job failed with status '{job.status}'"
                raise RuntimeError(message)

            if (time.monotonic() - start) >= timeout_s:
                raise TimeoutError(f"Timed out waiting for fine-tune job '{job_id}'")

            await asyncio.sleep(poll_interval_s)


def _build_default_adapter(provider: str, api_key: str) -> FineTuneAdapter:
    if provider == "openai":
        return OpenAIFineTuneAdapter(api_key)
    if provider == "together":
        return TogetherFineTuneAdapter(api_key)
    raise ValueError(f"Unsupported fine-tuning provider: {provider}")


def _extract_error(data: dict[str, Any]) -> str | None:
    err = data.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
        return str(msg) if msg is not None else json.dumps(err)
    if isinstance(err, str):
        return err
    return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
