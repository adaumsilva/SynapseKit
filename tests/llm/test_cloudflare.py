from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.cloudflare import CloudflareLLM


def make_config() -> LLMConfig:
    return LLMConfig(
        model="@cf/meta/llama-3.1-8b-instruct",
        api_key="test-token",
        provider="cloudflare",
    )


async def _line_iter(*lines: str):
    for line in lines:
        yield line


def _mock_stream(lines: list[str]) -> MagicMock:
    """Return a mocked httpx client whose .stream() is a usable async CM."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = lambda: _line_iter(*lines)

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_cm
    return mock_client


class TestCloudflareLLM:
    @pytest.fixture
    def llm(self) -> CloudflareLLM:
        return CloudflareLLM(make_config(), account_id="test-account")

    # ------------------------------------------------------------------
    # Construction & helpers
    # ------------------------------------------------------------------

    def test_endpoint_format(self, llm: CloudflareLLM) -> None:
        assert llm._endpoint() == (
            "https://api.cloudflare.com/client/v4/accounts"
            "/test-account/ai/run/@cf/meta/llama-3.1-8b-instruct"
        )

    def test_custom_base_url(self) -> None:
        llm = CloudflareLLM(
            make_config(),
            account_id="acct",
            base_url="https://custom.example.com",
        )
        assert llm._endpoint().startswith("https://custom.example.com")

    def test_import_error_without_httpx(self) -> None:
        llm = CloudflareLLM(make_config(), account_id="test-account")
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ImportError, match="httpx"):
                llm._get_client()

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, llm: CloudflareLLM) -> None:
        llm._client = _mock_stream([
            'data: {"response": "Hello"}',
            'data: {"response": " world"}',
            "data: [DONE]",
        ])

        tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_skips_non_data_lines(self, llm: CloudflareLLM) -> None:
        llm._client = _mock_stream([
            ": ping",
            "",
            'data: {"response": "Token"}',
            "data: [DONE]",
        ])

        tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Token"]

    @pytest.mark.asyncio
    async def test_stream_skips_invalid_json(self, llm: CloudflareLLM) -> None:
        llm._client = _mock_stream([
            "data: not-json",
            'data: {"response": "Good"}',
            "data: [DONE]",
        ])

        tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Good"]

    @pytest.mark.asyncio
    async def test_stream_stops_at_done(self, llm: CloudflareLLM) -> None:
        llm._client = _mock_stream([
            'data: {"response": "A"}',
            "data: [DONE]",
            'data: {"response": "B"}',  # should never be reached
        ])

        tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["A"]

    # ------------------------------------------------------------------
    # stream_with_messages
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_with_messages(self, llm: CloudflareLLM) -> None:
        llm._client = _mock_stream([
            'data: {"response": "Hi"}',
            "data: [DONE]",
        ])

        messages = [{"role": "user", "content": "hello"}]
        tokens = [t async for t in llm.stream_with_messages(messages)]
        assert tokens == ["Hi"]

    @pytest.mark.asyncio
    async def test_stream_with_messages_passes_kwargs(self, llm: CloudflareLLM) -> None:
        llm._client = _mock_stream(['data: {"response": "x"}', "data: [DONE]"])

        messages = [{"role": "user", "content": "q"}]
        _ = [t async for t in llm.stream_with_messages(messages, temperature=0.9, max_tokens=512)]

        call_kwargs = llm._client.stream.call_args
        payload = call_kwargs[1]["json"]
        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 512

    # ------------------------------------------------------------------
    # generate (integration of stream)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_generate_collects_tokens(self, llm: CloudflareLLM) -> None:
        llm._client = _mock_stream([
            'data: {"response": "Hello"}',
            'data: {"response": " world"}',
            "data: [DONE]",
        ])

        result = await llm.generate("hi")
        assert result == "Hello world"

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_output_tokens_tracked(self, llm: CloudflareLLM) -> None:
        llm._client = _mock_stream([
            'data: {"response": "A"}',
            'data: {"response": "B"}',
            'data: {"response": "C"}',
            "data: [DONE]",
        ])

        _ = [t async for t in llm.stream("hi")]
        assert llm._output_tokens == 3
