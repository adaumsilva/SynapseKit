"""Sentiment Analysis Tool: analyze sentiment using an LLM."""

from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolResult


class SentimentAnalysisTool(BaseTool):
    """Analyze the sentiment of text using an LLM.

    Usage::

        tool = SentimentAnalysisTool(llm=llm)
        result = await tool.run(text="I love this product!")
    """

    name = "sentiment_analysis"
    description = (
        "Analyze the sentiment of a piece of text. "
        "Input: the text to analyze. "
        "Returns: sentiment (positive/negative/neutral), confidence, and explanation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to analyze for sentiment",
            },
        },
        "required": ["text"],
    }

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def run(self, text: str = "", **kwargs: Any) -> ToolResult:
        input_text = text or kwargs.get("input", "")
        if not input_text:
            return ToolResult(output="", error="No text provided for sentiment analysis.")

        prompt = (
            "Analyze the sentiment of the following text. "
            "Respond with exactly three lines:\n"
            "Sentiment: positive/negative/neutral/mixed\n"
            "Confidence: high/medium/low\n"
            "Explanation: one sentence explaining why\n\n"
            f"Text:\n{input_text}"
        )

        try:
            result: str = await self._llm.generate(prompt)
            return ToolResult(output=result.strip())
        except Exception as e:
            return ToolResult(output="", error=f"Sentiment analysis failed: {e}")
