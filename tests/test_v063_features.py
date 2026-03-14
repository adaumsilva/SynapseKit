"""Tests for v0.6.3 features."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

# ── Typed State with Reducers (#253) ──


def test_state_field_creation():
    from synapsekit.graph.state import StateField

    sf = StateField(default=list, reducer=lambda cur, new: cur + new)
    assert callable(sf.default)
    assert sf.reducer is not None


def test_typed_state_initial_state():
    from synapsekit.graph.state import StateField, TypedState

    schema = TypedState(
        fields={
            "messages": StateField(default=list),
            "count": StateField(default=int),
            "name": StateField(default="default"),
        }
    )
    state = schema.initial_state()
    assert state == {"messages": [], "count": 0, "name": "default"}


def test_typed_state_merge_with_reducer():
    from synapsekit.graph.state import StateField, TypedState

    schema = TypedState(
        fields={
            "messages": StateField(default=list, reducer=lambda cur, new: cur + new),
            "count": StateField(default=int, reducer=lambda cur, new: cur + new),
            "result": StateField(default=str),
        }
    )
    state = schema.initial_state()
    schema.merge(state, {"messages": ["hello"], "count": 1, "result": "ok"})
    assert state == {"messages": ["hello"], "count": 1, "result": "ok"}

    schema.merge(state, {"messages": ["world"], "count": 2, "result": "done"})
    assert state["messages"] == ["hello", "world"]
    assert state["count"] == 3
    assert state["result"] == "done"  # last-write-wins


def test_typed_state_merge_unknown_keys():
    from synapsekit.graph.state import StateField, TypedState

    schema = TypedState(fields={"x": StateField(default=int)})
    state = schema.initial_state()
    schema.merge(state, {"y": "extra"})
    assert state["y"] == "extra"


@pytest.mark.asyncio
async def test_graph_with_typed_state():
    from synapsekit import StateGraph
    from synapsekit.graph.state import StateField, TypedState

    schema = TypedState(
        fields={
            "messages": StateField(default=list, reducer=lambda cur, new: cur + new),
            "count": StateField(default=int, reducer=lambda cur, new: cur + new),
        }
    )
    graph = StateGraph(state_schema=schema)

    def node_a(state):
        return {"messages": ["a"], "count": 1}

    def node_b(state):
        return {"messages": ["b"], "count": 1}

    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_edge("a", "b")
    graph.set_entry_point("a")
    graph.set_finish_point("b")

    compiled = graph.compile()
    result = await compiled.run({"messages": [], "count": 0})
    assert result["messages"] == ["a", "b"]
    assert result["count"] == 2


# ── Parallel Subgraph Execution / Fan-Out (#248) ──


@pytest.mark.asyncio
async def test_fan_out_basic():
    from synapsekit import StateGraph
    from synapsekit.graph.fan_out import fan_out_node

    def sub_fn(state):
        return {"output": state.get("input", "") + "_processed"}

    sub = StateGraph()
    sub.add_node("process", sub_fn)
    sub.set_entry_point("process")
    sub.set_finish_point("process")
    compiled_sub = sub.compile()

    fan = fan_out_node(
        subgraphs=[compiled_sub, compiled_sub],
        input_mappings=[{"query": "input"}, {"query": "input"}],
        output_key="results",
    )

    result = await fan({"query": "test"})
    assert len(result["results"]) == 2
    assert result["results"][0]["output"] == "test_processed"


@pytest.mark.asyncio
async def test_fan_out_with_merge():
    from synapsekit import StateGraph
    from synapsekit.graph.fan_out import fan_out_node

    def sub_fn(state):
        return {"output": state.get("input", "x")}

    sub = StateGraph()
    sub.add_node("p", sub_fn)
    sub.set_entry_point("p")
    sub.set_finish_point("p")
    compiled_sub = sub.compile()

    def merge(results):
        combined = " + ".join(r.get("output", "") for r in results)
        return {"combined": combined}

    fan = fan_out_node(subgraphs=[compiled_sub, compiled_sub], merge_fn=merge)
    result = await fan({"input": "hi"})
    assert "combined" in result
    assert "hi" in result["combined"]


def test_fan_out_mismatched_mappings():
    from synapsekit.graph.fan_out import fan_out_node

    with pytest.raises(ValueError, match="same length"):
        fan_out_node(subgraphs=["a", "b"], input_mappings=[{}])


# ── SSE Streaming (#238) ──


@pytest.mark.asyncio
async def test_sse_stream():
    from synapsekit import StateGraph
    from synapsekit.graph.streaming import sse_stream

    def node_a(state):
        return {"output": "hello"}

    graph = StateGraph()
    graph.add_node("a", node_a)
    graph.set_entry_point("a")
    graph.set_finish_point("a")
    compiled = graph.compile()

    events = []
    async for sse in sse_stream(compiled, {"input": "x"}):
        events.append(sse)

    assert len(events) == 2  # node_complete + done
    assert "event: node_complete" in events[0]
    assert "event: done" in events[1]


def test_graph_event_to_sse():
    from synapsekit.graph.streaming import GraphEvent

    event = GraphEvent(event_type="node_complete", node="a", state={"x": 1})
    sse = event.to_sse()
    assert "event: node_complete" in sse
    assert '"node": "a"' in sse


def test_graph_event_to_dict():
    from synapsekit.graph.streaming import GraphEvent

    event = GraphEvent(event_type="node_start", node="b")
    d = event.to_dict()
    assert d["event"] == "node_start"
    assert d["node"] == "b"


# ── Event Callbacks / Hooks (#240) ──


@pytest.mark.asyncio
async def test_event_hooks():
    from synapsekit import StateGraph
    from synapsekit.graph.streaming import EventHooks

    events_log = []

    hooks = EventHooks()
    hooks.on_node_start(lambda e: events_log.append(("start", e.node)))
    hooks.on_node_complete(lambda e: events_log.append(("complete", e.node)))
    hooks.on_wave_start(lambda e: events_log.append(("wave_start",)))
    hooks.on_wave_complete(lambda e: events_log.append(("wave_complete",)))

    def node_a(state):
        return {"output": "done"}

    graph = StateGraph()
    graph.add_node("a", node_a)
    graph.set_entry_point("a")
    graph.set_finish_point("a")
    compiled = graph.compile()

    await compiled.run({"input": "x"}, hooks=hooks)

    assert ("start", "a") in events_log
    assert ("complete", "a") in events_log
    assert ("wave_start",) in events_log
    assert ("wave_complete",) in events_log


@pytest.mark.asyncio
async def test_event_hooks_async_callback():
    from synapsekit.graph.streaming import EventHooks, GraphEvent

    results = []

    async def async_cb(event):
        results.append(event.event_type)

    hooks = EventHooks()
    hooks.on("test_event", async_cb)
    await hooks.emit(GraphEvent(event_type="test_event"))
    assert results == ["test_event"]


# ── Semantic LLM Cache (#196) ──


@pytest.mark.asyncio
async def test_semantic_cache_hit():
    from synapsekit.llm._semantic_cache import SemanticCache

    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(
        side_effect=[
            [1.0, 0.0, 0.0],  # put: "What is Python?"
            [0.99, 0.1, 0.0],  # get: "Tell me about Python" (similar)
        ]
    )

    cache = SemanticCache(embeddings=embeddings, threshold=0.9)
    await cache.put("What is Python?", "Python is a programming language.")
    result = await cache.get("Tell me about Python")

    assert result == "Python is a programming language."
    assert cache.hits == 1


@pytest.mark.asyncio
async def test_semantic_cache_miss():
    from synapsekit.llm._semantic_cache import SemanticCache

    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(
        side_effect=[
            [1.0, 0.0, 0.0],  # put
            [0.0, 1.0, 0.0],  # get (very different)
        ]
    )

    cache = SemanticCache(embeddings=embeddings, threshold=0.9)
    await cache.put("What is Python?", "Python is a programming language.")
    result = await cache.get("What is the weather?")

    assert result is None
    assert cache.misses == 1


@pytest.mark.asyncio
async def test_semantic_cache_empty():
    from synapsekit.llm._semantic_cache import SemanticCache

    embeddings = AsyncMock()
    cache = SemanticCache(embeddings=embeddings)
    result = await cache.get("anything")
    assert result is None
    assert cache.misses == 1


def test_semantic_cache_validation():
    from synapsekit.llm._semantic_cache import SemanticCache

    with pytest.raises(ValueError, match="threshold"):
        SemanticCache(embeddings=None, threshold=1.5)
    with pytest.raises(ValueError, match="maxsize"):
        SemanticCache(embeddings=None, maxsize=0)


@pytest.mark.asyncio
async def test_semantic_cache_eviction():
    from synapsekit.llm._semantic_cache import SemanticCache

    embeddings = AsyncMock()
    call_count = 0

    async def mock_embed(text):
        nonlocal call_count
        call_count += 1
        return [float(call_count), 0.0, 0.0]

    embeddings.embed = mock_embed

    cache = SemanticCache(embeddings=embeddings, maxsize=2)
    await cache.put("a", "response_a")
    await cache.put("b", "response_b")
    await cache.put("c", "response_c")  # evicts "a"

    assert len(cache) == 2


# ── Summarization Tool (#223) ──


@pytest.mark.asyncio
async def test_summarization_tool():
    from synapsekit.agents.tools.summarization import SummarizationTool

    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="This is a summary.")

    tool = SummarizationTool(llm=llm)
    assert tool.name == "summarize"

    result = await tool.run(text="Long text here...")
    assert result.output == "This is a summary."
    assert not result.is_error


@pytest.mark.asyncio
async def test_summarization_tool_bullet_points():
    from synapsekit.agents.tools.summarization import SummarizationTool

    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="- Point 1\n- Point 2")

    tool = SummarizationTool(llm=llm)
    result = await tool.run(text="Text", style="bullet_points")
    assert "Point 1" in result.output


@pytest.mark.asyncio
async def test_summarization_tool_no_text():
    from synapsekit.agents.tools.summarization import SummarizationTool

    tool = SummarizationTool(llm=AsyncMock())
    result = await tool.run()
    assert result.is_error


# ── Sentiment Analysis Tool (#225) ──


@pytest.mark.asyncio
async def test_sentiment_tool():
    from synapsekit.agents.tools.sentiment import SentimentAnalysisTool

    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value="Sentiment: positive\nConfidence: high\nExplanation: The text expresses enthusiasm."
    )

    tool = SentimentAnalysisTool(llm=llm)
    assert tool.name == "sentiment_analysis"

    result = await tool.run(text="I love this product!")
    assert "positive" in result.output
    assert not result.is_error


@pytest.mark.asyncio
async def test_sentiment_tool_no_text():
    from synapsekit.agents.tools.sentiment import SentimentAnalysisTool

    tool = SentimentAnalysisTool(llm=AsyncMock())
    result = await tool.run()
    assert result.is_error


# ── Translation Tool (#224) ──


@pytest.mark.asyncio
async def test_translation_tool():
    from synapsekit.agents.tools.translation import TranslationTool

    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="Hola mundo!")

    tool = TranslationTool(llm=llm)
    assert tool.name == "translate"

    result = await tool.run(text="Hello world!", target_language="Spanish")
    assert result.output == "Hola mundo!"
    assert not result.is_error


@pytest.mark.asyncio
async def test_translation_tool_with_source():
    from synapsekit.agents.tools.translation import TranslationTool

    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="Bonjour")

    tool = TranslationTool(llm=llm)
    result = await tool.run(text="Hello", target_language="French", source_language="English")
    assert result.output == "Bonjour"


@pytest.mark.asyncio
async def test_translation_tool_no_text():
    from synapsekit.agents.tools.translation import TranslationTool

    tool = TranslationTool(llm=AsyncMock())
    result = await tool.run(target_language="Spanish")
    assert result.is_error


@pytest.mark.asyncio
async def test_translation_tool_no_target():
    from synapsekit.agents.tools.translation import TranslationTool

    tool = TranslationTool(llm=AsyncMock())
    result = await tool.run(text="Hello")
    assert result.is_error


# ── Import Tests ──


def test_imports():
    from synapsekit import (
        EventHooks,
        GraphEvent,
        SentimentAnalysisTool,
        StateField,
        SummarizationTool,
        TranslationTool,
        TypedState,
        fan_out_node,
        sse_stream,
    )
    from synapsekit.llm._semantic_cache import SemanticCache

    assert StateField is not None
    assert TypedState is not None
    assert fan_out_node is not None
    assert sse_stream is not None
    assert EventHooks is not None
    assert GraphEvent is not None
    assert SemanticCache is not None
    assert SummarizationTool is not None
    assert SentimentAnalysisTool is not None
    assert TranslationTool is not None
