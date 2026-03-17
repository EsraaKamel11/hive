"""Tests for the NodeEvaluator hook in GraphExecutor.

Covers:
- Evaluator is called after a successful node execution
- EvalReport is stored in RuntimeLogger
- Evaluator is NOT called when node fails
- Evaluator exception is non-blocking (run continues normally)
- No evaluator → execution path is unchanged
"""

from __future__ import annotations

from typing import Any

import pytest

from framework.graph.edge import GraphSpec
from framework.graph.evaluator import NodeEvaluator
from framework.graph.executor import GraphExecutor
from framework.graph.goal import Goal
from framework.graph.node import NodeResult, NodeSpec
from framework.schemas.eval_report import EvalReport


# ---------------------------------------------------------------------------
# Shared test doubles
# ---------------------------------------------------------------------------


class DummyRuntime:
    execution_id = ""

    def start_run(self, **kwargs):
        return "run-1"

    def end_run(self, **kwargs):
        pass

    def report_problem(self, **kwargs):
        pass


def _make_graph(node_id: str = "n1", output_keys: list[str] | None = None) -> GraphSpec:
    return GraphSpec(
        id="graph-1",
        goal_id="g1",
        nodes=[
            NodeSpec(
                id=node_id,
                name="test-node",
                description="test",
                node_type="event_loop",
                input_keys=[],
                output_keys=output_keys or [],
                max_retries=0,
            )
        ],
        edges=[],
        entry_node=node_id,
        terminal_nodes=[node_id],
    )


class SuccessNode:
    """Returns success with one output key."""

    def validate_input(self, ctx):
        return []

    async def execute(self, ctx):
        return NodeResult(success=True, output={"result": "ok"}, tokens_used=10, latency_ms=5)


class FailureNode:
    """Always returns failure."""

    def validate_input(self, ctx):
        return []

    async def execute(self, ctx):
        return NodeResult(success=False, error="deliberate failure")


# ---------------------------------------------------------------------------
# NodeEvaluator implementations for testing
# ---------------------------------------------------------------------------


class RecordingEvaluator:
    """Records every evaluate() call and returns a fixed EvalReport."""

    def __init__(self, report: EvalReport | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._report = report

    async def evaluate(self, node_spec: NodeSpec, node_result: NodeResult, memory: dict) -> EvalReport:
        self.calls.append({"node_spec": node_spec, "node_result": node_result, "memory": memory})
        if self._report is not None:
            return self._report
        return EvalReport(
            node_id=node_spec.id,
            faithfulness=0.9,
            relevance=0.8,
            completeness=1.0,
            cost_efficiency=0.7,
            tokens_used=node_result.tokens_used,
        )


class RaisingEvaluator:
    """Always raises an exception from evaluate()."""

    async def evaluate(self, node_spec: NodeSpec, node_result: NodeResult, memory: dict) -> EvalReport:
        raise RuntimeError("evaluator exploded")


class MinimalRuntimeLogger:
    """Minimal RuntimeLogger stand-in that records stored eval reports."""

    def __init__(self) -> None:
        self._eval_reports: dict[str, EvalReport] = {}

    def start_run(self, **kwargs) -> str:
        return "run-1"

    async def end_run(self, **kwargs) -> None:
        pass

    def ensure_node_logged(self, **kwargs) -> None:
        pass

    def store_eval_report(self, report: EvalReport) -> None:
        self._eval_reports[report.node_id] = report

    def get_eval_reports(self) -> dict[str, EvalReport]:
        return dict(self._eval_reports)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluator_called_on_success():
    """NodeEvaluator.evaluate() is called exactly once for a successful node."""
    evaluator = RecordingEvaluator()
    executor = GraphExecutor(
        runtime=DummyRuntime(),
        node_registry={"n1": SuccessNode()},
        node_evaluator=evaluator,
    )
    result = await executor.execute(_make_graph(), goal=Goal(id="g1", name="test", description="test"))

    assert result.success
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0]["node_spec"].id == "n1"
    assert evaluator.calls[0]["node_result"].success is True


@pytest.mark.asyncio
async def test_evaluator_report_stored_in_runtime_logger():
    """EvalReport is forwarded to RuntimeLogger.store_eval_report()."""
    evaluator = RecordingEvaluator()
    runtime_logger = MinimalRuntimeLogger()

    executor = GraphExecutor(
        runtime=DummyRuntime(),
        node_registry={"n1": SuccessNode()},
        node_evaluator=evaluator,
        runtime_logger=runtime_logger,
    )
    result = await executor.execute(_make_graph(), goal=Goal(id="g1", name="test", description="test"))

    assert result.success
    reports = runtime_logger.get_eval_reports()
    assert "n1" in reports
    assert reports["n1"].faithfulness == pytest.approx(0.9)
    assert reports["n1"].tokens_used == 10


@pytest.mark.asyncio
async def test_evaluator_not_called_on_failure():
    """NodeEvaluator is NOT invoked when the node returns success=False."""
    evaluator = RecordingEvaluator()
    executor = GraphExecutor(
        runtime=DummyRuntime(),
        node_registry={"n1": FailureNode()},
        node_evaluator=evaluator,
    )
    result = await executor.execute(_make_graph(), goal=Goal(id="g1", name="test", description="test"))

    assert not result.success
    assert len(evaluator.calls) == 0


@pytest.mark.asyncio
async def test_evaluator_exception_is_non_blocking():
    """A raise from NodeEvaluator.evaluate() does not abort the run."""
    executor = GraphExecutor(
        runtime=DummyRuntime(),
        node_registry={"n1": SuccessNode()},
        node_evaluator=RaisingEvaluator(),
    )
    result = await executor.execute(_make_graph(), goal=Goal(id="g1", name="test", description="test"))

    # Run still succeeds — evaluator failure is a warning, not an error
    assert result.success


@pytest.mark.asyncio
async def test_no_evaluator_execution_unchanged():
    """Without a node_evaluator the execution path is identical to before."""
    executor = GraphExecutor(
        runtime=DummyRuntime(),
        node_registry={"n1": SuccessNode()},
        # node_evaluator intentionally omitted
    )
    result = await executor.execute(_make_graph(), goal=Goal(id="g1", name="test", description="test"))

    assert result.success
    assert result.output.get("result") == "ok"


@pytest.mark.asyncio
async def test_evaluator_receives_memory_snapshot():
    """memory dict passed to evaluate() reflects state before memory writes."""
    captured: dict = {}

    class MemoryCapturingEvaluator:
        async def evaluate(self, node_spec, node_result, memory):
            captured["memory"] = dict(memory)
            return EvalReport(node_id=node_spec.id)

    executor = GraphExecutor(
        runtime=DummyRuntime(),
        node_registry={"n1": SuccessNode()},
        node_evaluator=MemoryCapturingEvaluator(),
    )
    await executor.execute(_make_graph(), goal=Goal(id="g1", name="test", description="test"))

    # The evaluator saw memory BEFORE "result" was written
    assert "result" not in captured["memory"]


@pytest.mark.asyncio
async def test_runtime_checkable_protocol():
    """NodeEvaluator is runtime_checkable — isinstance() works."""
    evaluator = RecordingEvaluator()
    assert isinstance(evaluator, NodeEvaluator)

    # A plain object without evaluate() is not a NodeEvaluator
    assert not isinstance(object(), NodeEvaluator)


def test_eval_report_field_constraints():
    """EvalReport enforces 0.0–1.0 range on float dimensions."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        EvalReport(node_id="x", faithfulness=1.5)

    with pytest.raises(pydantic.ValidationError):
        EvalReport(node_id="x", relevance=-0.1)

    # Valid edge values are accepted
    report = EvalReport(node_id="x", faithfulness=0.0, cost_efficiency=1.0)
    assert report.faithfulness == 0.0
    assert report.cost_efficiency == 1.0
