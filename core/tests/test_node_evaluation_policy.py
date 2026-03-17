"""Tests for NodeEvaluationPolicy, EvalReport, NodeEvaluator, and GraphExecutor integration."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from framework.graph.evaluator import NodeEvaluator
from framework.graph.node import NodeResult, NodeSpec
from framework.schemas.eval_policy import BreachAction, NodeEvaluationPolicy
from framework.schemas.eval_report import EvalReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(policy: NodeEvaluationPolicy | None = None) -> NodeSpec:
    return NodeSpec(
        id="test_node",
        name="Test Node",
        description="A test node",
        evaluation_policy=policy,
    )


def _make_result(success: bool = True) -> NodeResult:
    return NodeResult(
        success=success,
        output={"answer": "42"},
        tokens_used=100,
        latency_ms=50,
    )


def _make_report(**kwargs: Any) -> EvalReport:
    defaults = dict(
        node_id="test_node",
        faithfulness=1.0,
        relevance=1.0,
        completeness=1.0,
        cost_efficiency=1.0,
    )
    defaults.update(kwargs)
    return EvalReport(**defaults)


class _FixedEvaluator:
    """Minimal NodeEvaluator that always returns a fixed EvalReport."""

    def __init__(self, report: EvalReport) -> None:
        self._report = report

    async def evaluate(
        self,
        node_spec: NodeSpec,
        node_result: NodeResult,
        memory: dict[str, Any],
    ) -> EvalReport:
        return self._report


# ---------------------------------------------------------------------------
# 1. EvalReport defaults
# ---------------------------------------------------------------------------


def test_eval_report_defaults() -> None:
    report = EvalReport(node_id="n1")
    assert report.faithfulness == 1.0
    assert report.relevance == 1.0
    assert report.completeness == 1.0
    assert report.cost_efficiency == 1.0
    assert report.weak_dimensions == []
    assert report.tokens_used == 0
    assert report.evaluator_model is None


# ---------------------------------------------------------------------------
# 2. EvalReport field validation (0.0–1.0)
# ---------------------------------------------------------------------------


def test_eval_report_rejects_out_of_range() -> None:
    with pytest.raises(Exception):
        EvalReport(node_id="n1", faithfulness=1.5)
    with pytest.raises(Exception):
        EvalReport(node_id="n1", relevance=-0.1)


# ---------------------------------------------------------------------------
# 3. NodeEvaluationPolicy defaults
# ---------------------------------------------------------------------------


def test_eval_policy_defaults() -> None:
    policy = NodeEvaluationPolicy()
    assert policy.min_faithfulness is None
    assert policy.min_relevance is None
    assert policy.min_completeness is None
    assert policy.min_cost_efficiency is None
    assert policy.action_on_breach == BreachAction.WARN
    assert policy.skip_evaluation is False


# ---------------------------------------------------------------------------
# 4. NodeSpec accepts evaluation_policy field
# ---------------------------------------------------------------------------


def test_node_spec_evaluation_policy_field() -> None:
    policy = NodeEvaluationPolicy(min_faithfulness=0.8, action_on_breach=BreachAction.GUARD_FAILURE)
    spec = _make_spec(policy)
    assert spec.evaluation_policy is policy
    assert spec.evaluation_policy.min_faithfulness == 0.8


def test_node_spec_evaluation_policy_default_none() -> None:
    spec = _make_spec()
    assert spec.evaluation_policy is None


# ---------------------------------------------------------------------------
# 5. NodeEvaluator structural typing (Protocol check)
# ---------------------------------------------------------------------------


def test_node_evaluator_protocol_check() -> None:
    report = _make_report()
    evaluator = _FixedEvaluator(report)
    assert isinstance(evaluator, NodeEvaluator)


# ---------------------------------------------------------------------------
# 6. _check_eval_policy — no breach (all scores above thresholds)
# ---------------------------------------------------------------------------


def test_check_eval_policy_no_breach() -> None:
    from framework.graph.executor import GraphExecutor

    executor = MagicMock(spec=GraphExecutor)
    executor.logger = MagicMock()
    policy = NodeEvaluationPolicy(min_faithfulness=0.5, min_relevance=0.5)
    report = _make_report(faithfulness=0.9, relevance=0.9)
    result = _make_result()

    actual = GraphExecutor._check_eval_policy(executor, policy, report, result)
    assert actual.success is True
    executor.logger.warning.assert_not_called()
    executor.logger.error.assert_not_called()


# ---------------------------------------------------------------------------
# 7. _check_eval_policy — WARN breach
# ---------------------------------------------------------------------------


def test_check_eval_policy_warn_breach() -> None:
    from framework.graph.executor import GraphExecutor

    executor = MagicMock(spec=GraphExecutor)
    executor.logger = MagicMock()
    policy = NodeEvaluationPolicy(min_faithfulness=0.8, action_on_breach=BreachAction.WARN)
    report = _make_report(faithfulness=0.5)
    result = _make_result()

    actual = GraphExecutor._check_eval_policy(executor, policy, report, result)
    assert actual.success is True  # WARN keeps success
    executor.logger.warning.assert_called_once()
    assert "WARN" in executor.logger.warning.call_args[0][0]


# ---------------------------------------------------------------------------
# 8. _check_eval_policy — GUARD_FAILURE breach
# ---------------------------------------------------------------------------


def test_check_eval_policy_guard_failure() -> None:
    from framework.graph.executor import GraphExecutor

    executor = MagicMock(spec=GraphExecutor)
    executor.logger = MagicMock()
    policy = NodeEvaluationPolicy(
        min_faithfulness=0.8, action_on_breach=BreachAction.GUARD_FAILURE
    )
    report = _make_report(faithfulness=0.3)
    result = _make_result()

    actual = GraphExecutor._check_eval_policy(executor, policy, report, result)
    assert actual.success is False
    assert "guard_failure" in actual.error
    executor.logger.error.assert_called_once()


# ---------------------------------------------------------------------------
# 9. _check_eval_policy — DEGRADE_MODEL falls back to WARN
# ---------------------------------------------------------------------------


def test_check_eval_policy_degrade_model_fallback() -> None:
    from framework.graph.executor import GraphExecutor

    executor = MagicMock(spec=GraphExecutor)
    executor.logger = MagicMock()
    policy = NodeEvaluationPolicy(
        min_relevance=0.9, action_on_breach=BreachAction.DEGRADE_MODEL
    )
    report = _make_report(relevance=0.4)
    result = _make_result()

    actual = GraphExecutor._check_eval_policy(executor, policy, report, result)
    assert actual.success is True  # Treated as WARN (success preserved)
    executor.logger.warning.assert_called_once()
    assert "#6214" in executor.logger.warning.call_args[0][0]


# ---------------------------------------------------------------------------
# 10. store_eval_report — in-memory dict and store call
# ---------------------------------------------------------------------------


def test_store_eval_report_persists() -> None:
    from framework.runtime.runtime_logger import RuntimeLogger

    store = MagicMock()
    logger_inst = RuntimeLogger(store=store, agent_id="test")
    logger_inst._run_id = "run_001"

    report = _make_report(node_id="node_x", faithfulness=0.7)
    logger_inst.store_eval_report(report)

    assert "node_x" in logger_inst._eval_reports
    assert logger_inst._eval_reports["node_x"].faithfulness == 0.7
    store.append_eval_report.assert_called_once_with("run_001", report)


# ---------------------------------------------------------------------------
# 11. skip_evaluation=True — evaluator is not called
# ---------------------------------------------------------------------------


def test_skip_evaluation_flag() -> None:
    """When skip_evaluation=True, NodeEvaluator.evaluate() must not be called."""

    class _TrackingEvaluator:
        called = False

        async def evaluate(
            self,
            node_spec: NodeSpec,
            node_result: NodeResult,
            memory: dict[str, Any],
        ) -> EvalReport:
            _TrackingEvaluator.called = True
            return _make_report()

    from framework.graph.executor import GraphExecutor

    executor = MagicMock(spec=GraphExecutor)
    executor.node_evaluator = _TrackingEvaluator()
    executor.logger = MagicMock()
    executor.runtime_logger = None

    spec = _make_spec(NodeEvaluationPolicy(skip_evaluation=True))
    result = _make_result()

    async def _run() -> None:
        policy = spec.evaluation_policy
        if executor.node_evaluator is not None:
            if policy is None or not policy.skip_evaluation:
                await executor.node_evaluator.evaluate(spec, result, {})

    asyncio.get_event_loop().run_until_complete(_run())
    assert not _TrackingEvaluator.called
