"""NodeEvaluator: Protocol for execution-time node quality evaluation."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from framework.graph.node import NodeResult, NodeSpec
from framework.schemas.eval_report import EvalReport


@runtime_checkable
class NodeEvaluator(Protocol):
    """Evaluate a node's output immediately after successful execution.

    The hook in GraphExecutor fires:
    - After output validation passes (result.success is definitively True)
    - Before memory writes (no side effects have been taken yet)
    - For every node type that reaches the success path

    Implementations can be LLM-based, rule-based, or hybrid.
    Structural typing — no inheritance or registration required.

    Non-blocking contract: if evaluate() raises, GraphExecutor catches the
    exception, logs a warning, and continues normally. Evaluator failure
    never affects routing or memory writes.
    """

    async def evaluate(
        self,
        node_spec: NodeSpec,
        node_result: NodeResult,
        memory: dict[str, Any],
    ) -> EvalReport: ...
