"""NodeEvaluationPolicy: per-node declarative evaluation governance."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BreachAction(str, Enum):
    """Action taken when a NodeEvaluationPolicy threshold is breached."""

    WARN = "warn"
    """Log a warning and continue. Default — zero disruption to execution."""

    GUARD_FAILURE = "guard_failure"
    """Fail the node: NodeResult.success=False, exit_status='guard_failure'.
    Memory writes are blocked. Triggers executor retry/failure path."""

    DEGRADE_MODEL = "degrade_model"
    """Trigger model degradation via DegradationPolicy.fallback_models (#6214).
    Falls back to WARN with an explanatory log until #6214 is merged."""


class NodeEvaluationPolicy(BaseModel):
    """Declarative thresholds and actions for a node's EvalReport scores.

    Set on NodeSpec.evaluation_policy. GraphExecutor reads this after
    store_eval_report() and dispatches action_on_breach if any enabled
    dimension falls below its minimum.

    Relationship to DegradationPolicy (#6214):
    - DegradationPolicy triggers on token_budget (quantitative resource limit).
    - NodeEvaluationPolicy triggers on EvalReport scores (qualitative signal).
    - action_on_breach=DEGRADE_MODEL bridges the two once #6214 is merged.

    Relationship to NodeEvaluator (#6542):
    - NodeEvaluator.evaluate() produces EvalReport (observation).
    - NodeEvaluationPolicy declares what to do about it (governance).
    """

    # Per-dimension minimums — None means the dimension is not checked
    min_faithfulness: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum faithfulness score. None = not checked.",
    )
    min_relevance: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score. None = not checked.",
    )
    min_completeness: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum completeness score. None = not checked.",
    )
    min_cost_efficiency: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cost_efficiency score. None = not checked. "
            "Feeds DegradationPolicy when action_on_breach=DEGRADE_MODEL (#6214)."
        ),
    )

    # Action taken when any enabled threshold is breached
    action_on_breach: BreachAction = Field(
        default=BreachAction.WARN,
        description="Action to take when a threshold is breached.",
    )

    # Opt-out: skip evaluation entirely for this node.
    # Use for routing nodes, trivial pass-throughs, or fixed-output nodes
    # where running an evaluator adds cost and noise without signal.
    skip_evaluation: bool = Field(
        default=False,
        description=(
            "When True, the NodeEvaluator is not called for this node. "
            "Use for routing or trivial nodes where evaluation adds no signal."
        ),
    )
