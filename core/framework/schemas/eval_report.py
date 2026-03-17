"""EvalReport: multi-dimensional quality report produced by a NodeEvaluator."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalReport(BaseModel):
    """Quality report for a single node execution.

    Produced by a NodeEvaluator after the node succeeds and before
    memory writes. All float dimensions are 0.0–1.0.

    Relationship to DecisionEvaluation (schemas/decision.py):
    - DecisionEvaluation is post-hoc (computed after the run ends).
    - EvalReport is execution-time (computed immediately after each node).
    These are sibling schemas; neither replaces the other.
    """

    node_id: str

    # Output grounded in inputs/context — primary hallucination signal
    faithfulness: float = Field(default=1.0, ge=0.0, le=1.0)

    # Output addresses the node's declared goal
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)

    # All output_keys populated meaningfully (no placeholder / empty values)
    completeness: float = Field(default=1.0, ge=0.0, le=1.0)

    # Quality-per-token ratio — feeds DegradationPolicy threshold checks
    cost_efficiency: float = Field(default=1.0, ge=0.0, le=1.0)

    # Dimension names scoring below threshold (populated by the evaluator)
    weak_dimensions: list[str] = Field(default_factory=list)

    # Mirrors NodeResult.tokens_used for evaluator-side accounting
    tokens_used: int = 0

    # Model that produced this report (None = rule-based / no LLM involved)
    evaluator_model: str | None = None
