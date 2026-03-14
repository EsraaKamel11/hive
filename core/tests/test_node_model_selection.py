"""Tests for per-node model selection and cost governance (#6214)."""

import pytest

from framework.graph.edge import EdgeCondition, EdgeSpec, GraphSpec
from framework.graph.executor import GraphExecutor, NodeCostState
from framework.graph.goal import Goal
from framework.graph.node import DegradationPolicy, NodeContext, NodeResult, NodeSpec
from framework.llm.mock import MockLLMProvider
from framework.llm.provider import LLMProvider, LLMResponse, Tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyRuntime:
    execution_id = ""

    def start_run(self, **kwargs):
        return "run-1"

    def end_run(self, **kwargs):
        pass

    def report_problem(self, **kwargs):
        pass


class ModelCapturingNode:
    """Fake node that records which LLM model it received via ctx.llm."""

    def __init__(self):
        self.captured_model: str | None = None

    def validate_input(self, ctx):
        return []

    async def execute(self, ctx: NodeContext) -> NodeResult:
        self.captured_model = getattr(ctx.llm, "model", None)
        return NodeResult(
            success=True,
            output={"result": "ok"},
            tokens_used=10,
            latency_ms=1,
            model_used=self.captured_model or "",
        )


class TokenBurningNode:
    """Fake node that returns a configurable tokens_used and captures the LLM model."""

    def __init__(self, tokens: int):
        self.tokens = tokens
        self.captured_models: list[str] = []

    def validate_input(self, ctx):
        return []

    async def execute(self, ctx: NodeContext) -> NodeResult:
        model = getattr(ctx.llm, "model", "unknown")
        self.captured_models.append(model)
        return NodeResult(
            success=True,
            output={"result": "ok"},
            tokens_used=self.tokens,
            latency_ms=1,
            model_used=model,
        )


class BounceNode:
    """Simple passthrough node for A→B→A cycle graphs."""

    def validate_input(self, ctx):
        return []

    async def execute(self, ctx: NodeContext) -> NodeResult:
        return NodeResult(success=True, output={"result": "ok"}, tokens_used=0, latency_ms=1)


def _simple_graph(node_spec: NodeSpec, entry: str = "n1") -> GraphSpec:
    """Build a minimal single-node graph."""
    return GraphSpec(
        id="g1",
        goal_id="goal-1",
        nodes=[node_spec],
        edges=[],
        entry_node=entry,
        terminal_nodes=[entry],
    )


def _cycle_graph(node_spec: NodeSpec, max_steps: int = 10) -> GraphSpec:
    """A→B→A cycle graph where A is the node under test (visited multiple times).

    B is a simple bounce node. Neither node is terminal — max_steps guards termination.
    A's max_node_visits controls how many times it actually executes.
    """
    bounce_spec = NodeSpec(
        id="bounce",
        name="Bounce",
        description="passthrough for cycle",
        node_type="event_loop",
        input_keys=[],
        output_keys=["result"],
        max_retries=0,
        max_node_visits=0,  # unlimited
    )
    return GraphSpec(
        id="g-cycle",
        goal_id="goal-cycle",
        nodes=[node_spec, bounce_spec],
        edges=[
            EdgeSpec(
                id="a_to_bounce",
                source=node_spec.id,
                target="bounce",
                condition=EdgeCondition.ON_SUCCESS,
            ),
            EdgeSpec(
                id="bounce_to_a",
                source="bounce",
                target=node_spec.id,
                condition=EdgeCondition.ON_SUCCESS,
            ),
        ],
        entry_node=node_spec.id,
        terminal_nodes=[],  # max_steps is the guard
        max_steps=max_steps,
    )


GOAL = Goal(id="goal-1", name="test", description="test")


# ---------------------------------------------------------------------------
# Test 1: Per-node model selection via NodeSpec.model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_spec_model_selects_per_node_llm():
    """When NodeSpec.model is set, the node receives an LLM with that model."""
    node_a = ModelCapturingNode()
    node_b = ModelCapturingNode()

    spec_a = NodeSpec(
        id="a",
        name="Node A",
        description="uses model-a",
        node_type="event_loop",
        model="model-a",
        input_keys=[],
        output_keys=["result"],
        max_retries=0,
    )
    spec_b = NodeSpec(
        id="b",
        name="Node B",
        description="uses model-b",
        node_type="event_loop",
        model="model-b",
        input_keys=[],
        output_keys=["result"],
        max_retries=0,
    )

    graph = GraphSpec(
        id="g-two",
        goal_id="goal-1",
        nodes=[spec_a, spec_b],
        edges=[
            EdgeSpec(id="a_to_b", source="a", target="b", condition=EdgeCondition.ON_SUCCESS),
        ],
        entry_node="a",
        terminal_nodes=["b"],
    )

    executor = GraphExecutor(
        runtime=DummyRuntime(),
        llm=MockLLMProvider(model="default-model"),
        node_registry={"a": node_a, "b": node_b},
    )

    result = await executor.execute(graph=graph, goal=GOAL)
    assert result.success is True
    assert node_a.captured_model == "model-a"
    assert node_b.captured_model == "model-b"


# ---------------------------------------------------------------------------
# Test 2: NodeSpec.model=None uses the executor's default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_spec_model_none_uses_default():
    """When NodeSpec.model is None, the node receives the default LLM."""
    node = ModelCapturingNode()

    spec = NodeSpec(
        id="n1",
        name="Default Node",
        description="no model override",
        node_type="event_loop",
        input_keys=[],
        output_keys=["result"],
        max_retries=0,
    )

    executor = GraphExecutor(
        runtime=DummyRuntime(),
        llm=MockLLMProvider(model="my-default"),
        node_registry={"n1": node},
    )

    result = await executor.execute(graph=_simple_graph(spec), goal=GOAL)
    assert result.success is True
    assert node.captured_model == "my-default"


# ---------------------------------------------------------------------------
# Test 3: Degradation triggers when budget exceeded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degradation_policy_triggers_on_budget():
    """After exceeding token_budget on visit 1, visit 2 gets the fallback model."""
    node = TokenBurningNode(tokens=150)

    spec = NodeSpec(
        id="n1",
        name="Expensive Node",
        description="burns tokens",
        node_type="event_loop",
        input_keys=[],
        output_keys=["result"],
        max_retries=0,
        max_node_visits=2,
        degradation_policy=DegradationPolicy(
            token_budget=100,
            fallback_models=["cheap-model"],
        ),
    )

    executor = GraphExecutor(
        runtime=DummyRuntime(),
        llm=MockLLMProvider(model="expensive-model"),
        node_registry={"n1": node, "bounce": BounceNode()},
    )

    result = await executor.execute(graph=_cycle_graph(spec), goal=GOAL, validate_graph=False)
    assert result.success is True

    # Visit 1: default model (budget not yet exceeded)
    assert node.captured_models[0] == "expensive-model"
    # Visit 2: degraded to cheap model (150 >= 100 budget)
    assert node.captured_models[1] == "cheap-model"

    # ExecutionResult should reflect degradation
    assert result.node_cost_states["n1"]["degraded"] is True
    assert result.node_cost_states["n1"]["tokens_used"] == 300  # 150 * 2 visits


# ---------------------------------------------------------------------------
# Test 4: Degradation NOT triggered when under budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degradation_policy_not_triggered_under_budget():
    """When token spend is under budget, the node keeps its original model."""
    node = TokenBurningNode(tokens=50)

    spec = NodeSpec(
        id="n1",
        name="Cheap Node",
        description="low token usage",
        node_type="event_loop",
        input_keys=[],
        output_keys=["result"],
        max_retries=0,
        max_node_visits=2,
        degradation_policy=DegradationPolicy(
            token_budget=1000,
            fallback_models=["cheap-model"],
        ),
    )

    executor = GraphExecutor(
        runtime=DummyRuntime(),
        llm=MockLLMProvider(model="expensive-model"),
        node_registry={"n1": node, "bounce": BounceNode()},
    )

    result = await executor.execute(graph=_cycle_graph(spec), goal=GOAL, validate_graph=False)
    assert result.success is True

    # Both visits should use the original model
    assert node.captured_models[0] == "expensive-model"
    assert node.captured_models[1] == "expensive-model"

    # No degradation
    assert result.node_cost_states["n1"]["degraded"] is False


# ---------------------------------------------------------------------------
# Test 5: Fallback chain tries models in order
# ---------------------------------------------------------------------------


class SelectiveMockProvider(MockLLMProvider):
    """Mock provider where with_model() fails for certain models."""

    def __init__(self, model: str = "default", blocked_models: set[str] | None = None):
        super().__init__(model=model)
        self._blocked = blocked_models or set()

    def with_model(self, model: str) -> "MockLLMProvider":
        if model in self._blocked:
            raise NotImplementedError(f"{model} is blocked")
        return MockLLMProvider(model=model)


@pytest.mark.asyncio
async def test_degradation_fallback_chain_tries_models_in_order():
    """If the first fallback model fails with_model(), the second is tried."""
    node = TokenBurningNode(tokens=200)

    spec = NodeSpec(
        id="n1",
        name="Chain Node",
        description="tests fallback chain",
        node_type="event_loop",
        input_keys=[],
        output_keys=["result"],
        max_retries=0,
        max_node_visits=2,
        degradation_policy=DegradationPolicy(
            token_budget=100,
            fallback_models=["blocked-model", "allowed-model"],
        ),
    )

    executor = GraphExecutor(
        runtime=DummyRuntime(),
        llm=SelectiveMockProvider(model="primary", blocked_models={"blocked-model"}),
        node_registry={"n1": node, "bounce": BounceNode()},
    )

    result = await executor.execute(graph=_cycle_graph(spec), goal=GOAL, validate_graph=False)
    assert result.success is True

    # Visit 1: primary model
    assert node.captured_models[0] == "primary"
    # Visit 2: should skip "blocked-model" and land on "allowed-model"
    assert node.captured_models[1] == "allowed-model"


# ---------------------------------------------------------------------------
# Test 6: Provider without with_model() — graceful fallback
# ---------------------------------------------------------------------------


class BareProvider(LLMProvider):
    """Minimal provider that does NOT implement with_model()."""

    def __init__(self, model: str = "bare"):
        self.model = model

    def complete(self, messages, system="", tools=None, max_tokens=1024,
                 response_format=None, json_mode=False, max_retries=None):
        return LLMResponse(content="ok", model=self.model)


@pytest.mark.asyncio
async def test_with_model_not_implemented_graceful_fallback():
    """If provider doesn't support with_model(), node uses default — no crash."""
    node = ModelCapturingNode()

    spec = NodeSpec(
        id="n1",
        name="Bare Node",
        description="uses bare provider",
        node_type="event_loop",
        model="custom-model",  # Requests a different model
        input_keys=[],
        output_keys=["result"],
        max_retries=0,
    )

    executor = GraphExecutor(
        runtime=DummyRuntime(),
        llm=BareProvider(model="bare"),
        node_registry={"n1": node},
    )

    result = await executor.execute(graph=_simple_graph(spec), goal=GOAL)
    assert result.success is True
    # Should fall back to default since with_model() raises NotImplementedError
    assert node.captured_model == "bare"


# ---------------------------------------------------------------------------
# Test 7: node_cost_states accumulated across visits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_cost_states_accumulated_across_visits():
    """Token spend accumulates correctly across multiple node visits."""
    node = TokenBurningNode(tokens=75)

    spec = NodeSpec(
        id="n1",
        name="Multi-visit Node",
        description="visited multiple times",
        node_type="event_loop",
        input_keys=[],
        output_keys=["result"],
        max_retries=0,
        max_node_visits=3,
    )

    executor = GraphExecutor(
        runtime=DummyRuntime(),
        llm=MockLLMProvider(model="default"),
        node_registry={"n1": node, "bounce": BounceNode()},
    )

    result = await executor.execute(graph=_cycle_graph(spec, max_steps=20), goal=GOAL, validate_graph=False)
    assert result.success is True

    # 3 visits × 75 tokens = 225
    assert result.node_cost_states["n1"]["tokens_used"] == 225


# ---------------------------------------------------------------------------
# Test 8: model_used field on NodeResult
# ---------------------------------------------------------------------------


def test_model_used_field_on_node_result():
    """NodeResult.model_used defaults to '' and can be set."""
    # Default
    r1 = NodeResult(success=True)
    assert r1.model_used == ""

    # Explicit
    r2 = NodeResult(success=True, model_used="gpt-4o")
    assert r2.model_used == "gpt-4o"


# ---------------------------------------------------------------------------
# Test 9: Composition with a mock fallback wrapper (#3801 compat)
# ---------------------------------------------------------------------------


class FallbackWrapper(LLMProvider):
    """Simulates a #3801-style fallback provider that wraps another provider.

    The key contract: with_model() must return a NEW FallbackWrapper
    (preserving the fallback chain) rather than bypassing the wrapper.
    """

    def __init__(self, inner: LLMProvider, chain: list[str] | None = None):
        self.inner = inner
        self.chain = chain or []
        self.model = getattr(inner, "model", "unknown")

    def complete(self, messages, system="", tools=None, max_tokens=1024,
                 response_format=None, json_mode=False, max_retries=None):
        return self.inner.complete(
            messages, system, tools, max_tokens, response_format, json_mode, max_retries
        )

    def with_model(self, model: str) -> "FallbackWrapper":
        """Preserve the fallback chain while switching the primary model."""
        new_inner = self.inner.with_model(model)
        return FallbackWrapper(inner=new_inner, chain=self.chain)


@pytest.mark.asyncio
async def test_composition_with_fallback_wrapper():
    """with_model() on a wrapper preserves the wrapper type (not bypassed)."""
    node = ModelCapturingNode()

    spec = NodeSpec(
        id="n1",
        name="Wrapped Node",
        description="uses fallback wrapper",
        node_type="event_loop",
        model="sonnet",
        input_keys=[],
        output_keys=["result"],
        max_retries=0,
    )

    inner = MockLLMProvider(model="default")
    wrapper = FallbackWrapper(inner=inner, chain=["haiku", "flash"])

    executor = GraphExecutor(
        runtime=DummyRuntime(),
        llm=wrapper,
        node_registry={"n1": node},
    )

    result = await executor.execute(graph=_simple_graph(spec), goal=GOAL)
    assert result.success is True

    # The node should have received "sonnet" as its model
    assert node.captured_model == "sonnet"
