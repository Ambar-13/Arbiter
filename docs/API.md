# API Reference

## Core (`constrai.formal`)

### `State(variables: dict)`

Immutable state snapshot. Deep-copies on creation, returns copies from `get()`.

```python
s = State({"x": 1, "items": [1, 2]})
s.get("x")                    # 1
s.get("missing", "default")   # "default"
s.with_updates({"x": 2})      # New State with x=2
s.without_keys({"x"})         # New State without x
s.keys()                       # {"x", "items"}
s.to_dict()                    # Deep copy of internal dict
s == State({"x": 1, "items": [1, 2]})  # True (value equality)
```

### `Effect(key: str, mode: str, value: Any = None)`

Declarative state mutation. Modes: `set`, `increment`, `decrement`, `multiply`, `append`, `remove`, `delete`.

```python
Effect("count", "increment", 1).apply(5)   # 6
Effect("status", "set", "done").apply(None) # "done"
Effect("items", "append", "x").apply([1])   # [1, "x"]
Effect("temp", "delete").apply("anything")  # _DELETED sentinel
```

### `ActionSpec(id, name, description, effects, cost, ...)`

Action definition. Data, not code.

```python
ActionSpec(
    id="deploy",
    name="Deploy to Production",
    description="Push build to prod servers",
    effects=(Effect("deployed", "set", True),),
    cost=10.0,
    risk_level="high",       # "low", "medium", "high", "critical"
    reversible=False,         # Can it be undone?
    category="deploy",        # For grouping
    preconditions_text="Tests must pass",
    postconditions_text="App is live",
)
```

Key methods:
- `simulate(state) -> State`: Apply effects to state (returns new State)
- `compute_inverse(state) -> tuple[Effect]`: Compute undo effects

### `Invariant(name, predicate, description, severity="critical")`

Safety predicate. Must return True for every reachable state.

```python
Invariant("max_cost", lambda s: s.get("spent", 0) <= 100, "Budget cap")
```

The predicate receives a State. If it throws an exception, that counts as a violation (safe default).

### `BudgetController(total: float)`

```python
bc = BudgetController(100.0)
bc.can_afford(50.0)        # (True, "")
bc.charge("action_1", 50.0)
bc.spent                    # 50.0
bc.remaining                # 50.0
bc.can_afford(60.0)        # (False, "Cannot afford 60.0, remaining 50.0")
bc.refund("action_1", 25.0)  # Partial refund
```

### `SafetyKernel(budget, invariants, min_action_cost=0.01)`

The gate. Everything goes through here.

```python
kernel = SafetyKernel(budget=100.0, invariants=[inv1, inv2], min_action_cost=1.0)
verdict = kernel.evaluate(state, action)  # SafetyVerdict
if verdict.approved:
    new_state, cost = kernel.execute(state, action)
```

- `evaluate(state, action) -> SafetyVerdict`: Simulate and check. Does not modify state.
- `execute(state, action) -> (State, float)`: Commit the transition. Modifies budget and trace.
- `rollback(prev, current, action) -> State`: Undo the transition.
- `kernel.trace.verify_integrity() -> (bool, str)`: Check hash chain.
- `kernel.max_steps`: Computed as `⌊budget/min_cost⌋`.

### `GuaranteeLevel`

Enum for tagging claims: `PROVEN`, `CONDITIONAL`, `EMPIRICAL`, `HEURISTIC`.

## Reasoning (`constrai.reasoning`)

### `Belief(alpha=1.0, beta=1.0)`

Beta distribution for action success probability.

```python
b = Belief()                # Uniform prior
b = b.observe(True)         # α += 1
b.mean                      # α / (α + β)
b.variance
b.confidence_interval(0.95)  # (low, high)
```

### `BeliefState()`

Manages beliefs for multiple actions.

```python
bs = BeliefState()
bs.observe("action:deploy:succeeds", True)
bs.get("action:deploy:succeeds").mean  # Updated belief
bs.set_prior("action:deploy:succeeds", alpha=1.0, beta=5.0)
```

### `CausalGraph()`

DAG of action dependencies.

```python
cg = CausalGraph()
cg.add_action("build", [])
cg.add_action("test", [("build", "need build")])
cg.can_execute("test")      # (False, ["build"])
cg.mark_completed("build")
cg.can_execute("test")      # (True, [])
cg.has_cycle()               # False
cg.topological_order()       # ["build", "test"]
```

### `MockLLMAdapter()`

Deterministic mock. Picks novel READY actions first, falls back to first available. Useful for testing without an LLM.

### `LLMAdapter` (Protocol)

Interface for real LLMs. Implement one method:

```python
class MyLLM:
    def complete(self, prompt: str, system_prompt: str = "",
                 temperature: float = 0.3, max_tokens: int = 2000) -> str:
        ...
```

## Orchestrator (`constrai.orchestrator`)

### `TaskDefinition(...)`

Everything the orchestrator needs to run a task.

```python
TaskDefinition(
    goal="description",
    initial_state=State({...}),
    available_actions=[...],
    invariants=[...],
    budget=100.0,
    goal_predicate=lambda s: s.get("done"),
    goal_progress_fn=lambda s: ...,      # Optional, 0.0–1.0
    dependencies={"action_id": [("dep_id", "reason")]},
    priors={"action:X:succeeds": (alpha, beta)},
    min_action_cost=0.5,
    max_consecutive_failures=10,
    stuck_patience=5,
    risk_aversion=0.5,
    system_prompt="extra context for LLM",
)
```

### `Orchestrator(task, llm=None)`

```python
engine = Orchestrator(task)          # MockLLM
engine = Orchestrator(task, llm=my_llm)

result = engine.run()  # Returns ExecutionResult
```

### `ExecutionResult`

```python
result.goal_achieved        # bool
result.termination_reason   # TerminationReason enum
result.final_state          # State
result.total_cost           # float
result.total_steps          # int
result.goal_progress        # 0.0–1.0
result.summary()            # Formatted string
result.metrics              # Dict with detailed stats
```

### `TerminationReason`

`GOAL_ACHIEVED`, `BUDGET_EXHAUSTED`, `STEP_LIMIT`, `LLM_STOP`, `STUCK`, `MAX_FAILURES`, `ERROR`.

## Hardening (`constrai.hardening`)

### `SubprocessAttestor(name, command, success_pattern="", working_dir=None)`

```python
att = SubprocessAttestor("test_check", ["npm", "test"],
                         success_pattern="All tests passed",
                         working_dir="/app")
result = att.verify(state, "goal", timeout_s=30.0)  # Attestation
```

Command must be in the allowlist. `shell=False` enforced.

### `PredicateAttestor(name, check_fn)`

```python
att = PredicateAttestor("db_check",
    lambda: (os.path.exists("/data/db"), "DB file status"))
```

### `AttestationGate(quorum=1)`

```python
gate = AttestationGate(quorum=2)
gate.add_attestor(att1)
gate.add_attestor(att2)
verified, attestations = gate.verify_goal(state, "goal", predicate)
```

### `ReadinessProbe(name, check_fn, initial_delay_s, interval_s, max_retries, backoff_factor)`

```python
probe = ReadinessProbe("db_ready",
    lambda: (db.ping(), "ping result"),
    initial_delay_s=2.0, interval_s=1.0, max_retries=10, backoff_factor=1.5)
ready, detail, wait_time = probe.wait_until_ready(use_real_time=True)
```

### `TemporalDependency(required_action_id, reason, readiness_probe=None, min_delay_s=0)`

### `TemporalCausalGraph(base_graph)`

Wraps a CausalGraph with temporal awareness.

### `CostAwarePriorFactory(total_budget, pessimism_factor=5.0)`

```python
factory = CostAwarePriorFactory(100.0, pessimism_factor=5.0)
alpha, beta, tier = factory.compute_prior(action)  # tier: EXPLORE/CAUTIOUS/GUARDED/GATED
factory.authorize("expensive_action")               # Unlock GATED action
tiers = factory.initialize_beliefs(belief_state, actions)
```

### `EnvironmentReconciler(drift_threshold=0.0, halt_on_drift=True)`

```python
recon = EnvironmentReconciler()
recon.add_probe(EnvironmentProbe("file_count", lambda: len(os.listdir("."))))
result = recon.reconcile(expected_state, "post-deploy")  # Raises on drift
```

### `MultiDimensionalAttestor(name, threshold=0.7)`

```python
md = MultiDimensionalAttestor("quality")
md.add_check(QualityDimension.EXISTENCE, 1.0, lambda: (1.0, "exists"))
md.add_check(QualityDimension.QUALITY, 1.0, lambda: (0.8, "good"))
result = md.verify(state, "goal")  # Fails if any dimension = 0
```

### `ResourceTracker()` / `ResourceDescriptor`

Tracks external resource lifecycles with validated state transitions.

```python
rt = ResourceTracker()
rt.register(ResourceDescriptor(kind="database", identifier="db-1"))
rt.transition("db-1", ResourceState.CREATING)  # (True, "...")
rt.transition("db-1", ResourceState.ABSENT)    # (False, "Invalid transition")
```

Valid transitions: ABSENT→CREATING→READY→DELETING→ABSENT, with FAILED and DEGRADED branches.
