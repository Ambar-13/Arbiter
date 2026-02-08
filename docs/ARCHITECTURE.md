# Architecture

## The problem ConstrAI solves

AI agents need to take actions in the real world. Those actions cost money, can break things, and can't always be undone. Current approaches either:

1. Let the LLM do whatever and hope for the best (AutoGPT, LangChain agents)
2. Hardcode rules that the LLM must follow (brittle, can't handle novel situations)
3. Use RLHF to make the LLM "want" to be safe (constrains text output, not actions)

ConstrAI takes a different approach: the LLM reasons freely, but its decisions pass through a formal verification layer before execution. The LLM can't bypass the math. The math can't bypass the LLM. Both must agree.

## Layer 0: Safety Kernel (`formal.py`)

This is the non-negotiable foundation. Nothing executes without passing through here.

### State

State is an immutable dictionary. Every mutation creates a new State object. The internal storage uses `MappingProxyType` (read-only dict view), so even if you get a reference to it, you can't mutate it.

```python
s = State({"x": 1, "y": [1, 2, 3]})
s.get("y").append(4)  # Returns a copy. Original unchanged.
s._vars["x"] = 99     # TypeError: mappingproxy doesn't support assignment
```

Why immutable state matters: it makes T7 (rollback) trivial. If you never mutate, you can always go back.

### Effects

Actions don't contain code. They contain declarative Effect specs:

```python
Effect("counter", "increment", 1)
Effect("status", "set", "ready")
Effect("items", "append", "new_item")
Effect("temp_file", "delete")
```

This is deliberate. Code can't be formally verified. Data can. The kernel simulates effects on the state dictionary without executing anything real. If the simulation violates an invariant, the action is blocked before any side effect occurs.

Supported operations: `set`, `increment`, `decrement`, `multiply`, `append`, `remove`, `delete`.

### Invariants

User-defined predicates that must hold on every reachable state:

```python
Invariant("budget_positive", lambda s: s.get("balance", 0) >= 0)
Invariant("max_instances", lambda s: s.get("count", 0) <= 10)
```

The kernel checks invariants on the simulated next-state. If any invariant would be violated, the action is rejected and the current state is unchanged.

### Budget Controller

Tracks spending with a simple ledger. `can_afford(cost)` checks before `charge(id, cost)` commits. Negative costs are rejected (assertion). This gives T1 (budget safety) and T4 (monotone spend) by construction.

### Execution Trace

Every action (approved or rejected) is recorded as a `TraceEntry` with a SHA-256 hash chain. Each entry's hash includes the previous entry's hash, creating a tamper-evident log. If anyone modifies a historical entry, `verify_integrity()` detects it.

### Safety Kernel

The gate. For every proposed action:

1. Check budget (`can_afford`)
2. Simulate state transition (`action.simulate(state)`)
3. Check all invariants on simulated state
4. If everything passes: commit (charge budget, update state, append trace)
5. If anything fails: reject (state and budget unchanged)

This simulate-then-commit pattern is what gives T3 (invariant preservation) and T5 (atomicity).

## Layer 1: Reasoning Engine (`reasoning.py`)

This is where intelligence lives. The safety kernel is a cage; the reasoning engine is the brain inside it.

### Bayesian Beliefs

For each action, ConstrAI tracks a Beta(α, β) distribution representing "how likely is this action to succeed."

- `observe(True)`: α += 1 (saw it work)
- `observe(False)`: β += 1 (saw it fail)
- Mean = α/(α+β), variance decreases with more observations

This is exact Bayesian inference. No approximations, no neural nets.

### Causal Graph

A DAG tracking which actions depend on which. If "deploy" depends on "test" and "test" hasn't completed, "deploy" is marked BLOCKED regardless of what the LLM wants.

The graph is populated from the `TaskDefinition.dependencies` dict, and can be extended at runtime by the Dynamic Dependency Discovery system.

### Action Value Computer

Scores each action on 5 dimensions:

- **Expected progress**: How much closer does this get us to the goal?
- **Information gain**: How much do we learn from trying this?
- **Cost ratio**: How expensive relative to remaining budget?
- **Risk**: How close are we to violating an invariant?
- **Opportunity cost**: What do we give up by choosing this?

These scores are fed to the LLM as part of the structured prompt. The LLM doesn't guess — it reasons over pre-computed analysis.

### Structured Prompts

The LLM receives a decision-support document, not a vague instruction. It contains: the goal, current state, all available actions ranked by value with READY/BLOCKED status, belief summaries, budget info, and recent history. The LLM returns JSON with a specific schema. If it returns garbage, the parser catches it and falls back to the highest-value READY action.

## Layer 2: Orchestrator (`orchestrator.py`)

The main loop:

```
while not done:
    1. Get available actions (affordable + dependency-ready)
    2. Compute action values
    3. Ask LLM to select (with structured prompt)
    4. Validate LLM response (catch hallucinations)
    5. Safety kernel evaluates (simulate + check invariants)
    6. If approved: execute, update state, update beliefs
    7. If rejected: record failure, continue
    8. Check termination (goal achieved? budget gone? stuck?)
```

Termination conditions: goal achieved, budget exhausted, step limit reached, LLM requests stop, stuck (no progress for N steps), too many consecutive failures.

If the LLM fails (parse error, hallucination, timeout), the orchestrator falls back to the highest-value READY action from the formal analysis. The system never gets stuck waiting for the LLM.

## Layer 3: Hardening (`hardening.py`)

Fixes for specific vulnerabilities found during adversarial testing.

### Sandboxed Attestors

External goal verification via subprocess. Commands are frozen tuples set at creation time. `shell=False` always. Binary allowlist prevents arbitrary execution. The agent cannot influence what command runs.

### Temporal Dependencies

Extends the causal graph with time awareness. A `ReadinessProbe` polls a resource with exponential backoff until it's truly ready (not just existing). Prevents the "deploy the moment the DB is provisioned but before it accepts connections" race condition.

### Cost-Aware Priors

Bayesian priors proportional to risk. Cheap actions get optimistic priors (try freely). Expensive actions get pessimistic priors (need evidence before trying). Irreversible + critical actions are GATED (blocked until explicitly authorized). This reduces first-strike budget waste.

### Environment Reconciliation

After each action, compares the model's expected state to probes reading the actual environment. If the model says 5 files exist but `ls | wc -l` returns 3, execution halts. Catches the spec-reality gap where the `ActionSpec` doesn't match what actually happens.

### Multi-Dimensional Attestation

Checks goals across multiple quality dimensions (existence, completeness, correctness, quality, safety, regression). Any dimension scoring zero fails the whole attestation. Harder to game than single-metric checks because the agent must satisfy all dimensions simultaneously.

## What the LLM sees

Here's an actual prompt the LLM receives (abbreviated):

```
GOAL: Deploy a web application
STATE: {"built": true, "tested": false, "deployed": false}
PROGRESS: 40.0%
BUDGET: $38.00 remaining of $50.00 (24.0% used)

AVAILABLE ACTIONS (ranked by computed value):

### [READY] Run Tests (id=test)
  Cost: $2.00 | Risk: low
  Value: 0.347 (progress=0.20, info=0.15, cost_ratio=0.05)
  Beliefs: Beta(2,1) → 66.7% success

### [BLOCKED] Deploy (id=deploy) — needs: test
  Cost: $5.00 | Risk: high
  Value: 0.122

Choose an action. Return JSON:
{"chosen_action_id": "...", "reasoning": "...", ...}
```

The LLM isn't guessing. It's reasoning over computed quantities within formal constraints.
