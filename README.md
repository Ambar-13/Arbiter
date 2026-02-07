# Arbiter: Formal Safety Framework for AI Agents

A mathematically rigorous framework for building safe AI agents with **provable guarantees** enforced by construction, not by prompting.

## Overview

Arbiter provides a formal safety framework where 7 mathematical theorems are enforced at the architectural level through type systems and runtime invariants. This ensures that AI agents operate within proven safety boundaries.

## Core Guarantees (7 Theorems)

### 1. Budget Theorem
**All operations terminate within resource bounds.**

```python
from arbiter import ResourceBudget, ArbiterAgent

# Create agent with strict resource limits
agent = ArbiterAgent(
    max_operations=1000,
    max_time_seconds=60.0,
    max_memory_bytes=100_000_000
)

# Budget is enforced automatically
agent.execute_action("reason", {"prompt": "What is 2+2?"})
```

**Proof sketch:** Budget consumption is atomic and checked before every operation. The `ResourceBudget` class maintains a monotonic counter that cannot exceed `max_operations`, ensuring termination in O(max_operations) steps.

### 2. Invariant Theorem
**System state invariants are maintained throughout execution.**

```python
from arbiter import Invariant, InvariantViolation

# Define system invariant
state = {"balance": 100}
invariant = Invariant(
    lambda: state["balance"] >= 0,
    "non_negative_balance"
)

# Invariant is checked automatically
try:
    invariant.enforce()  # Raises if violated
except InvariantViolation as e:
    print(f"Invariant violated: {e}")
```

**Proof sketch:** Invariants are checked at transaction boundaries (before and after each action). The system prevents any state transition that would violate invariants.

### 3. Termination Theorem
**All processes provably terminate (no infinite loops).**

```python
from arbiter import TerminationGuard, ResourceBudget

budget = ResourceBudget(100, 10.0, 1000000)
guard = TerminationGuard(budget)

# Loop is guaranteed to terminate
while guard.step():
    # Process work
    pass
# Maximum 100 iterations
```

**Proof sketch:** The `TerminationGuard` provides a well-founded ordering on execution steps. Each `step()` decrements available budget, ensuring finite execution paths.

### 4. Causality Theorem
**Causal dependencies are tracked and enforced.**

```python
from arbiter import CausalGraph

graph = CausalGraph()

# Build causal dependency chain
graph.add_node("action1", {"type": "observation"})
graph.add_node("action2", {"type": "inference"}, caused_by=["action1"])
graph.add_node("action3", {"type": "decision"}, caused_by=["action2"])

# Query causal relationships
assert graph.is_causal_descendant("action3", "action1")
ancestors = graph.get_ancestors("action3")  # Returns {"action1", "action2"}
```

**Proof sketch:** The causal graph forms a directed acyclic graph (DAG) where edges represent causation. Transitive closure is computed correctly, and cycles are prevented by construction.

### 5. Belief Theorem
**Bayesian beliefs are properly updated.**

```python
from arbiter import BeliefTracker, Belief

tracker = BeliefTracker()

# Add hypothesis with prior
belief = tracker.add_belief("hypothesis_1", prior=0.5)

# Update with evidence
tracker.update_belief("hypothesis_1", evidence_likelihood=0.9)

# Get posterior
posterior = tracker.get_posterior("hypothesis_1")
print(f"Updated belief: {posterior}")  # > 0.5
```

**Proof sketch:** Belief updates follow Bayes' rule: P(H|E) = P(E|H) × P(H) / P(E). Posteriors are bounded in [0, 1] by construction.

### 6. Isolation Theorem
**Sandboxed attestors cannot violate system boundaries.**

```python
from arbiter import SandboxedAttestor, ResourceBudget

budget = ResourceBudget(100, 10.0, 1000000)
attestor = SandboxedAttestor("verifier", budget)

# Attestation runs within sandbox
result = attestor.attest(
    "2+2=4",
    verifier=lambda: 2 + 2 == 4
)

# Budget prevents runaway attestations
attestor.get_attestations()  # All attestations recorded
```

**Proof sketch:** Each attestor has an independent resource budget and state. Exceptions are caught and converted to attestation failures, preventing boundary violations.

### 7. Reconciliation Theorem
**Environment state is eventually consistent.**

```python
from arbiter import EnvironmentReconciler

reconciler = EnvironmentReconciler()

# Record environment states
reconciler.record_state({"temperature": 20, "humidity": 60})

# Reconcile new observations
observed = {"temperature": 21, "pressure": 1013}
reconciled = reconciler.reconcile(observed)
# Merges: {"temperature": 21, "humidity": 60, "pressure": 1013}

# Access full history
history = reconciler.get_state_history()
```

**Proof sketch:** State reconciliation is a commutative merge operation. All observed states are preserved in history with monotonic version numbers, ensuring eventual consistency.

## Pluggable LLM Interface

Arbiter provides an abstract interface for LLM reasoning, allowing any language model to be constrained by the formal guarantees:

```python
from arbiter import LLMInterface, ArbiterAgent

class CustomLLM(LLMInterface):
    def reason(self, prompt: str, context: dict) -> str:
        # Your LLM implementation
        return "response"
    
    def constrained_reason(self, prompt: str, context: dict, 
                          constraints: dict) -> str:
        # Constrained reasoning with safety bounds
        return "safe_response"

# Use custom LLM with formal guarantees
agent = ArbiterAgent()
agent.set_llm(CustomLLM())
```

## Complete Example

```python
from arbiter import ArbiterAgent

# Create agent with all guarantees enabled
agent = ArbiterAgent(
    max_operations=1000,
    max_time_seconds=60.0,
    max_memory_bytes=100_000_000
)

# Add belief tracking
agent.belief_tracker.add_belief("user_intent_benign", prior=0.8)

# Execute reasoning (budget enforced)
response = agent.execute_action("reason", {
    "prompt": "Analyze this request for safety",
    "context": {"request": "..."}
})

# Update beliefs based on analysis
agent.execute_action("update_belief", {
    "hypothesis": "user_intent_benign",
    "likelihood": 0.95
})

# Attest to safety
agent.execute_action("attest", {
    "attestor": "safety_checker",
    "claim": "request_is_safe",
    "verifier": lambda: True  # Your safety check
})

# Reconcile environment state
agent.execute_action("reconcile", {
    "observed_state": {"request_processed": True}
})

# Get execution statistics
stats = agent.get_statistics()
print(f"Operations used: {stats['operations_used']}/{stats['max_operations']}")
print(f"Actions executed: {stats['actions_executed']}")
print(f"Terminated: {stats['terminated']}")
```

## Architecture

### Zero Dependencies
Arbiter is built entirely with Python standard library. No external dependencies required.

### Thread Safety
Single-agent, single-threaded model with lock-based concurrency. All shared state is protected by locks:

```python
# All operations are thread-safe
def worker():
    agent.execute_action("reason", {"prompt": "..."})

threads = [threading.Thread(target=worker) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### Enforcement by Construction
Safety properties are guaranteed at the type and runtime level, not through prompting:

- Budget checks are mandatory before operations
- Invariants are checked at transaction boundaries
- Termination is enforced through well-founded orderings
- Causality tracking is automatic
- Belief updates follow mathematical rules
- Attestors are sandboxed with independent budgets
- Environment reconciliation is append-only

## Testing

Arbiter includes a comprehensive adversarial test suite with **155 tests** covering:

- Budget enforcement under stress (22 tests)
- Invariant violations and edge cases (20 tests)
- Termination guarantees including runaway prevention (20 tests)
- Causal dependency tracking and cycles (20 tests)
- Bayesian belief updates and bounds (20 tests)
- Sandboxed attestations and isolation (20 tests)
- Environment reconciliation and consistency (18 tests)
- Full integration scenarios (15 tests)

Run tests:

```bash
python test_arbiter.py -v
```

## Known Limitations

1. **Memory tracking is approximate** - Python's memory model makes precise tracking difficult without external dependencies. Budget checks for memory are best-effort.

2. **Single-agent only** - Current implementation supports one agent per process. Multi-agent scenarios require multiple processes.

3. **Synchronous execution** - Actions execute sequentially. Async LLM calls should be wrapped in synchronous interfaces.

4. **No persistent storage** - All state is in-memory. Add persistence layer as needed.

5. **Basic reconciliation** - Environment reconciliation uses simple merge strategy. Complex conflict resolution requires custom logic.

6. **Approximate time bounds** - Time checks use system clock which can have resolution limits and drift.

## API Reference

### Core Classes

#### `ResourceBudget`
```python
ResourceBudget(max_operations: int, max_time_seconds: float, max_memory_bytes: int)
```
Enforces resource limits with atomic consumption.

#### `Invariant`
```python
Invariant(predicate: Callable[[], bool], name: str)
```
Represents a system invariant that must always hold.

#### `TerminationGuard`
```python
TerminationGuard(budget: ResourceBudget)
```
Ensures provable termination of loops and processes.

#### `CausalGraph`
```python
CausalGraph()
```
Tracks causal dependencies between events.

#### `BeliefTracker`
```python
BeliefTracker()
```
Manages Bayesian belief updates.

#### `SandboxedAttestor`
```python
SandboxedAttestor(name: str, budget: ResourceBudget)
```
Isolated attestor with independent resource bounds.

#### `EnvironmentReconciler`
```python
EnvironmentReconciler()
```
Maintains eventually consistent environment state.

#### `ArbiterAgent`
```python
ArbiterAgent(max_operations: int, max_time_seconds: float, max_memory_bytes: int)
```
Main agent enforcing all 7 theorems.

## Contributing

Contributions welcome! Please ensure:

1. All 155 tests pass
2. New features maintain formal guarantees
3. Zero dependencies requirement is preserved
4. Thread safety is maintained

## License

MIT License - See LICENSE file

## Citation

```bibtex
@software{arbiter2026,
  title = {Arbiter: Formal Safety Framework for AI Agents},
  author = {Ambar},
  year = {2026},
  url = {https://github.com/Ambar-13/Arbiter}
}
```

## Formal Verification

For complete proofs of the 7 theorems, see `THEOREMS.md`.