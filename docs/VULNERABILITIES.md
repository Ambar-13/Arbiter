# Known Vulnerabilities

This document lists every known vulnerability in ConstrAI, whether it's been mitigated, and what the residual risk is. No sugarcoating.

## Fixed Vulnerabilities

### 1. Command Injection via SubprocessAttestor

**Status**: Fixed in v2.

**What was wrong**: The original SubprocessAttestor could theoretically execute arbitrary commands if an agent influenced the command arguments.

**What we did**:
- Commands are frozen tuples set at `__init__`. Immutable after creation.
- `shell=False` always. Shell metacharacters (`;`, `|`, `&&`, backticks) are treated as literal strings, not shell operators.
- Binary allowlist: only known-safe commands (`ls`, `cat`, `grep`, `curl`, `npm`, `pytest`, etc.) are permitted. `rm`, `bash`, `sh`, `eval` are blocked.
- Environment sanitized: only `PATH`, `HOME`, `LANG` propagated.
- Output truncated to 8KB to prevent memory exhaustion.

**Residual risk**: If an allowed command itself has a vulnerability (e.g., a `curl` URL that triggers a server-side exploit), ConstrAI doesn't prevent that. We control what binary runs, not what the binary does internally.

**Test coverage**: Chaos fuzzer F6, v2 tests V1.1–V1.8.

### 2. Temporal Blindness (Readiness Race Conditions)

**Status**: Fixed in v2.

**What was wrong**: The causal graph was binary — dependencies were either "met" or "not met". A database could "exist" (provisioned) but not be "ready" (accepting connections). The kernel would approve deployment the instant provisioning completed, causing the app to crash on connection timeout.

**What we did**:
- `TemporalDependency` adds `min_delay_s` (minimum wait after completion) and `ReadinessProbe` (poll until truly ready).
- `ReadinessProbe` uses exponential backoff with bounded max retries. Won't spin forever.
- `TemporalCausalGraph` wraps the base graph. No temporal deps = standard behavior (backward compatible).

**Residual risk**: Probes must check the right thing. If your probe checks `HTTP 200` but the real requirement is "database migrations complete," you still have a gap. Probes are operator-defined, so this is a configuration problem, not a framework bug.

**Test coverage**: v2 tests V2.1–V2.8.

### 3. Bayesian Cold-Start (First-Strike Budget Waste)

**Status**: Fixed in v2.

**What was wrong**: Bayesian beliefs started at Beta(1,1) = uniform prior. An expensive action ($50 on a $100 budget) had a 50% chance of being the first action tried, even with zero evidence it would work.

**What we did**:
- `CostAwarePriorFactory` sets priors proportional to risk:
  - Cheap + reversible: Beta(3, 1), mean 0.75. Try freely.
  - Moderate cost: Beta(2, 1+K·c/B·0.5). Mildly pessimistic.
  - Expensive: Beta(1, 1+K·c/B). Strongly pessimistic.
  - Irreversible + critical: Beta(0.01, 100), mean ≈ 0. Blocked until explicitly authorized via `factory.authorize(action_id)`.
- This is exact Bayesian math. The prior *is* the safety mechanism. First-strike selection probability drops 42–66% for expensive actions (K=5–10).

**Residual risk**: The system still needs to try actions to learn. Cost-aware priors reduce the cost of learning, but don't eliminate it. If you truly cannot afford a single failure on any action, Bayesian learning is fundamentally the wrong tool — you need pre-verified action specs.

**Test coverage**: v2 tests V3.1–V3.8.

### 4. Spec-Reality Gap (Model Drift)

**Status**: Partially fixed in v2.

**What was wrong**: The formal proofs (T1–T7) protect the *internal model*. If an ActionSpec says "create a file" but the real operation also deletes a backup, the model stays clean while reality drifts into a bad state.

**What we did**:
- `EnvironmentReconciler` compares model state to environment probes after every action.
- If `model.file_count = 5` but `len(os.listdir(dir)) = 3`, it raises `EnvironmentDriftError` and halts.
- Configurable drift threshold and halt-on-drift behavior.

**Residual risk**: You can only detect drift on variables you probe. Unprobed variables are invisible. If the ActionSpec has a side effect you didn't anticipate and didn't write a probe for, it goes undetected.

This is the fundamental limitation of the approach: **the map is not the territory**. Environment reconciliation is how we check the territory, but we can only check what we look at.

**Test coverage**: v2 tests V4.1–V4.7.

### 5. Reward Hacking (Proxy Metric Gaming)

**Status**: Mitigated, not eliminated.

**What was wrong**: Single-metric attestors are trivially gameable. Goal is "create 5 files"? Agent creates 5 empty files. Goal is "100% test coverage"? Agent deletes complex code paths.

**What we did**:
- `MultiDimensionalAttestor` checks multiple quality dimensions: existence, completeness, correctness, quality, safety, regression.
- Any dimension scoring zero fails the entire attestation, even if others are perfect.
- Example: 5 empty files → existence=1.0, quality=0.0 → FAILED.

**Residual risk**: This is defense-in-depth, not a proof. A sufficiently clever agent could satisfy all dimensions with low-quality outputs that technically pass each check. More dimensions make this harder, but not provably impossible. We're honest about this — the guarantee is EMPIRICAL, not PROVEN.

The real defense against reward hacking is writing better attestors. ConstrAI gives you the framework; you supply the domain-specific quality checks.

**Test coverage**: v2 tests V5.1–V5.4.

## Unfixed Vulnerabilities (Open Problems)

### 6. Semantic Drift on Subjective Goals

**Status**: Not fixed. No clear fix exists.

**The problem**: If the goal is "write good code" or "optimize performance," what does "good" mean? Formal verification needs binary predicates. Subjective quality is inherently non-binary.

**Our honest position**: ConstrAI works best with goals that have clear, measurable success criteria. For subjective goals, the multi-dimensional attestor helps, but the quality checks are only as good as the human who writes them.

### 7. Multi-Agent Coordination

**Status**: Not implemented.

**The problem**: ConstrAI controls one agent in one execution loop. If you run multiple ConstrAI agents that share resources, they can interfere with each other. No distributed locking, no consensus protocol.

### 8. LLM Adversarial Prompting

**Status**: Structurally mitigated, not eliminated.

**The problem**: A crafted system state or goal description could theoretically trick the LLM into making bad decisions within the legal action space.

**What helps**: The safety kernel limits the damage. Even if the LLM is tricked, it can only select from defined actions, and those actions pass through invariant checking. The worst case is wasted budget, not safety violations.

**What doesn't help**: If all available actions are safe but the LLM picks the wrong *sequence*, the goal fails without any safety violation. Budget waste is the residual risk.

### 9. Deep State Mutation

**Status**: Partially mitigated.

**The problem**: State values that are deeply nested mutable objects (list of lists, dict of dicts) could theoretically be mutated through references.

**What we did**: `State.__init__` deep-copies inputs. `State.get()` deep-copies outputs. `_vars` uses `MappingProxyType`. This handles all normal usage patterns.

**Residual risk**: Python is not memory-safe. If you try hard enough (ctypes, gc manipulation), you can mutate anything. ConstrAI protects against accidental and normal-effort intentional mutation, not against a determined attacker with full access to the Python runtime.

## Vulnerability Classification Summary

| ID | Vulnerability | Status | Guarantee Level |
|----|--------------|--------|----------------|
| 1 | Command Injection | **Fixed** | PROVEN |
| 2 | Temporal Blindness | **Fixed** | CONDITIONAL |
| 3 | Bayesian Cold-Start | **Fixed** | PROVEN |
| 4 | Spec-Reality Gap | **Partial** | CONDITIONAL |
| 5 | Reward Hacking | **Mitigated** | EMPIRICAL |
| 6 | Semantic Drift | Open | — |
| 7 | Multi-Agent | Open | — |
| 8 | LLM Adversarial | Structural | — |
| 9 | Deep State Mutation | Partial | — |
