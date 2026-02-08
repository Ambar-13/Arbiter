# Theorem Proofs

These are the formal guarantees ConstrAI provides, with actual proofs. Each theorem is tagged with its guarantee level: PROVEN means it holds unconditionally by construction, CONDITIONAL means it holds under stated assumptions.

## T1: Budget Safety (PROVEN)

**Statement**: `spent(t) ≤ B₀` for all time steps t, where B₀ is the initial budget.

**Proof by induction on t**:

*Base case*: At t=0, spent(0) = 0 ≤ B₀. ✓

*Inductive step*: Assume spent(t) ≤ B₀. At step t+1:

1. `can_afford(cost)` checks: `self.spent + cost ≤ self.total`
2. If False: action rejected, spent(t+1) = spent(t) ≤ B₀. ✓
3. If True: `charge()` sets spent(t+1) = spent(t) + cost
4. Since can_afford passed: spent(t) + cost ≤ B₀, so spent(t+1) ≤ B₀. ✓

*Key detail*: `charge()` asserts `cost >= 0`, so no negative costs can reduce the spent counter. The check-then-charge pattern is atomic (single-threaded execution). ∎

**What this does NOT guarantee**: That the budget is well-spent. The agent can waste the entire budget on useless actions within the limit.

## T2: Termination (CONDITIONAL)

**Statement**: The execution loop halts in at most `⌊B₀/ε⌋` steps, where ε = `min_action_cost`.

**Assumption**: ε > 0 (there exists a minimum action cost).

**Proof**:

Each executed step charges at least ε to the budget (enforced by the kernel's min_cost check). After k steps, spent ≥ k·ε. When k·ε > B₀, no action can afford even ε, so the loop terminates.

Maximum steps: k_max = ⌊B₀/ε⌋.

The orchestrator also enforces `max_steps = ⌊B₀/ε⌋` as a hard cap. ∎

**What can break this**: If ε = 0, the bound is infinite. The constructor rejects ε ≤ 0, but if you bypass the constructor (don't), you lose this guarantee.

## T3: Invariant Preservation (PROVEN)

**Statement**: If I(s₀) = True for all invariants I, then I(sₜ) = True for all reachable states sₜ.

**Proof by induction on t**:

*Base case*: I(s₀) = True by assumption. ✓

*Inductive step*: Assume I(sₜ) = True. At step t+1:

1. Kernel simulates: `s' = action.simulate(sₜ)`
2. For each invariant I: check `I(s')`.
3. If any I(s') = False: reject action, sₜ₊₁ = sₜ, so I(sₜ₊₁) = I(sₜ) = True. ✓
4. If all I(s') = True: commit, sₜ₊₁ = s', so I(sₜ₊₁) = True. ✓

*Edge case*: If an invariant's check function throws an exception, the kernel treats it as a violation (safe default). ∎

**Critical caveat**: T3 proves invariants hold on the *model* state. If the ActionSpec's effects don't match what actually happens in the real world (the spec-reality gap), invariants hold in the model but reality may diverge. This is addressed by Environment Reconciliation in the hardening layer, but only for probed variables.

## T4: Monotone Resources (PROVEN)

**Statement**: `spent(t) ≤ spent(t+1)` for all t.

**Proof**: The only operation that modifies `spent` is `charge(id, cost)`, which asserts `cost >= 0` and sets `spent += cost`. Since cost ≥ 0, spent can only increase or stay the same. ∎

## T5: Action Atomicity (PROVEN)

**Statement**: If an action is rejected, neither the state nor the budget is modified.

**Proof**:

1. `evaluate()` simulates the action on a *copy* of the state.
2. If simulation fails any check, `evaluate()` returns `SafetyVerdict(approved=False)`.
3. `execute()` is only called if `approved=True`.
4. `execute()` re-checks `can_afford()` before charging (defense in depth).
5. State is immutable — `simulate()` creates a new State object, doesn't modify the original.

Therefore: rejected action → no state change, no budget change. ∎

## T6: Trace Integrity (PROVEN)

**Statement**: The execution trace is append-only and tamper-evident via SHA-256 hash chaining.

**Proof**:

Each `TraceEntry` is a frozen dataclass containing:
- action_id, state_hash, cost, approved, timestamp
- `prev_hash`: SHA-256 of the previous entry's hash

`verify_integrity()` walks the chain and checks each entry's hash against its predecessor. If any entry is modified after creation, the hash chain breaks.

The `entries` property returns a copy of the list, so external code cannot modify the internal trace. ∎

**What this does NOT prevent**: An attacker with access to the Python process's memory can do anything. Hash chains protect against accidental corruption and simple tampering, not against a sophisticated attacker who rewrites the entire chain.

## T7: Rollback Exactness (PROVEN)

**Statement**: `rollback(s_prev, s_new, action) == s_prev` — undoing an action perfectly restores the prior state.

**Proof**:

1. State is immutable. `s_prev` still exists unmodified after `execute()` creates `s_new`.
2. `rollback()` computes the inverse of each Effect:
   - `set(key, val)` → restore from `s_prev.get(key)`
   - `increment(key, n)` → `decrement(key, n)`
   - `append(key, val)` → `remove(key, val)`
   - `delete(key)` → `set(key, s_prev.get(key))`
3. Since State is a deep-copied immutable dict, `s_prev` is exactly what it was before execution.

Therefore: rollback produces a State equal to s_prev. ∎

**Alternative proof**: Since State is immutable and s_prev is never garbage-collected during the orchestration loop, you can just use s_prev directly. The inverse-effects approach exists for cases where you don't want to keep every historical state in memory.

## Hardening Claims

These are weaker than the core theorems. Honest about it.

### H1: Command Isolation (PROVEN)

SubprocessAttestor commands are frozen tuples. `shell=False` prevents metacharacter expansion. Allowlist blocks arbitrary binaries. No method accepts agent input as part of the command.

### H3: Cost-Bounded First-Strike (PROVEN)

With prior Beta(1, K·c/B), the mean probability of selecting an expensive action is 1/(1 + K·c/B). For K=5, c=$50, B=$100: P ≈ 0.29 vs 0.50 (uniform). This is a direct computation on the Beta distribution's mean.

### H2, H4, H5, H6, H7: See VULNERABILITIES.md

These are CONDITIONAL or EMPIRICAL claims. They help, but they can fail under specific conditions. That's documented honestly in the vulnerabilities doc.
