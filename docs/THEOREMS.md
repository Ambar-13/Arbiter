# Formal Claims

All claims are tagged with their epistemic status:

| Level | Meaning |
|-------|---------|
| `PROVEN` | Holds unconditionally by construction |
| `CONDITIONAL` | Proven under stated assumptions |
| `EMPIRICAL` | Measured with statistical confidence |
| `HEURISTIC` | Best-effort; no formal guarantee |

---

## T1 — Budget Safety  `PROVEN`

**Statement:** `spent_net(t) ≤ B₀` for all t.

**Proof** (induction on t):
- Base: `spent_net(0) = 0 ≤ B₀`. ✓
- Inductive step: Assume `spent_net(t) ≤ B₀`.
  Before charging cost `c`, `can_afford(c)` checks that `spent_net(t) + c ≤ B₀`.
  If false → action rejected, `spent_net` unchanged. ✓
  If true → `spent_net(t+1) = spent_net(t) + c ≤ B₀`. ✓ ∎

**Implementation:** Integer millicent arithmetic (× 100,000) avoids floating-point accumulation. A post-commit assertion fires on any violation.

---

## T2 — Termination  `CONDITIONAL`

**Statement:** The system halts in at most `⌊B₀ / ε⌋` steps.

**Assumptions:** All actions cost ≥ ε > 0 (`min_action_cost > 0`). Budget B₀ is finite.

**Proof:**
After n steps, `spent_gross ≥ n × ε` (each step costs at least ε, T4).
When `n > ⌊B₀/ε⌋`: remaining budget `< ε`.
The budget check rejects any action costing ≥ ε. No actions can execute → halt. ∎

**Note:** Emergency actions (T8) bypass this check. They must have `cost = 0` and `effects = ()`.

---

## T3 — Invariant Preservation  `PROVEN`

**Statement:** For every blocking-mode invariant `I`:
`I(s₀) = True ⟹ I(sₜ) = True` for all t.

**Scope:** Only `enforcement="blocking"` invariants. Monitoring-mode invariants log violations but never block.

**Proof** (induction on t):
- Base: `I(s₀) = True` (checked at startup).
- Inductive step: Assume `I(sₜ) = True`.
  - `s' = action.simulate(sₜ)` (pure; no side effects).
  - If `I(s') = False` → action rejected, `sₜ₊₁ = sₜ`, `I(sₜ₊₁) = True`. ✓
  - If `I(s') = True` → committed, `sₜ₊₁ = s'`, `I(sₜ₊₁) = True`. ✓ ∎

---

## T4 — Monotone Gross Spend  `PROVEN`

**Statement:** `spent_gross(t) ≤ spent_gross(t+1)` for all t.

**Proof:** `charge()` asserts `cost ≥ 0` before adding to `spent_gross`. A non-negative addition cannot decrease the total. ∎

**Relevance to T7:** Rollback refunds `spent_net` via a separate `spent_refunded` counter, leaving `spent_gross` unchanged. T7 and T4 coexist without conflict.

---

## T5 — Atomicity  `PROVEN`

**Statement:** Actions are all-or-nothing. Rejected actions leave state, budget, and step count unchanged.

**Proof:**
`evaluate()` simulates on a copy of state — no shared state mutated.
`execute()` only commits after `evaluate()` returns `approved = True`.
If `approved = False`, no commit occurs. ∎

**Concurrency:** Use `evaluate_and_execute_atomic()` in concurrent settings to eliminate the TOCTOU race between `evaluate()` and `execute()`.

---

## T6 — Trace Integrity  `PROVEN`

**Statement:** The execution log is append-only and tamper-evident.

**Proof:**
`TraceEntry` is a frozen dataclass (immutable after construction).
Each entry's hash covers all fields including `prev_hash`, forming a chain.
`verify_integrity()` walks the chain in O(n): any modification to entry `i` breaks the hash at `i+1`. ∎

**Note:** This protects against software-layer tampering, not adversarial memory access (`gc`, `ctypes`).

---

## T7 — Rollback Exactness  `PROVEN`

**Statement:** `rollback(execute(s, a)) == s`.

**Proof:**
`State` is immutable (P1): once constructed, no State object is ever modified.
`state_before` stored at commit time is preserved unchanged by the immutability guarantee.
`apply_rollback()` returns `state_before` directly.
Budget refund uses `budget.refund()`, decrementing `spent_net` without touching `spent_gross` (T4 preserved). ∎

---

## T8 — Emergency Escape  `CONDITIONAL`

**Statement:** The SAFE_HOVER emergency action is always executable.

**Assumptions:**
1. The action is registered via `kernel.register_emergency_action(id)`.
2. The action has `cost = 0.0` and `effects = ()`.

**Proof:**
Emergency actions bypass Check 0 (min cost) and Check 2 (step limit).
With `cost = 0.0`, `can_afford()` always returns True.
With `effects = ()`, `simulate()` returns an identical state, passing all invariant checks trivially. ∎

---

## Reference Monitor Guarantees

### M1 — IFC Lattice  `DETERMINISTIC`

Data flows only to sinks at equal or higher security level (`PUBLIC ⊑ INTERNAL ⊑ PII ⊑ SECRET`). Violations blocked by `ReferenceMonitor.enforce()`.

### M2 — CBF Barrier  `DETERMINISTIC`

`h(s_{t+1}) ≥ (1 - α) × h(s_t)` enforced at each step, where h(s) is the distance-to-boundary function. Bounds the approach rate to resource limits.

### M3 — QP Minimality  `DETERMINISTIC`

When action parameters violate M2, the QP projector finds the minimum-norm modification that restores safety. The repaired action is the closest safe action to the original.

---

## Composition Theorems

### OC-1 — Compositional Safety  `CONDITIONAL`

**Statement:** `Verified(A) ∧ Verified(B) ∧ compatible(A, B) ⟹ Verified(A∘B)`.

Interface compatibility: A's output interface must cover B's required inputs. Postconditions of A must satisfy preconditions of B (checked semantically). The composed task inherits both verification certificates.

### OC-2 — Incremental Verification  `CONDITIONAL`

**Statement:** Verifying k-task compositions requires O(k) interface checks, not O(k²) re-verification.

Each task is verified once. Composition checks only interface compatibility. Re-running the full safety kernel is not required.

---

## Heuristic Claims

### JSF-1, JSF-2 — Boundary Sensitivity  `HEURISTIC`

`GradientTracker` estimates which variables are near invariant boundaries via finite-difference perturbation. Diagnostic signal only; not safety enforcement. May miss nonlinear or cross-variable constraint interactions.

### AHJ-1, AHJ-2 — Active HJB Reachability  `HEURISTIC`

k-step lookahead to detect capture basins. Incomplete — bounded depth, finite action set. For complete reachability proofs, use TLA+ or SPIN.
