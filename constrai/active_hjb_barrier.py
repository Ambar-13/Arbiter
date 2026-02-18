"""
Active HJB Safety Barrier (Heuristic)
======================================

Hamilton-Jacobi-Bellman (HJB) reachability analysis wired directly into the
decision loop. Prevents the agent from entering forbidden state regions
(capture basins) by checking k-step forward reachability before each action.

GUARANTEE LEVEL: HEURISTIC
  Not complete reachability analysis. Explores a bounded lookahead tree, not
  the full state space. For formally-proven invariant preservation use the
  kernel's T3 (formal.py), which blocks any action violating a blocking-mode
  invariant on the very next step.

Why include this alongside T3?
  T3 is one-step look-ahead. HJB look-ahead catches multi-step traps:
  sequences of individually-safe actions that funnel the agent into a
  no-exit region. HJB + T3 together catch far more real-world failure modes.

Limitations:
  1. Exponential complexity: O(|actions|^k). Recommended k ≤ 3.
  2. Incomplete: only explores the given action set, not all transitions.
  3. Per-call memoization only (see _is_basin_reachable).
  4. Assumes actions are deterministic (no stochastic outcomes).

For formal reachability proofs, use an external model checker (TLA+, SPIN).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from .formal import State, ActionSpec, GuaranteeLevel
from .reference_monitor import CaptureBasin


class SafetyBarrierViolation(Enum):
    """Type of HJB barrier violation."""
    ALREADY_IN_BASIN = "already_in_basin"        # s is already in bad region
    ACTION_DIRECTLY_ENTERS = "direct_entry"      # Action leads directly to basin
    REACHABLE_IN_STEPS = "reachable_in_steps"    # Basin reachable within k steps
    NO_VIOLATION = "safe"                         # No violation


@dataclass
class HJBBarrierCheck:
    """Result of an HJB safety barrier evaluation."""
    violation_type: SafetyBarrierViolation
    is_safe: bool  # = (violation_type == SafetyBarrierViolation.NO_VIOLATION)
    basin_name: Optional[str]
    distance_to_basin: Optional[int]  # Steps until capture basin (if computable)
    recommendation: str  # Human explanation + recovery suggestion


class ActiveHJBBarrier:
    """
    k-step lookahead capture-basin avoidance barrier. [HEURISTIC]

    For each proposed action, checks three conditions in order:
      1. Is the current state already in a capture basin?
      2. Does the proposed action immediately enter a basin?
      3. Is any basin reachable within max_lookahead steps from the
         post-action state, given the available action set?

    When a violation is found, the orchestrator must reject the action and
    either rollback or enter Safe Hover mode (see RecoveryStrategy).

    Per-call memoization:
      _is_basin_reachable() caches results by
      (state_fingerprint, basin_name, steps_remaining) within each call to
      check_and_enforce(). This cuts redundant simulation in trees where
      multiple branches converge on the same (state, depth) pair.
      Cache is scoped per check_and_enforce() call, not globally.
    """

    def __init__(self, basins: List[CaptureBasin], max_lookahead: int = 3):
        if max_lookahead > 5:
            print(f"[HJB] Warning: lookahead depth {max_lookahead} causes "
                  f"O(n^{max_lookahead}) expansion. Consider k ≤ 3.")
        self.basins = basins
        self.max_lookahead = max_lookahead
    
    def check_and_enforce(
        self,
        state: State,
        proposed_action: ActionSpec,
        available_actions: List[ActionSpec],
        current_step: int,
        max_steps: int,
    ) -> Tuple[bool, HJBBarrierCheck]:
        """
        Evaluate whether proposed_action is safe under the HJB barrier.

        Returns:
            (is_safe, check_result)

        If is_safe=False, the caller must reject the action and trigger
        recovery (Safe Hover or rollback). The LLM cannot override this.
        """
        # Fresh per-call memoization cache (key = fingerprint + basin + depth).
        cache: Dict[Tuple[str, str, int], bool] = {}

        for basin in self.basins:
            # Check 1: Already trapped.
            if basin.is_bad(state):
                return False, HJBBarrierCheck(
                    violation_type=SafetyBarrierViolation.ALREADY_IN_BASIN,
                    is_safe=False,
                    basin_name=basin.name,
                    distance_to_basin=0,
                    recommendation=(
                        f"CRITICAL: Current state is already inside capture basin "
                        f"'{basin.name}'. Must enter Safe Hover and attempt rollback."
                    ),
                )

            # Check 2: Action enters basin immediately.
            next_state = proposed_action.simulate(state)
            if basin.is_bad(next_state):
                return False, HJBBarrierCheck(
                    violation_type=SafetyBarrierViolation.ACTION_DIRECTLY_ENTERS,
                    is_safe=False,
                    basin_name=basin.name,
                    distance_to_basin=1,
                    recommendation=(
                        f"BARRIER: Action '{proposed_action.name}' would immediately "
                        f"enter capture basin '{basin.name}'. Action rejected."
                    ),
                )

            # Check 3: Basin reachable within lookahead from post-action state.
            if self._is_basin_reachable(
                next_state, available_actions, basin,
                steps_remaining=self.max_lookahead, cache=cache,
            ):
                steps_left = max_steps - current_step
                if steps_left <= self.max_lookahead:
                    return False, HJBBarrierCheck(
                        violation_type=SafetyBarrierViolation.REACHABLE_IN_STEPS,
                        is_safe=False,
                        basin_name=basin.name,
                        distance_to_basin=steps_left,
                        recommendation=(
                            f"WARNING: Capture basin '{basin.name}' reachable within "
                            f"{steps_left} remaining step(s). Entering Safe Hover."
                        ),
                    )

        return True, HJBBarrierCheck(
            violation_type=SafetyBarrierViolation.NO_VIOLATION,
            is_safe=True,
            basin_name=None,
            distance_to_basin=None,
            recommendation="HJB barrier clear",
        )
    
    def _is_basin_reachable(
        self,
        state: State,
        actions: List[ActionSpec],
        basin: CaptureBasin,
        steps_remaining: int,
        cache: Dict[Tuple[str, str, int], bool],
    ) -> bool:
        """
        Recursive DFS reachability check with per-call memoization.

        Cache key: (state_fingerprint, basin.name, steps_remaining).
        Avoids re-exploring sub-trees when multiple branches converge on
        the same (state, depth). Cache is provided by the caller and scoped
        to one top-level check_and_enforce() invocation.
        """
        if steps_remaining <= 0:
            return False

        key = (state.fingerprint, basin.name, steps_remaining)
        if key in cache:
            return cache[key]

        result = False
        for action in actions:
            next_state = action.simulate(state)
            if basin.is_bad(next_state):
                result = True
                break
            if self._is_basin_reachable(next_state, actions, basin,
                                         steps_remaining - 1, cache):
                result = True
                break

        cache[key] = result
        return result


# ═════════════════════════════════════════════════════════════════════════════
# Recovery Strategy Enum
# ═════════════════════════════════════════════════════════════════════════════

class RecoveryStrategy(Enum):
    """What the orchestrator should do when the HJB barrier fires."""
    SAFE_HOVER          = "safe_hover"          # Stay in place; do nothing
    ROLLBACK_ONE        = "rollback_one"        # Undo the last committed action
    ROLLBACK_TO_SAFE    = "rollback_to_safe"    # Undo until well clear of basin
    HUMAN_INTERVENTION  = "human_ask"           # Escalate to a human operator
    GRACEFUL_HALT       = "halt"                # Stop execution cleanly


def choose_recovery_strategy(
    barrier_check: HJBBarrierCheck,
    current_step: int,
    max_steps: int,
    is_reversible_available: bool,
) -> RecoveryStrategy:
    """
    Select the least-disruptive recovery for a barrier violation.

    Priority:
      1. Rollback one step if rollback history is available (cheapest recovery).
      2. Safe Hover if we are near the step/budget limit.
      3. Default: Safe Hover.
    """
    if is_reversible_available and current_step > 0:
        return RecoveryStrategy.ROLLBACK_ONE

    if current_step > max_steps * 0.8:
        return RecoveryStrategy.SAFE_HOVER

    return RecoveryStrategy.SAFE_HOVER
