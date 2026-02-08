"""
ConstrAI — Layer 2: The Orchestrator
===================================

This is the MAIN ENGINE. It ties together:
  - Formal layer (safety kernel, budget, invariants, trace)
  - Reasoning layer (beliefs, causal graph, action values, LLM)
  
Into a complete autonomous execution loop:

  1. ANALYZE:  Compute action values, check dependencies
  2. REASON:   Ask LLM to select action (with full context)
  3. VERIFY:   Safety kernel checks T1-T7
  4. EXECUTE:  Commit state transition
  5. OBSERVE:  Update beliefs from outcome
  6. REPEAT:   Until goal reached, budget exhausted, or LLM says stop

The orchestrator also handles:
  - Fallback strategies when LLM fails
  - Automatic rollback on critical failures
  - Progress monitoring and stuck detection
  - Clean termination with summary
"""
from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple
)
from enum import Enum, auto

from .formal import (
    State, ActionSpec, Effect, SafetyKernel, SafetyVerdict,
    Invariant, GuaranteeLevel, Claim, CheckResult, BudgetController,
    ExecutionTrace, TraceEntry,
)
from .reasoning import (
    BeliefState, CausalGraph, ActionValueComputer, ActionValue,
    ReasoningRequest, ReasoningResponse, parse_llm_response,
    LLMAdapter, MockLLMAdapter, Belief,
)


# ═══════════════════════════════════════════════════════════════════════════
# §1  OUTCOME OBSERVATION
# ═══════════════════════════════════════════════════════════════════════════

class OutcomeType(Enum):
    SUCCESS = auto()
    PARTIAL = auto()
    FAILURE = auto()
    UNEXPECTED = auto()


@dataclass(frozen=True)
class Outcome:
    """Observed outcome of an action execution."""
    action_id: str
    outcome_type: OutcomeType
    actual_state: State
    expected_state: State
    details: str = ""

    @property
    def succeeded(self) -> bool:
        return self.outcome_type == OutcomeType.SUCCESS

    @property
    def state_matches_expected(self) -> bool:
        return self.actual_state == self.expected_state


# ═══════════════════════════════════════════════════════════════════════════
# §2  PROGRESS MONITOR — Stuck Detection
# ═══════════════════════════════════════════════════════════════════════════

class ProgressMonitor:
    """
    Tracks goal progress and detects when system is stuck.
    
    Stuck detection (EMPIRICAL):
      If progress hasn't improved in `patience` steps AND
      budget utilization > 50%, flag as stuck.
      
    This is a HEURISTIC — it can have false positives.
    The LLM gets the stuck flag as input and decides what to do.
    The formal layer doesn't care about stuck — it only cares about
    budget/invariants/termination.
    """
    def __init__(self, patience: int = 5):
        self.patience = patience
        self._history: List[Tuple[int, float]] = []  # (step, progress)

    def record(self, step: int, progress: float) -> None:
        self._history.append((step, progress))

    @property
    def current_progress(self) -> float:
        return self._history[-1][1] if self._history else 0.0

    @property
    def is_stuck(self) -> bool:
        if len(self._history) < self.patience:
            return False
        recent = [p for _, p in self._history[-self.patience:]]
        return max(recent) - min(recent) < 0.01  # No meaningful change

    @property
    def progress_rate(self) -> float:
        """Progress per step (moving average)."""
        if len(self._history) < 2:
            return 0.0
        recent = self._history[-min(5, len(self._history)):]
        dp = recent[-1][1] - recent[0][1]
        ds = recent[-1][0] - recent[0][0]
        return dp / max(ds, 1)

    def estimated_steps_to_goal(self) -> Optional[int]:
        """Estimate steps remaining to reach 100% progress."""
        rate = self.progress_rate
        if rate <= 0:
            return None
        remaining = 1.0 - self.current_progress
        return max(1, int(remaining / rate))

    def to_llm_text(self) -> str:
        lines = [f"Progress: {self.current_progress:.1%}"]
        if self.is_stuck:
            lines.append("⚠ STUCK: No progress in last {self.patience} steps")
        rate = self.progress_rate
        if rate > 0:
            est = self.estimated_steps_to_goal()
            lines.append(f"Rate: {rate:.3f}/step, ~{est} steps to completion")
        return " | ".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# §3  TASK DEFINITION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TaskDefinition:
    """
    Everything needed to define an autonomous task.
    
    This is the PUBLIC API for using ConstrAI. You define:
      - goal: what to achieve (natural language)
      - initial_state: starting world state
      - available_actions: what the agent can do
      - invariants: what must ALWAYS be true
      - budget: maximum resource spend
      - goal_predicate: formal success criterion
      - dependencies: which actions require which others
      - priors: initial beliefs about action success rates
    """
    goal: str
    initial_state: State
    available_actions: List[ActionSpec]
    invariants: List[Invariant]
    budget: float
    goal_predicate: Callable[[State], bool]
    goal_progress_fn: Optional[Callable[[State], float]] = None
    min_action_cost: float = 0.001
    dependencies: Optional[Dict[str, List[Tuple[str, str]]]] = None
    priors: Optional[Dict[str, Tuple[float, float]]] = None  # key → (α, β)
    max_retries_per_action: int = 3
    max_consecutive_failures: int = 5
    stuck_patience: int = 5
    system_prompt: str = ""
    risk_aversion: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# §4  EXECUTION RESULT
# ═══════════════════════════════════════════════════════════════════════════

class TerminationReason(Enum):
    GOAL_ACHIEVED = "goal_achieved"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STEP_LIMIT = "step_limit"
    LLM_STOP = "llm_requested_stop"
    STUCK = "stuck_detected"
    MAX_FAILURES = "max_consecutive_failures"
    ERROR = "unrecoverable_error"


@dataclass
class ExecutionResult:
    """Complete result of an ConstrAI execution run."""
    goal_achieved: bool
    termination_reason: TerminationReason
    final_state: State
    total_cost: float
    total_steps: int
    goal_progress: float
    execution_time_s: float
    trace_length: int
    beliefs_summary: str
    budget_summary: str
    errors: List[str] = field(default_factory=list)
    
    # Detailed metrics
    actions_attempted: int = 0
    actions_succeeded: int = 0
    actions_rejected_safety: int = 0
    actions_rejected_reasoning: int = 0
    rollbacks: int = 0
    llm_calls: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'goal_achieved': self.goal_achieved,
            'termination_reason': self.termination_reason.value,
            'total_cost': self.total_cost,
            'total_steps': self.total_steps,
            'goal_progress': self.goal_progress,
            'execution_time_s': self.execution_time_s,
            'actions_attempted': self.actions_attempted,
            'actions_succeeded': self.actions_succeeded,
            'actions_rejected_safety': self.actions_rejected_safety,
            'actions_rejected_reasoning': self.actions_rejected_reasoning,
            'rollbacks': self.rollbacks,
            'llm_calls': self.llm_calls,
            'errors': self.errors,
        }

    def summary(self) -> str:
        status = "✅ GOAL ACHIEVED" if self.goal_achieved else "❌ GOAL NOT ACHIEVED"
        return (
            f"\n{'='*60}\n"
            f"  ConstrAI Execution Summary\n"
            f"{'='*60}\n"
            f"  {status}\n"
            f"  Reason: {self.termination_reason.value}\n"
            f"  Progress: {self.goal_progress:.1%}\n"
            f"  Cost: ${self.total_cost:.2f}\n"
            f"  Steps: {self.total_steps}\n"
            f"  Time: {self.execution_time_s:.2f}s\n"
            f"  Actions: {self.actions_succeeded}/{self.actions_attempted} succeeded\n"
            f"  Safety rejections: {self.actions_rejected_safety}\n"
            f"  LLM calls: {self.llm_calls}\n"
            f"  Rollbacks: {self.rollbacks}\n"
            f"  Errors: {len(self.errors)}\n"
            f"{'='*60}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# §5  THE ORCHESTRATOR — Main Engine
# ═══════════════════════════════════════════════════════════════════════════

class Orchestrator:
    """
    The main ConstrAI execution engine.
    
    Architecture:
      ┌────────────────────────────────────────────┐
      │            Orchestrator (this)              │
      │  ┌──────────────────────────────────────┐  │
      │  │   LLM Adapter (pluggable)            │  │
      │  │   ↕ structured prompts/responses      │  │
      │  ├──────────────────────────────────────┤  │
      │  │   Reasoning Engine                    │  │
      │  │   • Belief State (Bayesian)           │  │
      │  │   • Causal Graph (DAG)                │  │
      │  │   • Action Value (info-theoretic)     │  │
      │  │   ↕ computed analysis                 │  │
      │  ├──────────────────────────────────────┤  │
      │  │   Safety Kernel (FORMAL)              │  │
      │  │   • Budget (T1)                       │  │
      │  │   • Termination (T2)                  │  │
      │  │   • Invariants (T3)                   │  │
      │  │   • Atomicity (T5)                    │  │
      │  │   • Trace (T6)                        │  │
      │  └──────────────────────────────────────┘  │
      └────────────────────────────────────────────┘
    
    The formal layer CANNOT be bypassed by the LLM.
    The LLM CANNOT be bypassed by the formal layer.
    Both must agree for any action to execute.
    """

    def __init__(self, task: TaskDefinition, llm: Optional[LLMAdapter] = None):
        self.task = task
        self.llm = llm or MockLLMAdapter()

        # ── Formal Layer ──
        self.kernel = SafetyKernel(
            budget=task.budget,
            invariants=task.invariants,
            min_action_cost=task.min_action_cost,
        )

        # ── Reasoning Layer ──
        self.beliefs = BeliefState()
        self.causal_graph = CausalGraph()
        self.value_computer = ActionValueComputer(
            risk_aversion=task.risk_aversion)
        self.progress_monitor = ProgressMonitor(
            patience=task.stuck_patience)

        # ── State ──
        self.current_state = task.initial_state
        self.action_map: Dict[str, ActionSpec] = {
            a.id: a for a in task.available_actions}
        self._state_history: List[State] = [task.initial_state]

        # ── Initialize causal graph ──
        if task.dependencies:
            for action_id, deps in task.dependencies.items():
                self.causal_graph.add_action(action_id, deps)
        # Also add actions without explicit deps
        for a in task.available_actions:
            if a.id not in (task.dependencies or {}):
                self.causal_graph.add_action(a.id)

        # ── Initialize priors ──
        if task.priors:
            for key, (alpha, beta) in task.priors.items():
                self.beliefs.set_prior(key, alpha, beta)

        # ── Metrics ──
        self._actions_attempted = 0
        self._actions_succeeded = 0
        self._actions_rejected_safety = 0
        self._actions_rejected_reasoning = 0
        self._rollbacks = 0
        self._llm_calls = 0
        self._consecutive_failures = 0
        self._errors: List[str] = []

    def _compute_progress(self) -> float:
        if self.task.goal_progress_fn:
            return self.task.goal_progress_fn(self.current_state)
        if self.task.goal_predicate(self.current_state):
            return 1.0
        return 0.0

    def _get_available_actions(self) -> List[ActionSpec]:
        """Filter actions to those affordable AND not already completed."""
        available = []
        for action in self.task.available_actions:
            can_afford, _ = self.kernel.budget.can_afford(action.cost)
            if not can_afford:
                continue
            # Skip actions already completed (in causal graph)
            # Unless they're designed to be repeatable (no deps tracking)
            if action.id in self.causal_graph._completed:
                # Check if this action's effects would still change state
                sim = action.simulate(self.current_state)
                if sim == self.current_state:
                    continue  # No-op, skip it
            available.append(action)
        return available

    def _compute_action_values(self, actions: List[ActionSpec]) -> List[ActionValue]:
        progress = self._compute_progress()
        steps_left = self.kernel.max_steps - self.kernel.step_count
        return [
            self.value_computer.compute(
                action=a, state=self.current_state,
                beliefs=self.beliefs,
                budget_remaining=self.kernel.budget.remaining,
                goal_progress=progress,
                steps_remaining=steps_left,
            )
            for a in actions
        ]

    def _build_history_summary(self, last_n: int = 5) -> str:
        entries = self.kernel.trace.last_n(last_n)
        if not entries:
            return "(no actions taken yet)"
        lines = []
        for e in entries:
            icon = "✓" if e.approved else "✗"
            lines.append(f"  [{icon}] {e.action_name} (${e.cost:.2f})")
            if e.reasoning_summary:
                lines.append(f"      → {e.reasoning_summary[:100]}")
        return "\n".join(lines)

    def _ask_llm(self, available: List[ActionSpec],
                 values: List[ActionValue]) -> ReasoningResponse:
        """Build structured prompt, query LLM, parse response."""
        request = ReasoningRequest(
            goal=self.task.goal,
            state=self.current_state,
            available_actions=available,
            action_values=values,
            beliefs=self.beliefs,
            causal_graph=self.causal_graph,
            safety_kernel=self.kernel,
            history_summary=self._build_history_summary(),
        )
        prompt = request.to_prompt()
        
        t0 = _time.time()
        raw = self.llm.complete(
            prompt=prompt,
            system_prompt=self.task.system_prompt or ConstrAI_SYSTEM_PROMPT,
            temperature=0.3,
        )
        latency = (_time.time() - t0) * 1000
        self._llm_calls += 1

        valid_ids = set(a.id for a in available)
        response = parse_llm_response(raw, valid_ids)
        response.latency_ms = latency
        return response

    def _execute_action(self, action: ActionSpec,
                        reasoning: str) -> Tuple[bool, str]:
        """
        Try to execute an action through the safety kernel.
        Returns (success, message).
        """
        self._actions_attempted += 1

        # ── Safety check (formal) ──
        verdict = self.kernel.evaluate(self.current_state, action)
        if not verdict.approved:
            self._actions_rejected_safety += 1
            self.kernel.record_rejection(
                self.current_state, action,
                verdict.rejection_reasons, reasoning)
            self.beliefs.observe(f"action:{action.id}:succeeds", False)
            return False, f"Safety rejected: {verdict.rejection_reasons}"

        # ── Execute (atomic — T5) ──
        try:
            new_state, trace_entry = self.kernel.execute(
                self.current_state, action, reasoning)
        except Exception as e:
            self._errors.append(f"Execute error: {e}")
            self.beliefs.observe(f"action:{action.id}:succeeds", False)
            return False, f"Execution error: {e}"

        # ── Observe outcome ──
        self.current_state = new_state
        self._state_history.append(new_state)
        self._actions_succeeded += 1
        self._consecutive_failures = 0

        # Update beliefs
        self.beliefs.observe(f"action:{action.id}:succeeds", True)
        self.causal_graph.mark_completed(action.id)

        # Update progress
        progress = self._compute_progress()
        self.progress_monitor.record(self.kernel.step_count, progress)
        # Store progress in state for visibility
        self.current_state = self.current_state.with_updates(
            {'_progress': progress})

        return True, f"Executed {action.name}, progress={progress:.1%}"

    def run(self) -> ExecutionResult:
        """
        Main execution loop. Returns when done.
        
        Loop invariants (maintained every iteration):
          - self.kernel.budget.spent ≤ self.kernel.budget.budget  (T1)
          - self.kernel.step_count ≤ self.kernel.max_steps        (T2)
          - All invariants hold on self.current_state              (T3)
        """
        t_start = _time.time()

        # Verify initial invariants
        for inv in self.task.invariants:
            ok, msg = inv.check(self.current_state)
            if not ok:
                return self._make_result(
                    TerminationReason.ERROR, t_start,
                    errors=[f"Initial state violates invariant: {msg}"])

        while True:
            # ── Check goal ──
            if self.task.goal_predicate(self.current_state):
                return self._make_result(
                    TerminationReason.GOAL_ACHIEVED, t_start)

            # ── Check termination conditions ──
            if self.kernel.step_count >= self.kernel.max_steps:
                return self._make_result(
                    TerminationReason.STEP_LIMIT, t_start)

            if self._consecutive_failures >= self.task.max_consecutive_failures:
                return self._make_result(
                    TerminationReason.MAX_FAILURES, t_start)

            # ── Get available actions ──
            available = self._get_available_actions()
            if not available:
                return self._make_result(
                    TerminationReason.BUDGET_EXHAUSTED, t_start)

            # ── Compute values (formal analysis) ──
            values = self._compute_action_values(available)

            # ── Ask LLM to reason ──
            try:
                response = self._ask_llm(available, values)
            except Exception as e:
                self._errors.append(f"LLM error: {e}")
                # Fallback: use highest-value action
                response = self._fallback_selection(available, values)

            # ── Handle LLM stop request ──
            if response.should_stop:
                return self._make_result(
                    TerminationReason.LLM_STOP, t_start)

            # ── Validate LLM choice ──
            if not response.is_valid:
                self._errors.append(f"Invalid LLM response: {response.parse_errors}")
                self._consecutive_failures += 1
                response = self._fallback_selection(available, values)
                if response.should_stop:
                    return self._make_result(
                        TerminationReason.ERROR, t_start,
                        errors=self._errors)

            # ── Execute chosen action ──
            action = self.action_map.get(response.chosen_action_id)
            if action is None:
                self._errors.append(f"Action {response.chosen_action_id} not found")
                self._consecutive_failures += 1
                continue

            success, msg = self._execute_action(action, response.reasoning)
            if not success:
                self._consecutive_failures += 1

            # ── Check stuck ──
            if self.progress_monitor.is_stuck:
                # Don't terminate immediately — let LLM know it's stuck
                # via the progress monitor text in next iteration
                if self._consecutive_failures > self.task.stuck_patience:
                    return self._make_result(
                        TerminationReason.STUCK, t_start)

    def _fallback_selection(self, available: List[ActionSpec],
                            values: List[ActionValue]) -> ReasoningResponse:
        """When LLM fails, select highest-value READY action."""
        ready_ids = set(self.causal_graph.ready_actions(
            [a.id for a in available]))
        
        ready_values = [v for v in values if v.action_id in ready_ids]
        if not ready_values:
            return ReasoningResponse(
                chosen_action_id="", reasoning="No ready actions",
                expected_outcome="", risk_assessment="",
                alternative_considered="", should_stop=True,
                stop_reason="No ready actions available")
        
        best = max(ready_values, key=lambda v: v.value_score)
        return ReasoningResponse(
            chosen_action_id=best.action_id,
            reasoning=f"Fallback: selected highest-value action {best.action_id}",
            expected_outcome="", risk_assessment="",
            alternative_considered="", should_stop=False,
            stop_reason="")

    def _make_result(self, reason: TerminationReason,
                     t_start: float,
                     errors: List[str] = None) -> ExecutionResult:
        progress = self._compute_progress()
        return ExecutionResult(
            goal_achieved=(reason == TerminationReason.GOAL_ACHIEVED),
            termination_reason=reason,
            final_state=self.current_state,
            total_cost=self.kernel.budget.spent,
            total_steps=self.kernel.step_count,
            goal_progress=progress,
            execution_time_s=_time.time() - t_start,
            trace_length=self.kernel.trace.length,
            beliefs_summary=self.beliefs.summary(),
            budget_summary=self.kernel.budget.summary(),
            errors=errors or self._errors,
            actions_attempted=self._actions_attempted,
            actions_succeeded=self._actions_succeeded,
            actions_rejected_safety=self._actions_rejected_safety,
            actions_rejected_reasoning=self._actions_rejected_reasoning,
            rollbacks=self._rollbacks,
            llm_calls=self._llm_calls,
        )


# ═══════════════════════════════════════════════════════════════════════════
# §6  DEFAULT SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

ConstrAI_SYSTEM_PROMPT = """You are an AI agent operating within the ConstrAI safety framework.

YOUR ROLE:
You select actions to achieve a goal. You are given formal analysis
(action values, beliefs, dependencies, budget) and must make a rational
decision.

CONSTRAINTS YOU CANNOT OVERRIDE:
- Budget limits are enforced by the safety kernel
- Invariants are enforced by the safety kernel
- Step limits are enforced by the safety kernel
- You can only choose from the READY actions listed

YOUR DECISION PROCESS:
1. Read the goal and current progress
2. Review the action value analysis (it's already computed for you)
3. Consider dependencies — don't pick BLOCKED actions
4. Consider beliefs — prefer actions with high success probability
5. Consider budget — don't waste money on low-value actions
6. Select the best action and explain your reasoning

RESPOND WITH VALID JSON ONLY.
"""
