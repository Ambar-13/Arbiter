"""
ConstrAI v2 Hardening Tests
==========================
Tests the 5 vulnerability fixes:
  V1: Command Injection (sandboxed attestors)
  V2: Temporal Blindness (readiness probes)
  V3: Bayesian Cold-Start (cost-aware priors)
  V4: Spec-Reality Gap (environment reconciliation)
  V5: Reward Hacking (multi-dimensional attestation)
"""
import json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constrai import (
    State, Effect, ActionSpec, Invariant,
    BeliefState, CausalGraph, Belief,
)
from constrai.hardening import (
    SubprocessAttestor, PredicateAttestor, AttestationGate,
    AttestationResult, Attestation,
    ReadinessProbe, TemporalDependency, TemporalCausalGraph,
    CostAwarePriorFactory,
    EnvironmentProbe, EnvironmentReconciler, EnvironmentDriftError,
    ReconciliationResult,
    MultiDimensionalAttestor, QualityDimension,
    DependencyDiscovery,
    ResourceTracker, ResourceDescriptor, ResourceState, Permission,
)

class T:
    passed = 0; failed = 0; errors = []
    @classmethod
    def check(cls, name, cond, detail=""):
        if cond:
            cls.passed += 1; print(f"  ✓ {name}")
        else:
            cls.failed += 1; msg = f"  ✗ {name}: {detail}"
            cls.errors.append(msg); print(msg)
    @classmethod
    def summary(cls):
        total = cls.passed + cls.failed
        print(f"\n{'='*60}")
        print(f"  v2 HARDENING: {cls.passed}/{total} passed, {cls.failed} failed")
        if cls.errors:
            for e in cls.errors: print(f"    {e}")
        print(f"{'='*60}")
        return cls.failed == 0

# ═══════════════════════════════════════════════════════════════════════════
# V1: COMMAND INJECTION HARDENING
# ═══════════════════════════════════════════════════════════════════════════

print("\n🔒 V1: Command Injection Hardening")
print("-" * 40)

# Allowlist enforcement
try:
    SubprocessAttestor("evil", ["rm", "-rf", "/"])
    T.check("V1.1 rm blocked by allowlist", False)
except ValueError:
    T.check("V1.1 rm blocked by allowlist", True)

try:
    SubprocessAttestor("evil", ["bash", "-c", "echo pwned"])
    T.check("V1.2 bash blocked by allowlist", False)
except ValueError:
    T.check("V1.2 bash blocked by allowlist", True)

try:
    SubprocessAttestor("evil", ["sh", "-c", "whoami"])
    T.check("V1.3 sh blocked by allowlist", False)
except ValueError:
    T.check("V1.3 sh blocked by allowlist", True)

# Valid command works
try:
    a = SubprocessAttestor("ls_check", ["ls", "/tmp"])
    T.check("V1.4 ls in allowlist", True)
except ValueError:
    T.check("V1.4 ls in allowlist", False)

# Command is frozen (tuple)
a = SubprocessAttestor("test", ["ls", "-la"])
T.check("V1.5 command is tuple", isinstance(a._command, tuple))

# Empty command rejected
try:
    SubprocessAttestor("empty", [])
    T.check("V1.6 empty command rejected", False)
except ValueError:
    T.check("V1.6 empty command rejected", True)

# Path traversal in binary name
try:
    SubprocessAttestor("traverse", ["../../../bin/sh"])
    T.check("V1.7 path traversal blocked", False)
except ValueError:
    T.check("V1.7 path traversal blocked", True)

# Injection via filename args doesn't matter because shell=False
# The args are passed as literal strings, not shell-expanded
injected = SubprocessAttestor("safe", ["ls", "; rm -rf /"])
result = injected.verify(State({}), "test", timeout_s=5.0)
T.check("V1.8 shell metachar in args harmless",
        result.result in (AttestationResult.FAILED, AttestationResult.ERROR))


# ═══════════════════════════════════════════════════════════════════════════
# V2: TEMPORAL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════

print("\n🕐 V2: Temporal Dependencies")
print("-" * 40)

# Basic readiness probe
ready_after_3 = [0]  # Mutable counter for closure
def check_ready():
    ready_after_3[0] += 1
    if ready_after_3[0] >= 3:
        return True, "Ready!"
    return False, f"Attempt {ready_after_3[0]}, not ready"

probe = ReadinessProbe("db_ready", check_ready, interval_s=0.01, max_retries=5)
ok, detail, wait = probe.wait_until_ready()
T.check("V2.1 probe succeeds after retries", ok)
T.check("V2.2 took 3+ attempts", ready_after_3[0] >= 3)

# Probe that never becomes ready
def never_ready():
    return False, "Nope"

probe2 = ReadinessProbe("never", never_ready, interval_s=0.01, max_retries=3)
ok2, _, _ = probe2.wait_until_ready()
T.check("V2.3 never-ready probe fails", not ok2)

# Temporal dependency with min delay
tdep = TemporalDependency("build", "need build", min_delay_s=5.0)
tdep.mark_completed(100.0)
ok, reason = tdep.is_satisfied(102.0)
T.check("V2.4 too early (2s < 5s)", not ok)
ok, reason = tdep.is_satisfied(106.0)
T.check("V2.5 enough time (6s > 5s)", ok)

# TemporalCausalGraph
base = CausalGraph()
base.add_action("build", [])
base.add_action("deploy", [("build", "need build")])
tcg = TemporalCausalGraph(base)

db_ready_flag = [False]
def db_probe():
    return db_ready_flag[0], "ready" if db_ready_flag[0] else "warming"

tcg.add_temporal_dep("deploy", TemporalDependency(
    "build", "DB warmup",
    readiness_probe=ReadinessProbe("db", db_probe, max_retries=1),
    min_delay_s=2.0))

tcg.mark_completed("build", at_time=10.0)
tcg.set_time(11.0)
ok, unmet = tcg.can_execute("deploy")
T.check("V2.6 deploy blocked (too early + not ready)", not ok)

tcg.set_time(13.0)
ok, unmet = tcg.can_execute("deploy")
T.check("V2.7 deploy blocked (time ok but not ready)", not ok)

db_ready_flag[0] = True
ok, unmet = tcg.can_execute("deploy")
T.check("V2.8 deploy allowed (time ok + ready)", ok)


# ═══════════════════════════════════════════════════════════════════════════
# V3: COST-AWARE BAYESIAN PRIORS
# ═══════════════════════════════════════════════════════════════════════════

print("\n💰 V3: Cost-Aware Priors")
print("-" * 40)

factory = CostAwarePriorFactory(total_budget=100.0, pessimism_factor=5.0)

cheap = ActionSpec(id="cheap", name="Cheap", description="",
    effects=(), cost=1.0, risk_level="low")
medium = ActionSpec(id="med", name="Medium", description="",
    effects=(), cost=15.0, risk_level="medium")
expensive = ActionSpec(id="exp", name="Expensive", description="",
    effects=(), cost=35.0, risk_level="high", reversible=False)

a_c, b_c, tier_c = factory.compute_prior(cheap)
a_m, b_m, tier_m = factory.compute_prior(medium)
a_e, b_e, tier_e = factory.compute_prior(expensive)

T.check("V3.1 cheap = EXPLORE", tier_c == "EXPLORE")
T.check("V3.2 medium = CAUTIOUS or GUARDED", tier_m in ("CAUTIOUS", "GUARDED"))
T.check("V3.3 expensive = GATED", tier_e == "GATED")

# GATED action has near-zero prior
mean_gated = a_e / (a_e + b_e)
T.check(f"V3.4 GATED mean ≈ 0 ({mean_gated:.4f})", mean_gated < 0.01)

# After authorization, becomes reasonable
factory.authorize("exp")
a_auth, b_auth, tier_auth = factory.compute_prior(expensive)
mean_auth = a_auth / (a_auth + b_auth)
T.check(f"V3.5 authorized mean > 0.1 ({mean_auth:.3f})", mean_auth > 0.1)

# Cheap has optimistic prior
mean_cheap = a_c / (a_c + b_c)
T.check(f"V3.6 cheap mean > 0.7 ({mean_cheap:.3f})", mean_cheap > 0.7)

# Initialize beliefs
beliefs = BeliefState()
tiers = factory.initialize_beliefs(beliefs, [cheap, medium, expensive])
T.check("V3.7 all tiers assigned", len(tiers) == 3)

# Verify cheap is more likely to be selected than expensive
b_cheap = beliefs.get("action:cheap:succeeds")
b_exp = beliefs.get("action:exp:succeeds")
T.check("V3.8 cheap prior > expensive prior",
        b_cheap.mean > b_exp.mean)


# ═══════════════════════════════════════════════════════════════════════════
# V4: ENVIRONMENT RECONCILIATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n🔍 V4: Environment Reconciliation")
print("-" * 40)

# Simulate an environment
env_state = {"file_count": 3, "db_status": "ready"}

reconciler = EnvironmentReconciler(drift_threshold=0.0, halt_on_drift=True)
reconciler.add_probe(EnvironmentProbe(
    "file_count", lambda: env_state["file_count"]))
reconciler.add_probe(EnvironmentProbe(
    "db_status", lambda: env_state["db_status"]))

# Consistent state
model_state = State({"file_count": 3, "db_status": "ready"})
result = reconciler.reconcile(model_state, "post-action-1")
T.check("V4.1 consistent state detected", result.is_consistent)
T.check("V4.2 zero drift", result.drift_severity == 0.0)

# DRIFT: model says 5 files but env has 3
drifted_model = State({"file_count": 5, "db_status": "ready"})
try:
    reconciler.reconcile(drifted_model, "post-action-2")
    T.check("V4.3 drift detected and halted", False)
except EnvironmentDriftError as e:
    T.check("V4.3 drift detected and halted", True)
    T.check("V4.4 drift has mismatch details",
            len(e.result.mismatches) > 0)

# Non-halting mode
reconciler2 = EnvironmentReconciler(drift_threshold=0.5, halt_on_drift=False)
reconciler2.add_probe(EnvironmentProbe("x", lambda: 10))
r = reconciler2.reconcile(State({"x": 99}), "test")
T.check("V4.5 non-halt mode detects mismatch", not r.is_consistent)
T.check("V4.6 drift severity > 0", r.drift_severity > 0)

# Probe error handling
reconciler3 = EnvironmentReconciler(halt_on_drift=False)
reconciler3.add_probe(EnvironmentProbe("broken", lambda: 1/0))
r3 = reconciler3.reconcile(State({"broken": "ok"}), "error_test")
T.check("V4.7 probe error = mismatch", not r3.is_consistent)


# ═══════════════════════════════════════════════════════════════════════════
# V5: MULTI-DIMENSIONAL ATTESTATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n📊 V5: Multi-Dimensional Attestation")
print("-" * 40)

# Agent creates 5 empty files — should FAIL quality check
md = MultiDimensionalAttestor("code_quality", threshold=0.7)
md.add_check(QualityDimension.EXISTENCE, 1.0,
    lambda: (1.0, "5 files exist"))  # PASS
md.add_check(QualityDimension.COMPLETENESS, 1.0,
    lambda: (1.0, "All required files present"))  # PASS
md.add_check(QualityDimension.QUALITY, 1.0,
    lambda: (0.0, "All files are EMPTY (0 bytes)"))  # FAIL — zero score
md.add_check(QualityDimension.CORRECTNESS, 1.0,
    lambda: (0.8, "Syntax valid"))  # PASS

result = md.verify(State({}), "Create working code")
T.check("V5.1 empty files caught by quality dim",
        result.result == AttestationResult.FAILED)

# Legit output passes all dimensions
md2 = MultiDimensionalAttestor("legit_check", threshold=0.7)
md2.add_check(QualityDimension.EXISTENCE, 1.0, lambda: (1.0, "exists"))
md2.add_check(QualityDimension.QUALITY, 1.0, lambda: (0.85, "good"))
md2.add_check(QualityDimension.CORRECTNESS, 1.0, lambda: (0.9, "works"))
r2 = md2.verify(State({}), "test")
T.check("V5.2 legit output passes", r2.result == AttestationResult.VERIFIED)

# Single dimension at zero blocks everything
md3 = MultiDimensionalAttestor("block_test", threshold=0.5)
md3.add_check(QualityDimension.SAFETY, 1.0, lambda: (0.0, "UNSAFE"))
md3.add_check(QualityDimension.QUALITY, 1.0, lambda: (1.0, "great"))
r3 = md3.verify(State({}), "test")
T.check("V5.3 zero safety blocks even with high quality",
        r3.result == AttestationResult.FAILED)

# Exception in check = zero score
md4 = MultiDimensionalAttestor("error_test", threshold=0.5)
md4.add_check(QualityDimension.CORRECTNESS, 1.0, lambda: (1/0, ""))
md4.add_check(QualityDimension.EXISTENCE, 1.0, lambda: (1.0, "ok"))
r4 = md4.verify(State({}), "test")
T.check("V5.4 exception = failed attestation",
        r4.result == AttestationResult.FAILED)


# ═══════════════════════════════════════════════════════════════════════════
# V6: INTEGRATION — Cost-Aware + Temporal + Reconciliation together
# ═══════════════════════════════════════════════════════════════════════════

print("\n🔗 V6: Integration")
print("-" * 40)

import random

# Simulate a DevOps scenario with all v2 features
budget = 100.0
factory = CostAwarePriorFactory(budget, pessimism_factor=5.0)
beliefs = BeliefState()

actions = [
    ActionSpec(id="vpc", name="VPC", description="", effects=(), cost=5.0),
    ActionSpec(id="db", name="DB", description="", effects=(), cost=20.0, risk_level="medium"),
    ActionSpec(id="deploy", name="Deploy", description="", effects=(),
               cost=40.0, risk_level="high", reversible=False),
]

tiers = factory.initialize_beliefs(beliefs, actions)
T.check("V6.1 VPC = EXPLORE or CAUTIOUS", tiers["vpc"] in ("EXPLORE", "CAUTIOUS"))
T.check("V6.2 DB = CAUTIOUS or GUARDED", tiers["db"] in ("CAUTIOUS", "GUARDED"))
T.check("V6.3 Deploy = GATED", tiers["deploy"] == "GATED")

# Deploy GATED — can't select without authorization
deploy_belief = beliefs.get("action:deploy:succeeds")
T.check(f"V6.4 Deploy prior ≈ 0 ({deploy_belief.mean:.4f})",
        deploy_belief.mean < 0.01)

# After authorization
factory.authorize("deploy")
factory.initialize_beliefs(beliefs, actions)
deploy_belief2 = beliefs.get("action:deploy:succeeds")
T.check(f"V6.5 Authorized deploy prior > 0.1 ({deploy_belief2.mean:.3f})",
        deploy_belief2.mean > 0.1)

# VPC prior is optimistic
vpc_belief = beliefs.get("action:vpc:succeeds")
T.check(f"V6.6 VPC prior > DB prior ({vpc_belief.mean:.3f})",
        vpc_belief.mean > beliefs.get("action:db:succeeds").mean)


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

all_ok = T.summary()
sys.exit(0 if all_ok else 1)
