"""
Arbiter Framework Examples

Practical examples demonstrating formal safety guarantees.
"""

from arbiter import (
    ArbiterAgent, ResourceBudget, Invariant, TerminationGuard,
    CausalGraph, BeliefTracker, SandboxedAttestor,
    EnvironmentReconciler, LLMInterface
)
import time


def example_1_basic_agent():
    """Example 1: Basic agent with all guarantees."""
    print("=" * 60)
    print("Example 1: Basic Agent with Formal Guarantees")
    print("=" * 60)
    
    # Create agent with resource bounds
    agent = ArbiterAgent(
        max_operations=50,
        max_time_seconds=30.0,
        max_memory_bytes=50_000_000
    )
    
    # Execute some reasoning
    for i in range(5):
        result = agent.execute_action("reason", {
            "prompt": f"Process request {i}",
            "context": {"request_id": i}
        })
        print(f"Action {i}: {result}")
    
    # Get statistics
    stats = agent.get_statistics()
    print(f"\nStatistics:")
    print(f"  Operations: {stats['operations_used']}/{stats['max_operations']}")
    print(f"  Time: {stats['execution_time']:.3f}s/{stats['max_time']}s")
    print(f"  Actions: {stats['actions_executed']}")
    print()


def example_2_budget_enforcement():
    """Example 2: Budget enforcement prevents runaway execution."""
    print("=" * 60)
    print("Example 2: Budget Enforcement")
    print("=" * 60)
    
    # Agent with very limited budget
    agent = ArbiterAgent(max_operations=10, max_time_seconds=5.0)
    
    # Try to execute many operations
    completed = 0
    try:
        for i in range(20):
            agent.execute_action("reason", {"prompt": f"Task {i}"})
            completed += 1
    except RuntimeError as e:
        print(f"Budget exhausted after {completed} operations")
        print(f"Error: {e}")
    
    print(f"Agent terminated safely within bounds")
    print()


def example_3_invariant_protection():
    """Example 3: Invariants protect system state."""
    print("=" * 60)
    print("Example 3: Invariant Protection")
    print("=" * 60)
    
    # System with invariants
    account = {"balance": 100, "transactions": 0}
    
    # Define invariants
    invariants = [
        Invariant(lambda: account["balance"] >= 0, "non_negative_balance"),
        Invariant(lambda: account["transactions"] >= 0, "non_negative_tx"),
    ]
    
    # Safe transaction
    print("Safe transaction: withdraw 50")
    account["balance"] -= 50
    account["transactions"] += 1
    for inv in invariants:
        inv.enforce()  # Passes
    print(f"  Balance: {account['balance']}")
    
    # Unsafe transaction would be caught
    print("\nUnsafe transaction: withdraw 100")
    account["balance"] -= 100  # Would go negative
    try:
        for inv in invariants:
            inv.enforce()
        print("  Transaction allowed")
    except Exception as e:
        # Rollback
        account["balance"] += 100
        print(f"  Transaction blocked: {e}")
        print(f"  Balance: {account['balance']} (rolled back)")
    
    print()


def example_4_guaranteed_termination():
    """Example 4: Guaranteed termination of loops."""
    print("=" * 60)
    print("Example 4: Guaranteed Termination")
    print("=" * 60)
    
    # Create termination guard
    budget = ResourceBudget(max_operations=100, max_time_seconds=10.0, max_memory_bytes=10**9)
    guard = TerminationGuard(budget)
    
    # Loop that would be infinite without guard
    iterations = 0
    print("Running potentially infinite loop with guard...")
    while guard.step():
        # Simulate work
        iterations += 1
        if iterations % 10 == 0:
            print(f"  Iteration {iterations}")
    
    print(f"Loop terminated safely after {iterations} iterations")
    print(f"Guard ensured termination within budget")
    print()


def example_5_causal_tracking():
    """Example 5: Causal dependency tracking."""
    print("=" * 60)
    print("Example 5: Causal Dependency Tracking")
    print("=" * 60)
    
    graph = CausalGraph()
    
    # Build a causal chain
    print("Building causal chain:")
    print("  sensor_reading → data_processing → decision → action")
    
    graph.add_node("sensor_reading", {"temp": 25})
    graph.add_node("data_processing", {"processed": True}, caused_by=["sensor_reading"])
    graph.add_node("decision", {"action": "cool"}, caused_by=["data_processing"])
    graph.add_node("action", {"executed": True}, caused_by=["decision"])
    
    # Query relationships
    print("\nCausal relationships:")
    print(f"  action depends on sensor_reading: {graph.is_causal_descendant('action', 'sensor_reading')}")
    
    ancestors = graph.get_ancestors("action")
    print(f"  All ancestors of action: {ancestors}")
    
    # Track effects
    print(f"\n  Effects of sensor_reading: {graph.nodes['sensor_reading'].effects}")
    print()


def example_6_bayesian_beliefs():
    """Example 6: Bayesian belief tracking."""
    print("=" * 60)
    print("Example 6: Bayesian Belief Tracking")
    print("=" * 60)
    
    tracker = BeliefTracker()
    
    # Initial belief
    tracker.add_belief("system_is_secure", prior=0.7)
    print(f"Initial belief (prior): {tracker.get_posterior('system_is_secure'):.3f}")
    
    # Update with positive evidence
    print("\nObserve: Security scan passed (evidence likelihood 0.9)")
    tracker.update_belief("system_is_secure", evidence_likelihood=0.9)
    print(f"Updated belief: {tracker.get_posterior('system_is_secure'):.3f}")
    
    # Update with more positive evidence
    print("\nObserve: Penetration test passed (evidence likelihood 0.95)")
    tracker.update_belief("system_is_secure", evidence_likelihood=0.95)
    print(f"Updated belief: {tracker.get_posterior('system_is_secure'):.3f}")
    
    # Update with negative evidence
    print("\nObserve: Minor vulnerability found (evidence likelihood 0.3)")
    tracker.update_belief("system_is_secure", evidence_likelihood=0.3)
    print(f"Updated belief: {tracker.get_posterior('system_is_secure'):.3f}")
    
    print()


def example_7_sandboxed_attestors():
    """Example 7: Sandboxed attestations."""
    print("=" * 60)
    print("Example 7: Sandboxed Attestors")
    print("=" * 60)
    
    # Create attestor with limited budget
    budget = ResourceBudget(5, 10.0, 1000000)
    attestor = SandboxedAttestor("security_verifier", budget)
    
    # Multiple attestations
    claims = [
        ("input_validated", lambda: True),
        ("output_safe", lambda: True),
        ("no_injection", lambda: True),
        ("rate_limit_ok", lambda: True),
        ("auth_verified", lambda: True),
    ]
    
    print("Running attestations:")
    for claim, verifier in claims:
        result = attestor.attest(claim, verifier)
        print(f"  {claim}: {'PASS' if result else 'FAIL'}")
    
    # Budget exhausted - next attestation fails
    print("\nAttempting attestation beyond budget:")
    try:
        attestor.attest("extra_check", lambda: True)
        print("  Attestation succeeded")
    except Exception as e:
        print(f"  Attestation blocked: {e}")
    
    # Review attestation history
    history = attestor.get_attestations()
    print(f"\nTotal attestations recorded: {len(history)}")
    print()


def example_8_environment_reconciliation():
    """Example 8: Environment state reconciliation."""
    print("=" * 60)
    print("Example 8: Environment Reconciliation")
    print("=" * 60)
    
    reconciler = EnvironmentReconciler()
    
    # Initial state
    print("Initial observation:")
    state1 = reconciler.reconcile({"temperature": 20, "humidity": 60})
    print(f"  {state1}")
    
    # Update with new observation
    print("\nNew observation (temperature changed):")
    state2 = reconciler.reconcile({"temperature": 22, "pressure": 1013})
    print(f"  {state2}")
    
    # Another update
    print("\nAnother observation (humidity changed):")
    state3 = reconciler.reconcile({"humidity": 65})
    print(f"  {state3}")
    
    # View history
    print("\nComplete state history:")
    for i, state in enumerate(reconciler.get_state_history()):
        print(f"  Version {state.version}: {state.state}")
    
    print()


def example_9_complete_workflow():
    """Example 9: Complete workflow with all theorems."""
    print("=" * 60)
    print("Example 9: Complete Workflow")
    print("=" * 60)
    
    # Create agent
    agent = ArbiterAgent(max_operations=100, max_time_seconds=30.0)
    
    # Step 1: Set up beliefs
    print("1. Initialize beliefs")
    agent.belief_tracker.add_belief("request_is_safe", prior=0.5)
    agent.belief_tracker.add_belief("user_is_authenticated", prior=0.8)
    
    # Step 2: Process request with reasoning
    print("2. Process request")
    result = agent.execute_action("reason", {
        "prompt": "Analyze incoming request",
        "context": {"user_id": "user123", "action": "read"}
    })
    print(f"   Result: {result}")
    
    # Step 3: Update beliefs based on analysis
    print("3. Update beliefs")
    agent.execute_action("update_belief", {
        "hypothesis": "request_is_safe",
        "likelihood": 0.9
    })
    print(f"   request_is_safe posterior: {agent.belief_tracker.get_posterior('request_is_safe'):.3f}")
    
    # Step 4: Attest to safety
    print("4. Attest to safety properties")
    def verify_request_safety():
        # Actual verification logic would go here
        return True
    
    agent.execute_action("attest", {
        "attestor": "safety_checker",
        "claim": "request_passed_safety_checks",
        "verifier": verify_request_safety
    })
    print("   Attestation: PASS")
    
    # Step 5: Reconcile environment
    print("5. Reconcile environment state")
    agent.execute_action("reconcile", {
        "observed_state": {
            "request_processed": True,
            "timestamp": time.time(),
            "user_id": "user123"
        }
    })
    
    # Step 6: Review statistics
    print("\n6. Final statistics:")
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\nWorkflow completed successfully with all guarantees enforced!")
    print()


def example_10_custom_llm():
    """Example 10: Custom LLM integration."""
    print("=" * 60)
    print("Example 10: Custom LLM Integration")
    print("=" * 60)
    
    # Define custom LLM
    class MyCustomLLM(LLMInterface):
        def __init__(self):
            self.call_count = 0
        
        def reason(self, prompt: str, context: dict) -> str:
            self.call_count += 1
            # Your actual LLM call would go here
            return f"Response to: {prompt[:30]}... (call #{self.call_count})"
        
        def constrained_reason(self, prompt: str, context: dict, 
                             constraints: dict) -> str:
            self.call_count += 1
            # Constrained reasoning with safety bounds
            max_tokens = constraints.get("max_tokens", 100)
            return f"Safe response (≤{max_tokens} tokens) (call #{self.call_count})"
    
    # Create agent with custom LLM
    agent = ArbiterAgent()
    custom_llm = MyCustomLLM()
    agent.set_llm(custom_llm)
    
    # Use the custom LLM
    print("Executing reasoning with custom LLM:")
    for i in range(3):
        result = agent.execute_action("reason", {
            "prompt": f"Generate response for query {i}",
            "context": {"query_id": i}
        })
        print(f"  {result}")
    
    print(f"\nCustom LLM was called {custom_llm.call_count} times")
    print(f"All calls were constrained by budget: {agent.get_statistics()['operations_used']} operations used")
    print()


def main():
    """Run all examples."""
    examples = [
        example_1_basic_agent,
        example_2_budget_enforcement,
        example_3_invariant_protection,
        example_4_guaranteed_termination,
        example_5_causal_tracking,
        example_6_bayesian_beliefs,
        example_7_sandboxed_attestors,
        example_8_environment_reconciliation,
        example_9_complete_workflow,
        example_10_custom_llm,
    ]
    
    print("\n" + "=" * 60)
    print("ARBITER FRAMEWORK EXAMPLES")
    print("Formal Safety Guarantees in Action")
    print("=" * 60 + "\n")
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Example failed: {e}\n")
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
