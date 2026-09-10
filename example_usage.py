from client import ReflexionAgent

def main():
    print("=== Testing Reflexion Agent Verbal Reinforcement ===")
    ref = ReflexionAgent(task_goal="Sort in ascending order")

    # Trial 1: Fails
    s1, traj1, ref1 = ref.execute_trial(
        lambda r: {"action_taken": "descending_sort", "result": [3, 2, 1]},
        lambda t, g: (0.0, False, "Order was descending")
    )
    print("Trial 1 success:", s1)
    print("Reflexion generated:", ref1)
    assert not s1

    # Trial 2: Conditioned on past reflection, succeeds
    s2, traj2, ref2 = ref.execute_trial(
        lambda r: {"action_taken": "ascending_sort", "result": [1, 2, 3]},
        lambda t, g: (1.0, True, "Perfect ascending order")
    )
    print("Trial 2 success:", s2)
    assert s2
    print("Total trials in history:", len(ref.trial_history))
    print("=== All tests passed successfully! ===")

if __name__ == "__main__":
    main()
