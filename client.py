class ReflexionAgent:
    """
    Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al.).
    Maintains trial memory, evaluates success/failure against goal, generates verbal
    self-reflections on mistakes, and incorporates past reflections into future attempts.
    """
    def __init__(self, task_goal):
        self.goal = task_goal
        self.reflections = []
        self.trial_history = []

    def execute_trial(self, actor_policy_fn, evaluator_fn):
        trajectory = actor_policy_fn(self.reflections)
        score, success, feedback = evaluator_fn(trajectory, self.goal)
        self.trial_history.append((trajectory, score, success))

        if not success:
            reflection = f"Trial {len(self.trial_history)} failed ({feedback}). In next attempt: avoid {trajectory['action_taken']} and adjust strategy."
            self.reflections.append(reflection)
            return False, trajectory, reflection
        return True, trajectory, "Goal reached successfully!"
