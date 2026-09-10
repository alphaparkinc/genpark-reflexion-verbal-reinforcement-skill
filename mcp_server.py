import sys
import json
from client import ReflexionAgent

agent = ReflexionAgent("Default Goal")

def handle_call(name, arguments):
    if name == "set_goal":
        agent.goal = arguments["goal"]
        return {"goal": agent.goal}
    elif name == "reflect":
        f = arguments["feedback"]
        act = arguments["action"]
        agent.trial_history.append(({"action_taken": act}, 0.0, False))
        ref = f"Avoid {act} because {f}"
        agent.reflections.append(ref)
        return {"reflection": ref, "history_len": len(agent.trial_history)}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_call(req.get("name"), req.get("arguments", {}))
            print(json.dumps({"id": req.get("id"), "result": res}))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
