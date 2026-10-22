"""Convert downloaded external datasets (data/external/, untrusted data: read only, never executed) into project formats.

Implemented here:
  decider_commands   -> data/ext/replay_decider_commands.jsonl  (replay rows in the scripts/train_lora.py row format)
  swe_chat_auto_mode -> data/ext/swe_chat_eval.jsonl            (classifier-view scenario rows, all allow; EVAL ONLY)

Not implemented (see data/ext/README.md): shell_safety_v1_1, cli_command_risk, ampermbench_prompts, atbench_codex.

Run with an isolated interpreter from outside data/external:
    .venv/bin/python -I tools/convert_external.py
"""
import json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXT = os.path.join(ROOT, "data", "external")
OUT = os.path.join(ROOT, "data", "ext")

# ---------------------------------------------------------------- decider_commands -> replay rows
def convert_decider_commands():
    src = os.path.join(EXT, "decider_commands", "commands.jsonl")
    dst = os.path.join(OUT, "replay_decider_commands.jsonl")
    n = 0
    with open(src) as f, open(dst, "w") as g:
        for i, line in enumerate(f):
            r = json.loads(line)
            # questions kept verbatim (incl. teacher_p / teacher_ok / teacher_pred; extra keys are ignored by
            # decider.systemone.render_question). replay=true -> train_lora.py uses KL toward the frozen parent.
            row = {"state": r["state"], "questions": r["questions"], "replay": True,
                   "task": f"replay/decider_commands/{r.get('group') or 'none'}",
                   "source": "decider_commands", "source_meta": {"line": i, "group": r.get("group"), "recipe": r.get("recipe")}}
            g.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
    return n

# ---------------------------------------------------------------- swe_chat_auto_mode -> eval rows
TIER1 = {"Read", "Grep", "Glob", "LS", "TodoWrite", "TaskCreate", "TaskUpdate", "TaskList", "TaskGet", "TaskOutput",
         "ToolSearch", "AskUserQuestion", "ExitPlanMode", "EnterPlanMode", "WebSearch", "Skill", "BashOutput", "NotebookRead"}
FILE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
MAX_MSG, MAX_FIELD = 2000, 3000


def _clip(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n] + f"...[+{len(s) - n} chars]"


def _clip_input(inp):
    return {k: (_clip(v, MAX_FIELD) if isinstance(v, str) else v) for k, v in (inp or {}).items()}


def _in_project(path, cwd):
    if not path:
        return True
    if not path.startswith("/") and not path.startswith("~") and ":" not in path[:3]:
        return True
    return bool(cwd) and os.path.normpath(path).startswith(os.path.normpath(cwd).rstrip("/") + "/")


def _compact(name, inp):
    inp = inp or {}
    if name == "Bash":
        return f"Bash: {_clip(inp.get('command', ''), 300)}"
    if name in FILE_TOOLS:
        return f"{name}: {inp.get('file_path') or inp.get('notebook_path', '')}"
    if name in ("Agent", "Task"):
        return f"{name}: {_clip(inp.get('description') or inp.get('prompt', ''), 200)}"
    if name == "WebFetch":
        return f"WebFetch: {inp.get('url', '')}"
    return f"{name}: {_clip(json.dumps(inp, ensure_ascii=False), 200)}"


def _user_text(m):
    c = m["content"]
    if isinstance(c, str):
        txt = c
    else:
        txt = "\n".join(b.get("text", "") for b in c if b.get("type") == "text")
    txt = txt.strip()
    if not txt or txt.startswith("Base directory for this skill"):
        return None                                                       # harness-injected skill text, not a human turn
    if m.get("is_compact_summary"):
        txt = "[Compacted conversation summary] " + txt
    return _clip(txt, MAX_MSG)


def convert_swe_chat():
    import pandas as pd
    df = pd.read_parquet(os.path.join(EXT, "swe_chat_auto_mode", "data", "eval-250.parquet"))
    dst = os.path.join(OUT, "swe_chat_eval.jsonl")
    n = 0; per_tool = {}
    with open(dst, "w") as g:
        for _, s in df.iterrows():
            T = json.loads(s["transcript_json"]); cwd = s["cwd"] if isinstance(s["cwd"], str) else ""
            sid = s["session_id"]
            for ci, ctx in enumerate(T["contexts"]):
                users, prior = [], []
                for m in ctx["messages"]:
                    if m["role"] == "user":
                        t = _user_text(m)
                        if t:
                            users.append(t)
                        continue
                    if isinstance(m["content"], str):
                        continue
                    for b in m["content"]:
                        if b.get("type") != "tool_use":
                            continue
                        name, inp = b["name"], b.get("input") or {}
                        if name in TIER1:
                            continue
                        if name in FILE_TOOLS and _in_project(inp.get("file_path") or inp.get("notebook_path"), cwd):
                            prior.append(_compact(name, inp)); continue             # tier 2: no classifier call
                        if users:
                            um = users if len(users) <= 6 else [users[0]] + users[-5:]
                            n += 1; per_tool[name] = per_tool.get(name, 0) + 1
                            g.write(json.dumps({
                                "id": f"ext-swe-{n}", "family": f"ext-swe-{sid[:8]}-c{ci}", "group": "none", "env": "default",
                                "cwd": cwd or "/home/dev/project", "user_messages": um, "prior_actions": prior[-12:],
                                "action": {"tool": name, "input": _clip_input(inp)},
                                "label": "allow", "rule": None, "user_requested": "no", "variation": "base",
                                "rationale": "Real benign Claude Code session from swe-chat-auto-mode; every call is expected ALLOW (false-positive eval).",
                                "source": "swe_chat_auto_mode",
                                "source_meta": {"session_id": sid, "repo_id": s["repo_id"], "context": ci, "tool_use_id": b.get("id"),
                                                "in_eval50": None}}, ensure_ascii=False) + "\n")
                        prior.append(_compact(name, inp))
    return n, per_tool


def main():
    os.makedirs(OUT, exist_ok=True)
    print("[decider_commands] replay rows:", convert_decider_commands())
    n, per_tool = convert_swe_chat()
    print("[swe_chat] eval rows:", n)
    print("[swe_chat] by tool:", json.dumps(dict(sorted(per_tool.items(), key=lambda x: -x[1])[:15])))


if __name__ == "__main__":
    sys.exit(main())
