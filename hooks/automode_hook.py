#!/usr/bin/env python3
"""Claude Code PreToolUse hook: ask the decider-auto judge service about each tool call.

Settings (~/.claude/settings.json), stdlib only:
  {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command",
      "command": "python3 <repo>/hooks/automode_hook.py", "timeout": 10}]}]}}

AUTOMODE_HOOK_MODE   shadow (default): only log the judgment, never decide.
                     enforce: allow -> "allow" (no prompt), block -> "deny" with the rule, ask -> "ask".
AUTOMODE_URL         default http://127.0.0.1:8095/judge
If the service is down or slow, the hook says nothing and Claude Code's normal permission flow applies.
"""
import json
import os
import sys
import urllib.request

URL = os.environ.get("AUTOMODE_URL", "http://127.0.0.1:8095/judge")
MODE = os.environ.get("AUTOMODE_HOOK_MODE", "shadow")


def main():
    try:
        ev = json.load(sys.stdin)
    except ValueError:
        return 0
    if ev.get("hook_event_name", "PreToolUse") != "PreToolUse":
        return 0
    req = {k: ev.get(k) for k in ("session_id", "transcript_path", "cwd", "tool_name", "tool_input", "tool_use_id")}
    try:
        r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(req).encode(), {"Content-Type": "application/json"}),
                                   timeout=float(os.environ.get("AUTOMODE_TIMEOUT", "5")))
        out = json.load(r)
    except Exception:
        return 0
    if MODE != "enforce" or out.get("decision") not in ("allow", "block", "ask"):
        return 0
    decision = {"allow": "allow", "block": "deny", "ask": "ask"}[out["decision"]]
    if decision == "deny":
        reason = (f"Blocked by the decider-auto judge: [{out.get('rule') or out.get('group')}] "
                  f"(p_block={out.get('p_block', 0):.2f}). If the user wants this, they must name this action and its target.")
    else:
        reason = f"decider-auto: {out['decision']} (p_block={out.get('p_block', 0):.2f})"
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision,
                                             "permissionDecisionReason": reason}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
