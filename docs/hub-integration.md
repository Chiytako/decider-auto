# Integration: the judge service and the agent side

The model is used through `automode/server.py`, which keeps one model on the GPU and a prefix cache per session.

## HTTP API

### `POST /judge`

```json
{"session_id": "user:conversation",
 "tool_name": "Bash", "tool_input": {"command": "...", "description": "..."},
 "user_messages": ["...", "..."], "prior_actions": ["Bash: git status", "Edit: README.md :: ..."],
 "cwd": "/home/dev/projects/billing-api",
 "env": "default",
 "rules": {"environment": ["$defaults"], "allow": ["$defaults", "..."], "soft_deny": ["$defaults", "..."]}}
```

- `prior_actions` are compact strings in the same form the training data used (`Tool: summary`).
- `env` selects an environment from `spec/envs.json` / `spec/envs.local.json`; `rules` are the user's overrides.
- `session_id` scopes the prefix cache: send the same id for consecutive calls of one conversation and the state as an
  append-only transcript (environment, messages, actions, proposed action last).

Response:

```json
{"decision": "allow|block|ask|skip", "source": "stage1|stage2|stage1_fallback",
 "p_block": 0.97, "stage1_decision": "block", "group": "vcs_review", "top_group": "vcs_review",
 "rule": "Git Destructive", "reason": "...", "p_user_requested": 0.03, "ms": 1290}
```

- `allow` / `block`: run it / do not run it. Give the agent `rule` and `reason` when blocking (the agent may ask the user, who
  can then name the action - the retry is judged again).
- `ask`: a stage-2 fallback or the caller's own policy wants a confirmation; show the user `p_block` and the group/rule.
- `skip`: the call never reaches the classifier (read-only tools, edits inside the working directory); the caller should
  follow its normal permission flow.
- If the service is unreachable the agent should not fail closed by default: keep working as before the feature, or ask the
  user (a deployment choice).

### `GET /policy?env=<id>`

The default rules (environment, allow, soft_deny, hard_deny) for a settings screen that shows what can be overridden.

### `GET /health`

`{"ok": true, "model": "...", "stats": {...}, "stage2": true|false}`.

## Agent side (as deployed in the GB10 Hub)

The hub (a JupyterHub running the pi coding agent in per-user containers) applies the modes and rules itself and calls this
service for Auto:

- **Permission modes per conversation**: Auto (this classifier + rules), Plan (research only; the agent presents a plan and
  the user switches to Auto/Edit), Edit (file edits inside the working directory are free, everything else asks).
- **Permission patterns** `allow` / `ask` / `deny` (`Bash(git push:*)`, `Edit(/path/**)`, `WebFetch(domain:...)`,
  `mcp__server__tool`), deny > ask > allow, applied before the judge; read-only tools and read-only commands never reach it.
- **Rule overrides** per user in the `autoMode` shape; the judge receives them in `rules` and stage 2 reads them.
- **Records**: every judged call is stored with the user's answer to an ask and an optional verdict
  (`correct` / `should_allow` / `should_ask` / `should_block`); `tools/import_hub.py` turns them into training rows.

That hub code is not in this repository. The Claude Code `PreToolUse` hook in `hooks/automode_hook.py` is a smaller,
standalone integration: `AUTOMODE_HOOK_MODE=shadow` logs only, `enforce` maps the response to allow / ask / deny.
