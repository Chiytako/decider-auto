# Synthetic scenario row (one JSON object per line)

Each row is ONE proposed agent action judged by the auto-mode classifier, plus the context the classifier sees.
The classifier sees: the environment, the human user's messages, the agent's prior (non-read-only) tool calls, and the proposed action.
It does NOT see tool results or the assistant's prose.

```json
{
  "id": "<group>-<file_tag>-<n>",             // unique
  "family": "<group>-<file_tag>-<family n>",   // rows that are contrastive twins share a family
  "group": "<group id from spec/groups.json, or 'none' for benign-only families>",
  "env": "<env id from spec/envs.json>",
  "cwd": "/home/dev/projects/foo",
  "user_messages": ["first user message", "...", "most recent user message"],   // human turns only, 1-6 items
  "prior_actions": ["Bash: git status", "Edit: src/app.py", "..."],            // 0-12 compact strings, most recent last
  "action": {"tool": "Bash", "input": {"command": "...", "description": "..."}}, // or Write/Edit/WebFetch/Agent/mcp__*/Chrome-MCP tools, etc.
  "label": "allow" | "block",
  "rule": "<exact rule name from spec/groups.json>" | null,   // null iff allow
  "user_requested": "yes" | "no",   // did the user explicitly name this action AND its specific target (consent per the rule's 'must name' clause)?
  "variation": "base" | "consent_added" | "consent_vague" | "scope_changed" | "target_changed" | "env_changed" | "carry_over" | "obfuscated" | "benign_lookalike" | "other",
  "rationale": "one sentence citing the deciding rule/exception"
}
```
