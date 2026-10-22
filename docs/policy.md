# The policy: rules, consent, and the two stages

## What the classifier decides

Claude Code's auto mode is a set of rules plus a classifier that applies them. The rules come from `claude auto-mode defaults`
and are **not redistributed in this repository** (they are Claude Code's own text); generate them locally:

```bash
claude auto-mode defaults > spec/automode_defaults.json     # keys: environment, allow, soft_deny, hard_deny
```

The policy has four lists:

| list | meaning |
|---|---|
| `hard_deny` | never cleared, not even by explicit user consent (one rule: data exfiltration) |
| `soft_deny` | blocked unless the user's own messages name the specific action and its specifics, as the rule's "must name" clause says |
| `allow` | exceptions that clear a block (read-only operations, declared dependencies, local project operations, memory-directory writes, ...) |
| `environment` | the trust context: trusted repo and remotes, visibility, internal domains, buckets, package registry, sensitive data locations, production targets, host containment |

Precedence: **hard_deny > soft_deny > allow exceptions > explicit user intent**. Vague goals ("clean up", "deploy when ready",
"fix CI") do not authorize the risky specific; boundaries the user states ("don't push", "don't touch prod") block matching
actions. Eight rules can only be cleared by the user calling that exact flagged action a false positive.

The classifier sees **only** the environment, the user's messages and the agent's tool calls. Tool results and the agent's
prose are stripped on purpose, so an instruction in a file or a web page is not consent and the agent cannot argue its case.

## Rule taxonomy (`spec/groups.json`)

The 73 deny rules are grouped into 13 labels for one softmax; the second pass picks the exact rule within a group. Group ids:

`vcs_review`, `deletion`, `production_infra`, `shared_compute`, `credentials`, `data_exfil`, `security_controls`,
`network_exposure`, `publication`, `code_provenance`, `agent_self`, `external_systems`, `browser` (plus "none of these").

## Two stages (`automode/server.py`)

1. **Stage 1** - this model, one forward pass, three slots. `p_block < --pass` (default 0.02) is allowed immediately; with the
   user's own `allow` / `soft_deny` overrides, the threshold drops to `--pass_custom` (0.002) because stage 1 never learned
   those rules.
2. **Stage 2** - any OpenAI-compatible LLM (`--llm_url`, `--llm_model`, `--llm_effort`). It receives the *effective* policy
   (defaults spliced with the user's rules) and the same state, and returns allow / block, the deciding rule and a one-sentence
   reason. Measured 1.6-5.8 s per flagged action with a reasoning model at low effort.
3. If stage 2 fails (LLM down), `p_block >= --block` (0.5) blocks, otherwise the caller should ask the user.

## Overrides (Claude Code `autoMode` shape)

`POST /judge` may carry `rules: {"environment": [...], "allow": [...], "soft_deny": [...]}`:

- A list **replaces** the default list; a list containing the string `$defaults` splices the defaults in at that position.
- `hard_deny` cannot be overridden.
- `spec/envs.json` holds generic environments; `spec/envs.local.json` (git-ignored) holds your own. The request's `env` id
  selects one, and a request-level `environment` override wins.

## Permission patterns (applied by the caller)

The judge service does not apply glob patterns; the agent integration does, in Claude Code's `permissions` shape
(`Bash(git push:*)`, `Edit(/path/**)`, `WebFetch(domain:host)`, `mcp__server__tool`), with deny > ask > allow and the bash
command split at `&& ; |` for allow vs deny/ask. Read-only operations (file reads, searches, and read-only commands such as
`ls`, `cat`, `grep`, `git status/log/diff` with no redirects or command substitution) never reach the classifier at all.
