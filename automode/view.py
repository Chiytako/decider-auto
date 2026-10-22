"""Serialize an auto-mode scenario into the decider "state" text (the Context of the prompt).

The classifier sees what the real auto-mode classifier sees: the environment, the human user's messages, the agent's
prior tool calls (not their results) and the proposed tool call. Tool outputs and assistant prose are never rendered.

Layout (plain text, append-only so a session prefix can be KV-cached; the proposed action is always last):

    Environment:
    - Organization: ...
    Working directory: /home/dev/projects/foo
    Git: repo /home/dev/projects/foo, branch feature-x (default main); remotes: acme/foo (private)

    User messages (oldest first):
    [1] first message
    [... 2 earlier messages omitted ...]
    [4] most recent message

    Agent actions so far (oldest first; results are not shown):
    (3 earlier actions not shown)
    - Bash: git status
    - Edit: src/app.py
    Proposed action:
    Git status now: 3 modified
    Tool: Bash
    command: git push origin feature-x
    description: Push the branch

`render()` returns the two halves: `prefix` (everything up to and including the newline after the last prior action)
and `suffix` (the proposed action, starting at "Proposed action:"). Every section boundary inside the prefix is a single
"\\n" followed by a non-newline character, and the Qwen pre-tokenizer never merges across such a boundary, so
tok(prefix + suffix) == tok(prefix) + tok(suffix), and a prefix that grows by one more prior action line extends the
previous prefix's tokens exactly (automode.infer then runs only the new line). There is deliberately no blank line before
"Proposed action:": "\\n\\n" is one token, which would break that extension. A few fields that change on every call
(git status, a cwd that moved) are placed in the suffix so they do not invalidate the cache.

Token budget (default 1800 tokens of the decider-4b tokenizer, counted on the state alone): strings are cut first
(commands ~1200 chars keeping head and tail, user messages ~700 chars, prior actions ~400 chars); then the oldest prior
actions are dropped down to a few recent ones (in steps that keep the cut point stable, PRIOR_DROP_STEP), then middle
user messages (the first and the last are always kept), then the
remaining prior actions; as a last resort the kept user messages and the action fields are cut harder. The proposed action
is never dropped.

Accepted inputs (`normalize`):
  * synthetic rows (spec/scenario_schema.md): env, cwd, user_messages, prior_actions, action{tool,input}
  * real rows (data/real/all.jsonl): tool_name, tool_input, user_messages, n_user_messages_total, prior_actions,
    n_prior_actions_total, cwd, git_context (environment defaults to "gb10_lab")
  * a session/action pair as used by automode.infer (same keys as a synthetic row)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVS_PATH = os.path.join(ROOT, "spec", "envs.json")
DEFAULT_MODEL = os.path.join(ROOT, "models", "decider-4b")
REAL_ENV = "gb10_effective"   # the user's effective autoMode environment (claude auto-mode config)

DEFAULT_BUDGET = 2300          # the gb10_effective environment alone is ~640 tokens
# character limits applied before the token budget
LIMITS = dict(
    command=1200,        # Bash command, Agent/Workflow prompt, Write content
    field=600,           # any other action input field
    user=700,            # each user message
    user_min=300,        # first/last user message when the budget is still exceeded after all drops
    prior=400,           # each prior action line
    action_min=300,      # floor for action fields under budget pressure
)
LONG_FIELDS = ("command", "content", "prompt", "new_string", "old_string", "code", "script", "query", "body", "text")
KEEP_RECENT_PRIOR = 6    # prior actions kept before middle user messages start being dropped
PRIOR_DROP_STEP = 8      # when old actions are dropped, the first shown action's absolute index is a multiple of this, so
                         # the rendered prefix stays the same for several consecutive calls (prefix-cache hits)

_ENVS = None
_TOK = {}


def envs():
    global _ENVS
    if _ENVS is None:
        _ENVS = json.load(open(ENVS_PATH))
        local = os.path.join(os.path.dirname(ENVS_PATH), "envs.local.json")
        if os.path.exists(local):
            _ENVS.update(json.load(open(local)))
    return _ENVS


def get_tokenizer(path=DEFAULT_MODEL):
    """The decider tokenizer (cached). Loading takes about a second, CPU only."""
    if path not in _TOK:
        from transformers import AutoTokenizer
        _TOK[path] = AutoTokenizer.from_pretrained(path)
    return _TOK[path]


# ---------------------------------------------------------------------------------------------------------------------
# normalization

@dataclass
class Scenario:
    env_lines: list
    cwd: str | None
    git: dict | None                      # real git_context (compact form from tools/extract_transcripts.py)
    user_messages: list
    user_index0: int                      # 1-based number of user_messages[1] (gap after the first message when > 2)
    prior_actions: list
    prior_hidden: int                     # earlier prior actions not present in the row
    tool: str
    tool_input: object
    extra: dict = field(default_factory=dict)


def normalize(row, env=None):
    """Any supported row -> Scenario. `env` overrides the environment id (or pass a list of lines)."""
    if "action" in row and isinstance(row["action"], dict):
        tool, inp = row["action"].get("tool"), row["action"].get("input", {})
    else:
        tool, inp = row.get("tool_name") or row.get("tool"), row.get("tool_input", row.get("input", {}))
    env = env if env is not None else row.get("env_lines") or row.get("env") or (REAL_ENV if "tool_name" in row else "default")
    env_lines = list(env) if isinstance(env, (list, tuple)) else envs().get(env, envs()["default"])
    um = [str(m) for m in (row.get("user_messages") or [])]
    n_total = int(row.get("n_user_messages_total") or len(um))
    # real rows keep the first + last 29 turns: number them as in the full conversation
    idx0 = n_total - len(um) + 2 if len(um) >= 2 else 2
    pa = [str(a) for a in (row.get("prior_actions") or [])]
    hidden = max(0, int(row.get("n_prior_actions_total") or len(pa)) - len(pa))
    return Scenario(env_lines=env_lines, cwd=row.get("cwd"), git=row.get("git_context") or None, user_messages=um,
                    user_index0=idx0, prior_actions=pa, prior_hidden=hidden, tool=str(tool), tool_input=inp)


# ---------------------------------------------------------------------------------------------------------------------
# string helpers

def cut(s, limit, head_frac=2 / 3):
    """Keep the head and the tail of a long string, with a marker saying how much was cut."""
    s = str(s)
    if limit is None or len(s) <= limit:
        return s
    h = max(1, int(limit * head_frac)); t = max(1, limit - h)
    return f"{s[:h]} …[{len(s) - h - t} chars cut]… {s[-t:]}"


def one_line(s):
    return " ".join(str(s).split())


def _value_lines(key, val, limit):
    """`key: value` for single-line strings, a YAML-like `key: |` block for multi-line ones, compact JSON otherwise."""
    if not isinstance(val, str):
        val = json.dumps(val, ensure_ascii=False, separators=(",", ":"))
    val = cut(val.rstrip(), limit)
    if "\n" not in val:
        return [f"{key}: {val}"]
    return [f"{key}: |"] + ["  " + ln.rstrip() for ln in val.split("\n")]


def render_action(tool, inp, long_limit, field_limit):
    lines = [f"Tool: {tool}"]
    if isinstance(inp, dict):
        for k, v in inp.items():
            if v is None or v == "":
                continue
            lines += _value_lines(k, v, long_limit if k in LONG_FIELDS else field_limit)
    elif inp not in (None, ""):
        lines += _value_lines("input", inp, long_limit)
    return lines


def _git_stable(g):
    """The parts of git_context that stay fixed through a session (repo, branch, remotes) as one line, or None."""
    if not g:
        return None
    parts = []
    if g.get("error") == "not_a_repo" or not g.get("root"):
        parts.append("not inside a git repository")
    else:
        b = g.get("branch") or "?"
        d = f" (default {g['default_branch']})" if g.get("default_branch") else ""
        parts.append(f"repo {g['root']}, branch {b}{d}")
    rem = g.get("remotes") or []
    if rem:
        parts.append("remotes: " + ", ".join(f"{r.get('remote')} ({r.get('visibility') or 'unknown'})" for r in rem))
    return "Git: " + "; ".join(parts)


def _git_volatile(g, cwd):
    """Per-call git facts (working-tree status, a cwd that moved), rendered in the suffix."""
    out = []
    if not g:
        return out
    if g.get("live_cwd") and cwd and g["live_cwd"] != cwd:
        out.append(f"Current directory now: {g['live_cwd']}")
    st = g.get("status")
    if isinstance(st, dict) and g.get("root"):
        if st.get("clean"):
            out.append("Git status now: clean")
        else:
            c = st.get("counts") or {}
            bits = [f"{c[k]} {k}" for k in ("staged", "modified") if c.get(k)]
            u = c.get("untracked_normal") or c.get("untracked")
            if u:
                bits.append(f"{u} untracked")
            out.append("Git status now: " + (", ".join(bits) or "dirty"))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# rendering

@dataclass
class View:
    prefix: str
    suffix: str
    n_tokens: int | None = None           # tokens of prefix + suffix (None without a tokenizer)
    dropped_prior: int = 0
    dropped_users: int = 0
    over_budget: bool = False

    @property
    def text(self):
        return self.prefix + self.suffix


def _count(tok, s):
    if tok is None:
        return max(1, len(s) // 3)          # rough fallback: ~3 chars per token for mixed code / CJK text
    return len(tok.encode(s, add_special_tokens=False))


def render(row, tok=None, budget=DEFAULT_BUDGET, include_git=True, env=None, limits=None):
    """Row -> View(prefix, suffix). `tok=None` skips the token budget (only the character limits apply)."""
    L = dict(LIMITS, **(limits or {}))
    sc = row if isinstance(row, Scenario) else normalize(row, env=env)

    head = ["Environment:"] + [f"- {one_line(x)}" for x in sc.env_lines]
    if sc.cwd:
        head.append(f"Working directory: {sc.cwd}")
    if include_git:
        g = _git_stable(sc.git)
        if g:
            head.append(g)
    head_txt = "\n".join(head)

    # user messages: (number, text) pairs
    users = []
    for i, m in enumerate(sc.user_messages):
        n = 1 if i == 0 else sc.user_index0 + i - 1
        users.append((n, m.strip()))
    prior = [cut(one_line(a), L["prior"]) for a in sc.prior_actions]
    user_limit = L["user"]
    long_lim, field_lim = L["command"], L["field"]
    vol = _git_volatile(sc.git, sc.cwd) if include_git else []

    def user_line(n, m, lim):
        m = cut(m, lim)
        return f"[{n}] " + m.replace("\n", "\n    ")       # indent continuation lines under the number

    def build(users_kept, prior_kept, prior_dropped, ulim, alim, flim):
        u_lines = ["User messages (oldest first):"]
        if not users_kept:
            u_lines.append("(none)")
        for j, (n, m) in enumerate(users_kept):
            if j > 0:
                gap = n - users_kept[j - 1][0] - 1
                if gap > 0:
                    u_lines.append(f"[... {gap} earlier message{'s' if gap > 1 else ''} omitted ...]")
            first_or_last = j == 0 or j == len(users_kept) - 1
            u_lines.append(user_line(n, m, ulim if first_or_last else min(ulim, L["user"])))
        p_lines = ["Agent actions so far (oldest first; results are not shown):"]
        hidden = sc.prior_hidden + prior_dropped
        if hidden:
            p_lines.append(f"({hidden} earlier action{'s' if hidden > 1 else ''} not shown)")
        p_lines += [f"- {a}" for a in prior_kept] if prior_kept else (["(none)"] if not hidden else [])
        prefix = head_txt + "\n\n" + "\n".join(u_lines) + "\n\n" + "\n".join(p_lines) + "\n"
        suffix = "\n".join(["Proposed action:"] + vol + render_action(sc.tool, sc.tool_input, alim, flim))
        return prefix, suffix

    prefix, suffix = build(users, prior, 0, user_limit, long_lim, field_lim)
    view = View(prefix, suffix)
    if tok is None:
        return view

    total = _count(tok, prefix + suffix)
    users_kept, prior_kept, dropped_p, dropped_u = list(users), list(prior), 0, 0
    # cost of each removable piece (approximate, re-measured after every change)
    while total > budget:
        if len(prior_kept) > KEEP_RECENT_PRIOR:
            prior_kept.pop(0); dropped_p += 1
        elif len(users_kept) > 2:
            users_kept.pop(1); dropped_u += 1
        elif prior_kept:
            prior_kept.pop(0); dropped_p += 1
        elif user_limit > L["user_min"]:
            user_limit = L["user_min"]
        elif long_lim > L["action_min"] or field_lim > L["action_min"]:
            long_lim = max(L["action_min"], long_lim // 2); field_lim = max(L["action_min"], field_lim // 2)
        else:
            view.over_budget = True
            break
        prefix, suffix = build(users_kept, prior_kept, dropped_p, user_limit, long_lim, field_lim)
        total = _count(tok, prefix + suffix)
    # align the cut so the prefix is stable while the session grows (deterministic in the row: train == serve)
    extra = (-(sc.prior_hidden + dropped_p)) % PRIOR_DROP_STEP if dropped_p else 0
    if extra and dropped_u == 0 and len(prior_kept) - extra >= KEEP_RECENT_PRIOR:
        prior_kept = prior_kept[extra:]; dropped_p += extra
        prefix, suffix = build(users_kept, prior_kept, dropped_p, user_limit, long_lim, field_lim)
        total = _count(tok, prefix + suffix)
    view.prefix, view.suffix, view.n_tokens = prefix, suffix, total
    view.dropped_prior, view.dropped_users = dropped_p, dropped_u
    return view


def serialize(row, tok=None, budget=DEFAULT_BUDGET, **kw):
    """Row -> the state string (prefix + suffix)."""
    return render(row, tok=tok, budget=budget, **kw).text


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Print the serialized state of rows from a JSONL file.")
    ap.add_argument("path")
    ap.add_argument("-n", type=int, default=2)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--no_tok", action="store_true")
    a = ap.parse_args()
    tok = None if a.no_tok else get_tokenizer()
    with open(a.path) as f:
        for i, line in enumerate(f):
            if i < a.offset:
                continue
            if i >= a.offset + a.n:
                break
            v = render(json.loads(line), tok=tok, budget=a.budget)
            sys.stdout.write(f"===== row {i}: {v.n_tokens} tokens, dropped prior {v.dropped_prior}, users {v.dropped_users}\n")
            sys.stdout.write(v.text + "\n")
