"""Read a live Claude Code transcript (.jsonl) into the session state automode.view renders.

Mirrors tools/extract_transcripts.py (the training data came from it): human turns with harness tags stripped,
AskUserQuestion answers as user text, compaction summaries, /clear resets, and the agent's prior non-read-only tool calls
as compact strings. Tool results and assistant prose are not kept: the classifier never sees them.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import extract_transcripts as X  # noqa: E402

MAX_USER = 30   # first message + the most recent ones, as in the training rows
MAX_PRIOR = 20


class _State:
    def __init__(self):
        self.user, self.n_user, self.prior, self.n_prior = [], 0, [], 0
        self.tools = {}     # tool_use id -> name (to recognise AskUserQuestion results)
        self.cwd = None

    def add_user(self, txt):
        self.n_user += 1
        self.user.append(X.trunc(txt, X.USER_MSG_HEAD, X.USER_MSG_TAIL))
        if len(self.user) > MAX_USER:
            self.user = [self.user[0]] + self.user[-(MAX_USER - 1):]


def session_state(path, skip_tool_use_id=None, pending_input=None):
    """State before the proposed call. The proposed call itself may already be in the file (PreToolUse runs after the
    assistant message is written): it is skipped by id, or by matching its input when the id is unknown."""
    st = _State()
    try:
        fh = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return {"user_messages": [], "prior_actions": [], "cwd": None}
    last_uses = []
    with fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("cwd"):
                st.cwd = ev["cwd"]
            t = ev.get("type")
            if t == "user":
                msg = ev.get("message") or {}
                content = msg.get("content")
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_result" and not b.get("is_error"):
                            if st.tools.get(b.get("tool_use_id")) == "AskUserQuestion":
                                txt = X.result_text(b)
                                if txt:
                                    st.add_user("[answer to AskUserQuestion] " + txt)
                has_text = isinstance(content, str) or (
                    isinstance(content, list) and any(isinstance(x, dict) and x.get("type") in ("text", "image") for x in content))
                if not has_text:
                    continue
                raw = X.content_text(content)
                if ev.get("isCompactSummary"):
                    st.user = []
                    st.add_user("[compacted conversation summary] " + raw)
                    continue
                if ev.get("isMeta"):
                    continue
                origin = (ev.get("origin") or {}).get("kind")
                if origin and origin != "human":
                    continue
                txt = X.clean_user_text(raw)
                if not txt:
                    continue
                if txt.startswith("/clear"):
                    st.user, st.prior = [], []
                    continue
                st.add_user(txt)
            elif t == "assistant":
                for b in (ev.get("message") or {}).get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        name, inp = b.get("name") or "?", b.get("input")
                        st.tools[b.get("id")] = name
                        if name in X.READ_ONLY or name in X.NOT_GATED:
                            continue
                        last_uses.append((b.get("id"), inp))
                        st.prior.append(X.action_string(name, inp))
                        st.n_prior += 1
                        if len(st.prior) > MAX_PRIOR * 3:
                            st.prior = st.prior[-MAX_PRIOR:]
    # drop the proposed call if it was already written
    if st.prior and last_uses:
        lid, linp = last_uses[-1]
        if (skip_tool_use_id and lid == skip_tool_use_id) or (pending_input is not None and linp == pending_input):
            st.prior.pop()
            st.n_prior -= 1
    return {"user_messages": st.user, "n_user_messages_total": st.n_user, "prior_actions": st.prior[-MAX_PRIOR:],
            "n_prior_actions_total": st.n_prior, "cwd": st.cwd}
