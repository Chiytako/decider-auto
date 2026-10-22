#!/usr/bin/env python3
"""Mine Claude Code transcripts into labelled auto-mode classifier examples.

Stdlib only. Reads ~/.claude/projects/**/*.jsonl READ-ONLY and writes one JSON
object per agent tool call (default: only tool calls made while auto mode was
active) to an output JSONL file.

Schema facts this relies on (verified on CLI 2.1.278 .. 2.1.302 transcripts):

* Every line is an event with a ``type``: user / assistant / attachment /
  system / permission-mode / mode / ... .
* ``assistant`` events carry ``message.content`` blocks; ``tool_use`` blocks
  have ``id``, ``name``, ``input``. In auto mode the event also carries
  ``serverClassifierRequest`` (an id shared with the result's context).
* ``user`` events carry ``tool_result`` blocks. Since ~2.1.280 a result whose
  call went through the server-side auto-mode classifier carries
  ``serverClassifierContext`` ({request, context:{git_state, live_cwd, ..}}).
  Newer versions also add ``permissionDecision`` = {decision: accept|reject,
  source, reasonType: classifier|mode|rule|other|subcommandResults|safetyCheck}
  and denials carry ``toolDenialKind`` (automode-blocked, user-rejected,
  permission-rule, interrupted).
* Human prompts are ``user`` events with ``permissionMode`` set and
  ``origin.kind == "human"`` (older lines: no origin, not isMeta).
* Subagent transcripts live in ``<project>/<session>/subagents/[workflows/wf_*/]
  agent-<id>.jsonl`` with an ``agent-<id>.meta.json`` next to them (contains
  ``toolUseId`` of the parent's spawning call for Agent-tool subagents).

Usage:
  python3 tools/extract_transcripts.py --out data/real/all.jsonl \
      [--root ~/.claude/projects] [--exclude-session SID ...] [--all-modes]
      [--no-redact] [--claude-md-out data/real/claude_md.jsonl]
"""
import argparse
import bisect
import glob
import hashlib
import json
import os
import re
import sys

PRIOR_N = 20
MAX_USER_MSGS = 30          # first message + most recent (MAX-1)
USER_MSG_HEAD = 1600
USER_MSG_TAIL = 400

# Tools that never go through a permission decision (harness/meta tools).
NOT_GATED = {
    "TodoWrite", "TaskCreate", "TaskUpdate", "TaskList", "TaskGet", "TaskOutput",
    "TaskStop", "ToolSearch", "StructuredOutput", "AskUserQuestion",
    "EnterPlanMode", "ExitPlanMode", "Skill", "SendFeedback", "ScheduleWakeup",
    "ListAgents", "ReportFindings", "CronList",
}
# Read-only lookups: skip the classifier (Tier 1) and, per the docs, are not
# shown to the classifier as prior actions.
READ_ONLY = {"Read", "Grep", "Glob", "LS", "NotebookRead", "ListMcpResourcesTool",
             "ReadMcpResourceTool"}
# Tools the classifier is known (from tagged transcripts) to evaluate.
CLASSIFIED = {"Bash", "WebFetch", "WebSearch", "Agent", "Task", "Workflow", "Monitor",
              "SendMessage", "SubagentHandback", "PowerShell", "Artifact"}
FILE_WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}

CLASSIFIER_DENY_RE = re.compile(
    r"denied by the Claude Code auto mode classifier\. Reason: (?:\[([^\]]+)\]|(.*?)(?:\. If you have| If you have|$))",
    re.S)
NOREASON_MARK = "gave no explanation"

# ---------------------------------------------------------------- redaction
SECRET_PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "anthropic_key"),
    (re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"), "openai_key"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"), "github_token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{30,}"), "github_token"),
    (re.compile(r"hf_[A-Za-z0-9]{30,}"), "hf_token"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws_key_id"),
    (re.compile(r"xox[abposr]-[A-Za-z0-9\-]{10,}"), "slack_token"),
    (re.compile(r"glpat-[A-Za-z0-9_\-]{20,}"), "gitlab_token"),
    (re.compile(r"tskey-[A-Za-z0-9\-]{20,}"), "tailscale_key"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{35}"), "google_api_key"),
    (re.compile(r"ya29\.[0-9A-Za-z_\-]{20,}"), "google_oauth"),
    (re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), "jwt"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.S),
     "private_key"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.=]{20,}"), "bearer"),
    (re.compile(r"(?i)((?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)"
                r"[\"']?\s*[:=]\s*[\"']?)([^\s\"'&;,]{8,})"), "kv_secret"),
]


def redact(s):
    if not isinstance(s, str) or not s:
        return s, 0
    n = 0
    for pat, name in SECRET_PATTERNS:
        if name in ("bearer",):
            s, k = pat.subn(lambda m: m.group(1) + "<REDACTED:%s>" % name, s)
        elif name == "kv_secret":
            def kv(m):
                val = m.group(2)
                if val.startswith("<REDACTED") or val.startswith("$") or val.startswith("${"):
                    return m.group(0)
                return m.group(1) + "<REDACTED:secret>"
            s, k = pat.subn(kv, s)
        else:
            s, k = pat.subn("<REDACTED:%s>" % name, s)
        n += k
    return s, n


def redact_obj(o):
    """Redact strings inside nested JSON-like structure; return (obj, count)."""
    if isinstance(o, str):
        return redact(o)
    if isinstance(o, list):
        out, n = [], 0
        for x in o:
            y, k = redact_obj(x)
            out.append(y)
            n += k
        return out, n
    if isinstance(o, dict):
        out, n = {}, 0
        for kk, v in o.items():
            y, k = redact_obj(v)
            out[kk] = y
            n += k
        return out, n
    return o, 0


# ---------------------------------------------------------------- text utils
def trunc(s, head, tail=0):
    if not isinstance(s, str):
        return s
    if len(s) <= head + tail + 40:
        return s
    if tail:
        return s[:head] + "\n...[%d chars truncated]...\n" % (len(s) - head - tail) + s[-tail:]
    return s[:head] + "...[%d chars truncated]" % (len(s) - head)


def one_line(s, n):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n] + "..."


TAG_DROP_RE = re.compile(
    r"<(system-reminder|local-command-caveat|local-command-stdout|local-command-stderr|"
    r"bash-stdout|bash-stderr|task-notification|command-message)>.*?</\1>", re.S)


def clean_user_text(t):
    """Strip harness tags from a human turn; keep what the user actually wrote."""
    if not t:
        return ""
    # slash commands: <command-name>/goal</command-name> ... <command-args>X</command-args>
    m = re.search(r"<command-name>(.*?)</command-name>", t, re.S)
    if m:
        args = re.search(r"<command-args>(.*?)</command-args>", t, re.S)
        cmd = m.group(1).strip()
        a = args.group(1).strip() if args else ""
        t = TAG_DROP_RE.sub("", t)
        t = re.sub(r"<command-(name|args)>.*?</command-\1>", "", t, flags=re.S).strip()
        return (cmd + (" " + a if a else "") + ("\n" + t if t else "")).strip()
    t = TAG_DROP_RE.sub("", t)
    t = re.sub(r"<bash-input>(.*?)</bash-input>", lambda m: "! " + m.group(1).strip(), t, flags=re.S)
    t = re.sub(r"</?pasted_content[^>]*>", "", t)
    t = t.replace("[Request interrupted by user for tool use]", "").replace(
        "[Request interrupted by user]", "")
    return t.strip()


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif b.get("type") == "image":
                parts.append("[image]")
        return "\n".join(parts)
    return ""


def result_text(block):
    c = block.get("content")
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
    return c if isinstance(c, str) else ""


# ---------------------------------------------------------------- tool views
def compact_input(name, inp):
    """Truncated copy of the tool input (what the classifier is shown)."""
    if not isinstance(inp, dict):
        return inp
    out = {}
    for k, v in inp.items():
        if name == "Bash" and k == "command":
            out[k] = trunc(v, 3500, 500)
        elif name == "Write" and k == "content":
            out[k] = trunc(v, 1500, 300)
            out["content_chars"] = len(v) if isinstance(v, str) else None
        elif name in ("Edit", "MultiEdit") and k in ("old_string", "new_string"):
            out[k] = trunc(v, 800 if k == "old_string" else 1500, 200)
        elif k in ("prompt", "message", "script"):
            out[k] = trunc(v, 3000, 500)
        elif isinstance(v, str):
            out[k] = trunc(v, 2000, 200)
        elif isinstance(v, (list, dict)):
            s = json.dumps(v, ensure_ascii=False)
            out[k] = v if len(s) <= 2500 else trunc(s, 2000, 300)
        else:
            out[k] = v
    return out


def action_string(name, inp, n=400):
    """Compact one-line rendering of a tool call for prior_actions."""
    if not isinstance(inp, dict):
        return "%s: %s" % (name, one_line(str(inp), n))
    if name == "Bash":
        return "Bash: " + one_line(inp.get("command", ""), n)
    if name in ("Write",):
        return "Write: %s (%d chars)" % (inp.get("file_path"), len(inp.get("content") or ""))
    if name in ("Edit", "MultiEdit"):
        return "%s: %s :: %s" % (name, inp.get("file_path"), one_line(inp.get("new_string", ""), 120))
    if name == "WebFetch":
        return "WebFetch: " + str(inp.get("url"))
    if name == "WebSearch":
        return "WebSearch: " + one_line(inp.get("query", ""), n)
    if name in ("Agent", "Task"):
        return "%s: [%s] %s" % (name, inp.get("description", ""), one_line(inp.get("prompt", ""), n - 60))
    s = json.dumps(inp, ensure_ascii=False)
    return "%s: %s" % (name, one_line(s, n))


def path_outside(path, cwd):
    if not path or not cwd:
        return None
    p = os.path.normpath(path)
    c = os.path.normpath(cwd)
    return not (p == c or p.startswith(c.rstrip("/") + "/"))


def git_summary(scc):
    if not isinstance(scc, dict):
        return None
    ctx = scc.get("context") or {}
    gs = ctx.get("git_state") or {}
    vis = gs.get("visibility") or {}
    remotes = []
    for r in (vis.get("visibility_cache") or [])[:4]:
        if isinstance(r, dict):
            remotes.append({k: r.get(k) for k in ("remote", "visibility") if k in r})
    return {
        "live_cwd": ctx.get("live_cwd"),
        "root": gs.get("root"),
        "branch": gs.get("branch"),
        "default_branch": gs.get("default_branch"),
        "status": trunc(gs.get("status"), 300) if isinstance(gs.get("status"), str) else gs.get("status"),
        "remotes": remotes or None,
        "error": gs.get("error"),
    }


# ---------------------------------------------------------------- per file
class FileState:
    def __init__(self, is_sub):
        self.is_sub = is_sub
        self.mode = None
        self.mode_src = None
        self.user_msgs = []       # list of str
        self.n_user_total = 0
        self.prior = []           # compact strings of classifier-visible calls
        self.n_prior_total = 0
        self.last_text = ""       # last assistant prose (not classifier-visible)
        self.claude_md = {}       # path -> sha
        self.compacted = False
        self.pending = {}         # tool_use_id -> partial example
        self.hookdec = {}
        self.first_user_seen = False
        # gated, non-read-only calls that ran WITHOUT a classifier check since the
        # last classifier-checked call (Tier-2 edits, allow-rule hits, ...). The
        # classifier reads the whole transcript, so these can drive the next verdict.
        self.unchecked_since = []
        self.recent_blocks = []   # [(n_prior_total_at_block, rule)]

    def add_user(self, text):
        text = trunc(text, USER_MSG_HEAD, USER_MSG_TAIL)
        self.user_msgs.append(text)
        self.n_user_total += 1
        if len(self.user_msgs) > MAX_USER_MSGS:
            # keep the first message (usually the task) + most recent ones
            self.user_msgs = [self.user_msgs[0]] + self.user_msgs[-(MAX_USER_MSGS - 1):]


def classify_result(block, ev):
    """Return dict with label/block_rule/block_source/result_status/decision_source."""
    txt = result_text(block)
    is_err = bool(block.get("is_error"))
    tdk = ev.get("toolDenialKind")
    pd = ev.get("permissionDecision") or {}
    r = {"label": "allow", "block_rule": None, "block_source": None,
         "result_status": "ok" if not is_err else "error",
         "denial_kind": tdk, "permission_decision": pd or None}
    if tdk == "automode-blocked" or (is_err and "auto mode classifier" in txt and "denied" in txt):
        r["label"] = "block"
        r["result_status"] = "denied"
        if NOREASON_MARK in txt:
            r["block_source"] = "classifier_noreason"
        else:
            m = CLASSIFIER_DENY_RE.search(txt)
            rule = (m.group(1) or m.group(2) or "").strip() if m else None
            r["block_source"] = "classifier"
            r["block_rule"] = rule or None
        return r
    if tdk == "user-rejected" or (is_err and "doesn't want to proceed with this tool use" in txt):
        r.update(label="block", block_source="user_reject", result_status="denied")
        return r
    if tdk == "permission-rule" or (is_err and "denied by a built-in Claude Code safety check" in txt):
        r.update(label="block", block_source="safety_check", result_status="denied")
        return r
    if tdk == "interrupted" or (is_err and txt.startswith("[Tool call interrupted")) \
            or (is_err and txt.startswith("Interrupted: the response that made this tool call was stopped")):
        r.update(label=None, result_status="interrupted")
        return r
    return r


def checked_status(name, inp, cwd, ev, version_has_scc):
    """Did the auto-mode classifier evaluate this call? -> (bool|None, evidence, skip_reason)."""
    pd = ev.get("permissionDecision") or {}
    rt = pd.get("reasonType")
    if ev.get("toolDenialKind") == "automode-blocked":
        return True, "denial", None
    if pd:
        if rt == "classifier":
            return True, "permissionDecision", None
        if pd.get("source") == "user_temporary":
            return False, "permissionDecision", "user_answer"
        if name in NOT_GATED:
            return False, "permissionDecision", "not_gated"
        return False, "permissionDecision", {
            "mode": "mode_auto_allow", "rule": "allow_rule", "other": "tier1_or_tier2",
            "subcommandResults": "allow_rule_subcommands", "safetyCheck": "safety_check",
        }.get(rt, "no_reason_type")
    if "serverClassifierContext" in ev:
        return True, "scc", None
    if version_has_scc:
        if name in NOT_GATED:
            return False, "scc_absent", "not_gated"
        if name in READ_ONLY:
            return False, "scc_absent", "read_only"
        if name in FILE_WRITE_TOOLS:
            return False, "scc_absent", "in_project_edit_or_rule"
        # mostly read-only Bash (cat/grep/sed -n ...) auto-allowed without a classifier call
        return False, "scc_absent", "readonly_bash_or_allow_rule"
    # old CLI without markers: heuristic
    if name in NOT_GATED:
        return False, "heuristic", "not_gated"
    if name in READ_ONLY:
        return False, "heuristic", "read_only"
    if name in FILE_WRITE_TOOLS:
        # Path-vs-cwd is NOT a reliable predictor (additional dirs, scratchpad and
        # memory dirs are auto-allowed; some in-project edits are classified), so unknown.
        return None, "heuristic", None
    if name in CLASSIFIED or name.startswith("mcp__"):
        return None, "heuristic", None
    return None, "heuristic", None


def version_tuple(v):
    try:
        return tuple(int(x) for x in v.split("."))
    except Exception:
        return (0,)


SCC_SINCE = (2, 1, 280)


def process_file(path, info, parent_ctx, claude_md_store, keep_all_modes, do_redact, examples):
    """Parse one transcript; append examples. Returns per-file context (for subagents)."""
    is_sub = info["is_subagent"]
    st = FileState(is_sub)
    ctx_out = {"mode_timeline": [], "human_timeline": [], "tool_mode": {}}
    if is_sub and parent_ctx:
        # inherit parent's permission mode at spawn time
        m = None
        tu = info.get("spawn_tool_use")
        if tu and tu in parent_ctx["tool_mode"]:
            m = parent_ctx["tool_mode"][tu]
        st.parent_mode_timeline = parent_ctx["mode_timeline"]
        st.parent_human = parent_ctx["human_timeline"]
        if m:
            st.mode, st.mode_src = m, "parent_spawn"
    else:
        st.parent_mode_timeline = None
        st.parent_human = None
    parent_users_snapshot = None
    last_cwd = None

    try:
        fh = open(path, "r", encoding="utf-8", errors="replace")
    except OSError as e:
        print("skip %s: %s" % (path, e), file=sys.stderr)
        return ctx_out
    with fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            t = ev.get("type")
            ts = ev.get("timestamp")
            if ev.get("cwd"):
                last_cwd = ev["cwd"]

            if t == "permission-mode":
                pm = ev.get("permissionMode")
                if pm:
                    st.mode, st.mode_src = pm, "permission-mode-event"
                continue
            if t == "attachment":
                a = ev.get("attachment") or {}
                at = a.get("type")
                if at == "auto_mode":
                    st.mode, st.mode_src = "auto", "auto_mode_attachment"
                elif at == "plan_mode":
                    st.mode, st.mode_src = "plan", "plan_mode_attachment"
                elif at == "auto_mode_exit" and st.mode == "auto":
                    st.mode, st.mode_src = "default", "auto_mode_exit"
                elif at == "hook_permission_decision":
                    st.hookdec[a.get("toolUseID")] = a.get("decision")
                elif at in ("nested_memory", "instructions"):
                    files = a.get("files") or ([a.get("content")] if isinstance(a.get("content"), dict) else [])
                    for f in files:
                        if not isinstance(f, dict):
                            continue
                        c = f.get("content") or ""
                        p = f.get("path") or a.get("path")
                        if not c:
                            continue
                        h = hashlib.sha1(c.encode("utf-8", "replace")).hexdigest()[:12]
                        st.claude_md[p] = h
                        if h not in claude_md_store:
                            claude_md_store[h] = {"sha": h, "path": p, "type": f.get("type"), "content": c}
                if ts and st.mode:
                    ctx_out["mode_timeline"].append((ts, st.mode))
                continue
            if t == "system":
                if ev.get("subtype") == "compact_boundary":
                    st.compacted = True
                continue
            if t == "user":
                if ev.get("permissionMode"):
                    st.mode, st.mode_src = ev["permissionMode"], "user_message"
                    if ts:
                        ctx_out["mode_timeline"].append((ts, st.mode))
                msg = ev.get("message") or {}
                content = msg.get("content")
                # ---- tool results
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_result":
                            tid = b.get("tool_use_id")
                            ex = st.pending.pop(tid, None)
                            if ex is None:
                                continue
                            finalize(ex, b, ev, st, examples, keep_all_modes, do_redact)
                            # AskUserQuestion answers are user-provided consent text
                            if ex["tool_name"] == "AskUserQuestion" and not b.get("is_error"):
                                txt = result_text(b)
                                if txt:
                                    st.add_user("[answer to AskUserQuestion] " + txt)
                # ---- text (human / task prompt / notifications)
                has_text = isinstance(content, str) or (
                    isinstance(content, list) and any(isinstance(x, dict) and x.get("type") in ("text", "image")
                                                      for x in content))
                if not has_text:
                    continue
                raw = content_text(content)
                origin = (ev.get("origin") or {}).get("kind")
                if ev.get("isCompactSummary"):
                    st.user_msgs = []
                    st.add_user("[compacted conversation summary] " + raw)
                    st.compacted = True
                    continue
                if is_sub and not st.first_user_seen and ev.get("parentUuid") is None:
                    st.first_user_seen = True
                    st.add_user("[delegated task from parent agent] " + clean_user_text(raw))
                    continue
                if ev.get("isMeta"):
                    continue
                if origin and origin != "human":
                    continue  # task-notification, peer, auto-continuation, coordinator
                txt = clean_user_text(raw)
                if not txt:
                    continue
                if txt.startswith("/clear"):
                    st.user_msgs, st.prior, st.compacted = [], [], False
                    continue
                if is_sub:
                    txt = "[message from parent agent] " + txt
                st.add_user(txt)
                if not is_sub and ts:
                    ctx_out["human_timeline"].append((ts, txt))
                continue
            if t == "assistant":
                msg = ev.get("message") or {}
                content = msg.get("content") or []
                if not isinstance(content, list):
                    continue
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    bt = b.get("type")
                    if bt == "text":
                        if b.get("text", "").strip():
                            st.last_text = b["text"]
                    elif bt == "tool_use":
                        name = b.get("name") or "?"
                        inp = b.get("input")
                        mode = st.mode
                        msrc = st.mode_src
                        if is_sub and st.parent_mode_timeline and msrc != "parent_spawn" and ts:
                            i = bisect.bisect_right([x[0] for x in st.parent_mode_timeline], ts) - 1
                            if i >= 0:
                                mode, msrc = st.parent_mode_timeline[i][1], "parent_timeline"
                        ex = {
                            "id": b.get("id"),
                            "session_id": info["session_id"],
                            "agent_id": info.get("agent_id"),
                            "agent_type": info.get("agent_type"),
                            "project": info["project"],
                            "file": info["rel"],
                            "timestamp": ts,
                            "version": ev.get("version"),
                            "message_id": msg.get("id"),
                            "is_subagent": is_sub,
                            "tool_name": name,
                            "tool_input": compact_input(name, inp),
                            "_raw_input": inp,
                            "user_messages": list(st.user_msgs),
                            "n_user_messages_total": st.n_user_total,
                            "prior_actions": st.prior[-PRIOR_N:],
                            "n_prior_actions_total": st.n_prior_total,
                            "assistant_text_before": trunc(st.last_text, 600) if st.last_text else None,
                            "cwd": ev.get("cwd") or last_cwd,
                            "permission_mode": mode,
                            "mode_source": msrc,
                            "server_classifier_request": "serverClassifierRequest" in ev,
                            "after_compaction": st.compacted,
                            "claude_md": sorted(st.claude_md.values()),
                            "unchecked_since_last_check": list(st.unchecked_since[-10:]),
                            "recent_blocks": [
                                {"rule": r, "actions_ago": st.n_prior_total - n + 1}
                                for n, r in st.recent_blocks if st.n_prior_total - n < PRIOR_N],
                            "_action_str": action_string(name, inp),
                        }
                        if is_sub and st.parent_human is not None:
                            # parent's human turns up to now (sub's own view has only the task)
                            ph = [h[1] for h in st.parent_human if ts and h[0] <= ts]
                            ex["parent_user_messages"] = ([ph[0]] + ph[-9:]) if len(ph) > 10 else ph
                        st.pending[b.get("id")] = ex
                        ctx_out["tool_mode"][b.get("id")] = mode
                        if name not in READ_ONLY and name not in NOT_GATED:
                            st.prior.append(action_string(name, inp))
                            st.n_prior_total += 1
                            if len(st.prior) > PRIOR_N * 3:
                                st.prior = st.prior[-PRIOR_N:]
                        st.last_text = ""
                continue
    # tool calls that never got a result
    for tid, ex in st.pending.items():
        ex.update(label=None, block_rule=None, block_source=None, result_status="missing",
                  denial_kind=None, permission_decision=None, classifier_checked=None,
                  checked_evidence="no_result", skip_reason=None, git_context=None,
                  classifier_request_id=None,
                  hook_decision=st.hookdec.get(tid))
        emit(ex, examples, keep_all_modes, do_redact)
    return ctx_out


def finalize(ex, block, ev, st, examples, keep_all_modes, do_redact):
    r = classify_result(block, ev)
    ex.update(r)
    v = version_tuple(ex.get("version") or "0")
    checked, evidence, skip = checked_status(ex["tool_name"], ex["_raw_input"], ex["cwd"], ev, v >= SCC_SINCE)
    if r["block_source"] in ("classifier", "classifier_noreason"):
        checked, evidence, skip = True, "denial", None
        st.recent_blocks.append((st.n_prior_total, r["block_rule"] or "(no reason)"))
        st.recent_blocks = st.recent_blocks[-10:]
    a = ex.get("_action_str")
    if checked:
        st.unchecked_since = []
    elif ex["tool_name"] not in READ_ONLY and ex["tool_name"] not in NOT_GATED and a \
            and r["label"] == "allow":
        st.unchecked_since.append(a)
    ex["classifier_checked"] = checked
    ex["checked_evidence"] = evidence
    ex["skip_reason"] = skip
    ex["git_context"] = git_summary(ev.get("serverClassifierContext"))
    ex["classifier_request_id"] = (ev.get("serverClassifierContext") or {}).get("request")
    ex["hook_decision"] = st.hookdec.get(ex["id"])
    # evidence that auto mode was on overrides the tracked mode
    if checked and evidence in ("permissionDecision", "scc", "denial"):
        if ex["permission_mode"] != "auto":
            ex["mode_source"] = "classifier_evidence(was %s)" % ex["permission_mode"]
        ex["permission_mode"] = "auto"
    emit(ex, examples, keep_all_modes, do_redact)


def emit(ex, examples, keep_all_modes, do_redact):
    ex.pop("_raw_input", None)
    ex.pop("_action_str", None)
    in_auto = ex["permission_mode"] == "auto" or (
        ex.get("server_classifier_request") and ex["permission_mode"] in (None,))
    ex["in_auto"] = bool(in_auto)
    if not in_auto and not keep_all_modes:
        return
    name = ex["tool_name"]
    ex["gated"] = name not in NOT_GATED
    ex["train_eligible"] = bool(
        ex["in_auto"] and ex["gated"] and ex["label"] in ("allow", "block")
        and ex["block_source"] in (None, "classifier", "classifier_noreason")
        and ex["classifier_checked"] is not False
        and ex.get("hook_decision") is None)
    notes = []
    if ex["classifier_checked"] is None:
        notes.append("classifier involvement unknown (old CLI / heuristic)")
    if ex.get("hook_decision"):
        notes.append("PermissionRequest hook decided: %s" % ex["hook_decision"])
    if ex["result_status"] == "missing":
        notes.append("no tool_result recorded")
    if ex["block_source"] == "user_reject":
        notes.append("rejected by user at a permission prompt; not a classifier decision")
    if ex["block_source"] == "safety_check":
        notes.append("built-in safety check (rm path guard), not the classifier")
    if ex.get("after_compaction"):
        notes.append("after context compaction")
    ex["notes"] = "; ".join(notes) or None
    if do_redact:
        n = 0
        for k in ("tool_input", "user_messages", "prior_actions", "assistant_text_before",
                  "parent_user_messages", "unchecked_since_last_check"):
            if k in ex:
                ex[k], c = redact_obj(ex[k])
                n += c
        ex["redactions"] = n
    examples.append(ex)


# ---------------------------------------------------------------- discovery
def discover(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True))
    sessions = {}
    for f in files:
        rel = os.path.relpath(f, root)
        parts = rel.split(os.sep)
        project = parts[0]
        if "subagents" in parts:
            sid = parts[1]
            agent_id = os.path.basename(f)[:-6].replace("agent-", "")
            meta = {}
            mp = f[:-6] + ".meta.json"
            if os.path.exists(mp):
                try:
                    meta = json.load(open(mp))
                except Exception:
                    meta = {}
            info = {"rel": rel, "project": project, "session_id": sid, "is_subagent": True,
                    "agent_id": agent_id, "agent_type": meta.get("agentType"),
                    "spawn_tool_use": meta.get("toolUseId"),
                    "workflow": parts[3] if len(parts) > 4 and parts[2] == "workflows" else None}
        elif len(parts) == 2:
            sid = parts[1][:-6]
            info = {"rel": rel, "project": project, "session_id": sid, "is_subagent": False}
        else:
            continue  # memory dirs etc.
        sessions.setdefault((project, sid), {"main": None, "subs": []})
        if info["is_subagent"]:
            sessions[(project, sid)]["subs"].append((f, info))
        else:
            sessions[(project, sid)]["main"] = (f, info)
    return sessions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--claude-md-out", default=None,
                    help="write unique CLAUDE.md / memory instruction contents (keyed by sha) here")
    ap.add_argument("--exclude-session", action="append", default=[],
                    help="session id (dir/file stem) to skip, e.g. the live session running this tool")
    ap.add_argument("--all-modes", action="store_true", help="also emit calls outside auto mode")
    ap.add_argument("--no-redact", action="store_true")
    args = ap.parse_args()

    sessions = discover(args.root)
    examples, cmd_store = [], {}
    nfiles = 0
    for (project, sid), grp in sorted(sessions.items()):
        if sid in args.exclude_session:
            continue
        parent_ctx = None
        if grp["main"]:
            f, info = grp["main"]
            parent_ctx = process_file(f, info, None, cmd_store, args.all_modes, not args.no_redact, examples)
            nfiles += 1
        for f, info in grp["subs"]:
            process_file(f, info, parent_ctx, cmd_store, args.all_modes, not args.no_redact, examples)
            nfiles += 1

    # dedupe identical tool_use ids (resumed sessions copy history into a new file)
    seen, out = set(), []
    for ex in examples:
        if ex["id"] in seen:
            continue
        seen.add(ex["id"])
        out.append(ex)
    # parallel tool calls emitted in one assistant message share one classifier request;
    # a block on one sibling can coincide with blocks on the others
    per_msg = {}
    for ex in out:
        per_msg.setdefault((ex["file"], ex.get("message_id")), []).append(ex)
    for grp in per_msg.values():
        nb = sum(1 for e in grp if e["block_source"] in ("classifier", "classifier_noreason"))
        for i, e in enumerate(grp):
            e["batch_size"] = len(grp)
            e["batch_index"] = i
            e["batch_classifier_blocks"] = nb
    # mark duplicate actions (same tool + same input + same outcome) for dedup-aware stats
    first = {}
    for ex in sorted(out, key=lambda e: e.get("timestamp") or ""):
        key = hashlib.sha1(json.dumps([ex["tool_name"], ex["tool_input"], ex["label"], ex["block_rule"]],
                                      sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        ex["action_key"] = key
        ex["dup_of"] = first.get(key)
        first.setdefault(key, ex["id"])
    out.sort(key=lambda e: (e.get("timestamp") or "", e["id"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fo:
        for ex in out:
            fo.write(json.dumps(ex, ensure_ascii=False) + "\n")
    if args.claude_md_out:
        with open(args.claude_md_out, "w", encoding="utf-8") as fo:
            for v in cmd_store.values():
                if not args.no_redact:
                    v = dict(v, content=redact(v["content"])[0])
                fo.write(json.dumps(v, ensure_ascii=False) + "\n")
    print("files=%d examples=%d (raw %d)" % (nfiles, len(out), len(examples)), file=sys.stderr)


if __name__ == "__main__":
    main()
