"""HTTP judge service for auto mode: stage 1 (the fine-tuned decider, on the GPU, per-session prefix caches) and stage 2 (an
LLM that reads the full effective policy, including the user's own rules) for the cases stage 1 flags. JSONL decision log.

    .venv/bin/python -m automode.server --model models/automode-current --port 8095 \
        --llm_url http://127.0.0.1:8000/v1/chat/completions --llm_model your-stage-2-model

POST /judge  {"session_id", "tool_name", "tool_input", "tool_use_id"?, "transcript_path"? | "user_messages"/"prior_actions",
              "cwd"?, "env"? (env id; default --env), "rules"? ({environment, allow, soft_deny} overrides, "$defaults" splices)}
  -> {"decision": "allow"|"block"|"ask"|"skip", "source": "stage1"|"stage2"|"stage1_fallback", "p_block", "group", "rule",
      "reason", "stage2": {...}?, "ms", ...}
  stage 1 p_block < --pass (--pass_custom when the user changed allow / soft_deny, which stage 1 never learned) -> allow;
  --pass_web raises the threshold for WebFetch/WebSearch/web_url_read (their benign base rate is high; stage 2 still
  sees anything that really looks suspicious). Otherwise stage 2 decides allow / block; --llm_effort_fast (e.g. "none")
  lowers the stage-2 reasoning budget for those read-only web tools. If stage 2 fails, one fast retry (reasoning off,
  --llm_retry_timeout) is tried; if that fails too: block when p_block >= --block, else ask.
  "skip" means the tool never reaches the classifier (read-only tools, edits inside the working directory).
GET /policy?env=<id> -> the default rules (environment, allow, soft_deny, hard_deny) for settings screens
GET /health -> {"ok": true, "model": ..., "stats": {...}}
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from urllib.parse import parse_qs, urlparse

from automode import policy as POL
from automode import stage2 as S2
from automode import transcript as TR
from automode.infer import AutoModeJudge

FILE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
WEB_TOOLS = {"WebFetch", "WebSearch", "mcp__search__searxng_web_search", "mcp__search__web_url_read"}


def pass_threshold(args, tool, customized):
    """stage-1 pass threshold. Web tools (read-only fetches/searches) have a high benign base rate, so --pass_web lets
    them skip stage 2 unless stage 1 is really suspicious; it is ignored when the user changed allow/soft_deny rules,
    because stage 1 never learned those."""
    if getattr(args, "pass_web", 0) and tool in WEB_TOOLS and not (customized["allow"] or customized["soft_deny"]):
        return args.pass_web
    return args.pass_custom if (customized["allow"] or customized["soft_deny"]) else args.pass_


def stage2_effort(args, tool):
    """--llm_effort_fast (e.g. "none") for read-only web tools: their policy decisions are simple and stage 2 at low
    effort spends 200-700 reasoning tokens, which is most of the 2.3 s latency."""
    if getattr(args, "llm_effort_fast", "") and tool in WEB_TOOLS:
        return args.llm_effort_fast
    return args.llm_effort


def stage2_with_retry(args, policy, state, action, hint):
    """Stage 2 with one fast retry. The stage-2 LLM is a shared workstation model: under other users' load it can
    exceed the timeout, and a reasoning reply can come back empty (finish_reason=length). Retrying once with
    reasoning off ("none", a few hundred tokens) recovers both, well inside the Hub's 90 s wait. The first
    attempt's errors are kept in the returned record."""
    first = stage2_effort(args, action.get("tool"))
    attempts = [(first, getattr(args, "llm_timeout", 45.0), S2.MAX_TOKENS),
                ("none" if first != "none" else first, getattr(args, "llm_retry_timeout", 15.0), 4000)]
    errors = []
    for effort, timeout, max_tokens in attempts:
        try:
            r2 = S2.review(args.llm_url, args.llm_model, policy, state, action, hint, effort=effort,
                           timeout=timeout, max_tokens=max_tokens)
            if errors:
                r2["retry"] = {"errors": errors, "effort": effort}
            return r2
        except Exception as e:  # noqa: BLE001  try the next attempt; the caller falls back to stage 1
            errors.append(f"{type(e).__name__}: {e}"[:200])
    raise RuntimeError("; ".join(errors))


def skip_reason(tool, inp, cwd):
    """Tier 1/2 of auto mode: these never reach the classifier."""
    if tool in TR.X.READ_ONLY or tool in TR.X.NOT_GATED:
        return "read_only"
    if tool in FILE_TOOLS and isinstance(inp, dict):
        path = inp.get("file_path") or inp.get("notebook_path")
        if path and cwd:
            p = path if str(path).startswith("/") else os.path.join(cwd, path)
            if TR.X.path_outside(p, cwd) is False and "/.claude/" not in p and "/.git/" not in p:
                return "in_project_edit"
    return None


class Service:
    def __init__(self, args):
        self.args = args
        self.judge = AutoModeJudge(args.model, block_threshold=args.block, ask_threshold=args.ask,
                                   max_sessions=args.max_sessions)
        self.lock = threading.Lock()            # one GPU, one request at a time
        self.env = args.env
        self.log = open(args.log, "a") if args.log else None
        self.log_lock = threading.Lock()
        self.model = args.model

    def handle(self, req):
        tool, inp = req.get("tool_name"), req.get("tool_input") or {}
        sid = req.get("session_id")
        if req.get("transcript_path"):
            state = TR.session_state(req["transcript_path"], req.get("tool_use_id"), inp)
        else:
            state = {k: req.get(k) for k in ("user_messages", "prior_actions", "n_user_messages_total",
                                             "n_prior_actions_total") if req.get(k) is not None}
        cwd = req.get("cwd") or state.get("cwd")
        env_id = req.get("env") if isinstance(req.get("env"), str) else self.env
        rules = req.get("rules") if isinstance(req.get("rules"), dict) else None
        policy = POL.effective(rules, env_id)
        state.update(cwd=cwd, env=policy["environment"], session_id=sid)
        why = skip_reason(tool, inp, cwd)
        t0 = time.perf_counter()
        if why:
            out = {"decision": "skip", "reason": why}
        else:
            with self.lock:
                out = self.judge.judge(state, {"tool": tool, "input": inp}, session_id=sid)
            out["stage1_decision"] = out["decision"]
            out = self.decide(out, policy, rules, env_id, state, {"tool": tool, "input": inp})
        out["ms"] = (time.perf_counter() - t0) * 1000
        if self.log:
            rec = {"t": time.time(), "session_id": sid, "tool": tool, "input": inp, "cwd": cwd, "custom_rules": bool(rules), **out}
            with self.log_lock:
                self.log.write(json.dumps(rec, ensure_ascii=False) + "\n"); self.log.flush()
        return out

    def decide(self, out, policy, rules, env_id, state, action):
        a = self.args
        p = out["p_block"]
        custom = POL.customized(rules, env_id)
        threshold = pass_threshold(a, action.get("tool"), custom)
        if not a.llm_url:                       # stage 1 only: its own block / ask thresholds
            return dict(out, source="stage1")
        if p < threshold:
            return dict(out, decision="allow", source="stage1")
        hint = {"p_block": p, "rule": out.get("rule"), "group": out.get("top_group")}
        try:
            r2 = stage2_with_retry(a, policy, state, action, hint)
        except Exception as e:  # noqa: BLE001  stage 2 unavailable: fall back on stage 1
            dec = "block" if p >= a.block else "ask"
            return dict(out, decision=dec, source="stage1_fallback", stage2={"error": f"{type(e).__name__}: {e}"[:300]})
        return dict(out, decision=r2["decision"], source="stage2", rule=r2["rule"] if r2["decision"] == "block" else None,
                    group=out.get("top_group") if r2["decision"] == "block" else None, reason=r2["reason"], stage2=r2)


def make_handler(svc):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/health":
                self._send(200, {"ok": True, "model": svc.model, "stats": dict(svc.judge.stats), "stage2": bool(svc.args.llm_url)})
            elif u.path == "/policy":
                env_id = (parse_qs(u.query).get("env") or [svc.env])[0]
                self._send(200, POL.defaults(env_id))
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/judge":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                self._send(200, svc.handle(req))
            except Exception as e:  # report, never crash the server
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):
            pass

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--block", type=float, default=0.5, help="p_block threshold for block")
    ap.add_argument("--ask", type=float, default=None, help="p_block threshold for ask (below --block)")
    ap.add_argument("--env", default="default", help="environment id from spec/envs.json / envs.local.json used when a request has none")
    ap.add_argument("--pass", dest="pass_", type=float, default=0.02, help="stage-1 p_block below this is allowed without stage 2")
    ap.add_argument("--pass_custom", type=float, default=0.002, help="the same when the user changed allow / soft_deny rules")
    ap.add_argument("--pass_web", type=float, default=0.0, help="the same for WebFetch/WebSearch/web_url_read (0: use --pass)")
    ap.add_argument("--llm_url", default="", help="OpenAI-compatible chat completions URL for stage 2 (empty: stage 1 only)")
    ap.add_argument("--llm_model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--llm_effort", default="low", help="reasoning_effort sent to the stage-2 LLM ('' to omit)")
    ap.add_argument("--llm_effort_fast", default="", help="the same for web tools (e.g. 'none'; '' to use --llm_effort)")
    ap.add_argument("--llm_timeout", type=float, default=45.0, help="seconds for the first stage-2 attempt")
    ap.add_argument("--llm_retry_timeout", type=float, default=15.0, help="seconds for the one fast retry (reasoning off)")
    ap.add_argument("--max_sessions", type=int, default=16)
    ap.add_argument("--log", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "logs", "automode_server.jsonl"))
    args = ap.parse_args()
    svc = Service(args)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(svc))
    print(f"automode judge on http://{args.host}:{args.port} model={args.model}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
