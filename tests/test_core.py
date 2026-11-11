"""Unit tests for the parts that need no GPU and no policy text. Run with `pytest -q tests` (or `python -m pytest`).

Policy composition tests use a stand-in for the defaults because spec/automode_defaults.json is not distributed
(`claude auto-mode defaults > spec/automode_defaults.json` generates it locally).
"""

import json
import os

import pytest

from automode import policy as P
from automode import questions as QS
from automode import stage2 as S2

FAKE_DEFAULTS = {
    "environment": ["default environment line"],
    "allow": ["Read-Only Operations: reading is fine"],
    "soft_deny": ["Git Destructive: force-push unless named", "Production Deploy: deploys unless named"],
    "hard_deny": ["Data Exfiltration: never"],
}


@pytest.fixture
def fake_defaults(monkeypatch):
    monkeypatch.setattr(P, "_DEFAULTS", FAKE_DEFAULTS)
    return FAKE_DEFAULTS


def test_splice(fake_defaults):
    assert P.splice(fake_defaults["allow"], None) == fake_defaults["allow"]
    assert P.splice(fake_defaults["allow"], []) == []
    assert P.splice(fake_defaults["allow"], ["$defaults", "Mine: ok"]) == fake_defaults["allow"] + ["Mine: ok"]
    assert P.splice(fake_defaults["allow"], ["Mine: ok", "$defaults"]) == ["Mine: ok"] + fake_defaults["allow"]
    assert P.splice(fake_defaults["allow"], ["  spaced  ", "", 3]) == ["spaced"]   # blanks and non-strings dropped


def test_effective_and_customized(fake_defaults):
    eff = P.effective(None, "default")
    assert eff["soft_deny"] == fake_defaults["soft_deny"] and eff["hard_deny"] == fake_defaults["hard_deny"]
    assert P.customized(None, "default") == {"environment": False, "allow": False, "soft_deny": False}

    rules = {"soft_deny": ["$defaults", "Dataset Protection: delete only when named"], "allow": ["$defaults"]}
    eff = P.effective(rules, "default")
    assert eff["soft_deny"][-1].startswith("Dataset Protection") and len(eff["soft_deny"]) == 3
    assert P.customized(rules, "default") == {"environment": False, "allow": False, "soft_deny": True}
    # hard_deny cannot be overridden
    assert P.effective({"hard_deny": []}, "default")["hard_deny"] == fake_defaults["hard_deny"]


def test_rule_name():
    assert P.rule_name("Git Destructive [named+specifics — must name: X]: text") == "Git Destructive"
    assert P.rule_name("Plain") == "Plain"


def test_questions_round_trip():
    assert QS.canon_rule("git destructive") == "Git Destructive"
    assert QS.group_of("git destructive") == "vcs_review"
    assert QS.group_of("no such rule") is None
    spec = QS.q_rule("credentials")
    assert set(spec["criteria"]) == set(QS.GROUP_RULES["credentials"])
    rows = QS.questions_for(label="block", group="vcs_review", rule="Git Destructive", user_requested="no")
    assert [k for k, _ in rows] == ["block", "group", "req", "rule"]
    assert [q["instructions"] for _, q in rows][:3] == [QS.Q_BLOCK["instructions"], QS.Q_GROUP["instructions"], QS.Q_REQ["instructions"]]
    assert rows[3][1]["instructions"] == QS.Q_RULE_TEXT and rows[1][1]["answer"] == QS.GROUP_LABEL["vcs_review"]
    assert rows[2][1]["answer"] is False
    allow = QS.questions_for(label="allow")
    assert allow[0][1]["answer"] is False and allow[1][1]["answer"] == QS.NONE_LABEL
    assert QS.questions_for(label="block", group="vcs_review")[1][1]["answer"] == QS.GROUP_LABEL["vcs_review"]


def test_stage2_prompt_and_case():
    pol = {"environment": ["e1", "e2"], "hard_deny": ["H: hard"], "soft_deny": ["S: soft"], "allow": ["A: allow"]}
    sys_p = S2.system_prompt(pol)
    assert "H: hard" in sys_p and "S: soft" in sys_p and "e1" in sys_p and "$defaults" not in sys_p
    case = S2.case_text({"cwd": "/w", "user_messages": ["fix it"], "prior_actions": ["Bash: ls"]},
                        {"tool": "Bash", "input": {"command": "git push"}}, hint={"p_block": 0.4, "rule": "Git Destructive"})
    assert '"git push"' in case and "Git Destructive" in case and "p=0.40" in case
    assert "proposed_action" in S2.case_text({"user_messages": []}, {"tool": "Read", "input": {}})


def test_stage2_reply_parsing(monkeypatch):
    """A reasoning model may wrap the JSON in fences/prose, or leave content empty and put everything in
    reasoning_content (finish_reason=length). Both used to end as "判定できませんでした"."""
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data)
        seen["timeout"] = timeout
        payload = seen["reply"]

        class R:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(payload).encode()
        return R()

    monkeypatch.setattr(S2.urllib.request, "urlopen", fake_urlopen)
    pol = {"environment": [], "hard_deny": [], "soft_deny": [], "allow": []}
    state, action = {"user_messages": []}, {"tool": "Bash", "input": {"command": "x"}}

    seen["reply"] = {"choices": [{"message": {"content": "Sure.\n```json\n{\"decision\": \"block\", \"rule\": \"R\", \"reason\": \"why\"}\n```"},
                                 "finish_reason": "stop"}]}
    r = S2.review("http://x", "m", pol, state, action)
    assert r["decision"] == "block" and r["rule"] == "R"
    assert seen["body"]["max_tokens"] == S2.MAX_TOKENS and seen["body"]["max_tokens"] >= 16000

    seen["reply"] = {"choices": [{"message": {"content": "", "reasoning_content": "maybe {noise} then {\"decision\": \"allow\"}"},
                                 "finish_reason": "length"}]}
    r = S2.review("http://x", "m", pol, state, action)
    assert r["decision"] == "allow" and r["rule"] is None

    seen["reply"] = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
    with pytest.raises(ValueError, match="length"):
        S2.review("http://x", "m", pol, state, action)


def test_stage2_retry(monkeypatch):
    """A timeout or empty reply gets one fast retry (reasoning off) before falling back to stage 1."""
    pytest.importorskip("torch")
    from types import SimpleNamespace
    from automode.server import stage2_with_retry
    a = SimpleNamespace(llm_url="http://x", llm_model="m", llm_effort="low", llm_effort_fast="none",
                        llm_timeout=45.0, llm_retry_timeout=15.0)
    calls = []

    def fake(url, model, policy, state, action, hint=None, effort="low", timeout=60.0, max_tokens=0):
        calls.append((effort, timeout, max_tokens))
        if len(calls) == 1:
            raise TimeoutError("timed out")
        return {"decision": "allow", "rule": None, "reason": "", "ms": 1.0}

    monkeypatch.setattr(S2, "review", fake)
    r = stage2_with_retry(a, {}, {}, {"tool": "Bash", "input": {}}, {})
    assert [c[0] for c in calls] == ["low", "none"] and calls[1][1] == 15.0
    assert r["decision"] == "allow" and "timed out" in r["retry"]["errors"][0]

    calls.clear()

    def always_fail(*args, **kw):
        calls.append((kw["effort"], kw["timeout"], kw["max_tokens"]))
        raise ValueError("empty reply")

    monkeypatch.setattr(S2, "review", always_fail)
    with pytest.raises(RuntimeError) as e:
        stage2_with_retry(a, {}, {}, {"tool": "WebFetch", "input": {}}, {})   # first effort is already "none"
    assert len(calls) == 2 and calls[0][0] == "none" and "empty reply" in str(e.value)


def test_skip_reason():
    pytest.importorskip("torch")
    from automode.server import skip_reason
    assert skip_reason("Read", {"file_path": "/etc/passwd"}, "/w") == "read_only"
    assert skip_reason("Grep", {"pattern": "x"}, "/w") == "read_only"
    assert skip_reason("Edit", {"file_path": "/w/a.py"}, "/w") == "in_project_edit"
    assert skip_reason("Edit", {"file_path": "a.py"}, "/w") == "in_project_edit"
    assert skip_reason("Edit", {"file_path": "/w/.git/config"}, "/w") is None
    assert skip_reason("Write", {"file_path": "/other/a.py"}, "/w") is None
    assert skip_reason("Bash", {"command": "ls"}, "/w") is None


def test_envs_split():
    envs = json.load(open(os.path.join(P.SPEC, "envs.json")))
    assert "default" in envs
    assert not {"gb10_effective", "gb10hub_container"} & set(envs)      # private environments stay out of the public file


def test_web_thresholds_and_stage2_effort():
    pytest.importorskip("torch")
    from types import SimpleNamespace
    from automode.server import pass_threshold, stage2_effort
    a = SimpleNamespace(pass_=0.004, pass_custom=0.002, pass_web=0.05, llm_effort="low", llm_effort_fast="none")
    plain = {"allow": False, "soft_deny": False}
    assert pass_threshold(a, "WebFetch", plain) == 0.05
    assert pass_threshold(a, "mcp__search__searxng_web_search", plain) == 0.05
    assert pass_threshold(a, "mcp__search__web_url_read", plain) == 0.05
    assert pass_threshold(a, "Bash", plain) == 0.004
    assert pass_threshold(a, "WebFetch", {"allow": True, "soft_deny": False}) == 0.002   # 規則を変えた人には効かせない
    a.pass_web = 0.0
    assert pass_threshold(a, "WebFetch", plain) == 0.004
    assert stage2_effort(a, "WebFetch") == "none"
    assert stage2_effort(a, "Bash") == "low"
    a.llm_effort_fast = ""
    assert stage2_effort(a, "WebFetch") == "low"


def test_stage2_cooldown_and_stage1_only(monkeypatch):
    """A failed stage 2 is skipped for --llm_cooldown seconds; stage1_only never calls it."""
    from types import SimpleNamespace
    from automode import server as SV
    a = SimpleNamespace(llm_url="http://x", llm_model="m", pass_=0.004, pass_custom=0.002, pass_web=0.0, block=0.5, llm_cooldown=60.0)
    svc = SimpleNamespace(args=a, s2_down_until=0.0)
    calls = []

    def boom(*_a, **_k):
        calls.append(1)
        raise RuntimeError("timed out")

    monkeypatch.setattr(SV, "stage2_with_retry", boom)
    monkeypatch.setattr(SV.POL, "customized", lambda rules, env: {"allow": False, "soft_deny": False, "environment": False})
    act = {"tool": "Bash", "input": {}}
    r = SV.Service.decide(svc, {"p_block": 0.05, "decision": "allow"}, {}, None, "e", {}, act)
    assert (r["decision"], r["source"], len(calls)) == ("ask", "stage1_fallback", 1)
    r = SV.Service.decide(svc, {"p_block": 0.7, "decision": "block"}, {}, None, "e", {}, act)   # cooling down: not called again
    assert (r["decision"], r["source"], len(calls)) == ("block", "stage1_fallback", 1)
    assert "skipped" in r["stage2"]["error"]
    r = SV.Service.decide(svc, {"p_block": 0.001, "decision": "allow"}, {}, None, "e", {}, act)  # stage 1 still passes
    assert (r["decision"], r["source"]) == ("allow", "stage1")
    svc.s2_down_until = 0.0
    r = SV.Service.decide(svc, {"p_block": 0.05, "decision": "allow"}, {}, None, "e", {}, act, stage1_only=True)
    assert (r["decision"], r["source"], len(calls)) == ("ask", "stage1_fallback", 1)
    monkeypatch.setattr(SV, "stage2_with_retry", lambda *_a, **_k: {"decision": "allow", "rule": None, "reason": "fine"})
    r = SV.Service.decide(svc, {"p_block": 0.05, "decision": "allow"}, {}, None, "e", {}, act)
    assert (r["decision"], r["source"]) == ("allow", "stage2")
