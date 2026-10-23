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
