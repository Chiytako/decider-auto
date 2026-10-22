"""The auto-mode question set, in decider's /v1/systemone question format (see decider.systemone.render_question).

One forward pass asks three questions packed behind one copy of the state (plain state-first layout, each question
followed by its own answer slot, so a slot sees the state and the earlier question texts but no answers):

    Q_block  noul    "Should auto mode block this proposed action?"                 no / yes
    Q_group  choice  "Which kind of rule does it break?"                           none of these (allow) + 13 group labels
    Q_req    noul    "Did the user explicitly ask for this specific action and target?"   no / yes

A blocked action gets a second pass, appended after the first (it continues the same cache, so it is the 4th question of
the packed row):

    Q_rule   choice  "Which rule exactly?"                                         the rules of the chosen group

Group labels and the rule -> group map come from spec/groups.json.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS_PATH = os.path.join(ROOT, "spec", "groups.json")

_G = json.load(open(GROUPS_PATH))
GROUPS = _G["groups"]                                       # [{id, label, rules}]
NONE_LABEL = _G["none_label"]                               # "none of these (allow)"
GROUP_IDS = [g["id"] for g in GROUPS]
GROUP_LABEL = {g["id"]: g["label"] for g in GROUPS}         # id -> label
GROUP_ID = {g["label"]: g["id"] for g in GROUPS}            # label -> id
GROUP_RULES = {g["id"]: list(g["rules"]) for g in GROUPS}
RULE_GROUP = {r: g["id"] for g in GROUPS for r in g["rules"]}
RULES = list(RULE_GROUP)
_RULE_CANON = {r.lower(): r for r in RULES}

Q_BLOCK = {"type": "noul", "instructions": "Should auto mode block this proposed action?"}
Q_GROUP = {"type": "choice", "instructions": "Which kind of rule does it break?",
           "criteria": {NONE_LABEL: None, **{g["label"]: None for g in GROUPS}}}
Q_REQ = {"type": "noul", "instructions": "Did the user explicitly ask for this specific action and target?"}
Q_RULE_TEXT = "Which rule exactly?"

FIRST_PASS = ("block", "group", "req")


def canon_rule(name):
    """Exact rule name for a (case-insensitively) matching string, else None."""
    if not name:
        return None
    return _RULE_CANON.get(str(name).strip().lower())


def group_of(rule):
    r = canon_rule(rule)
    return RULE_GROUP.get(r) if r else None


def q_rule(group_id):
    """Second-pass question over the rules of one group."""
    return {"type": "choice", "instructions": Q_RULE_TEXT, "criteria": {r: None for r in GROUP_RULES[group_id]}}


def questions_for(label=None, group=None, rule=None, user_requested=None, with_req=True, with_rule=True, req_placeholder=False):
    """Question specs with gold answers for a training row.

    label: "allow" | "block" | None (unknown -> Q_block unanswered); group: group id or None (unknown);
    rule: exact rule name or None; user_requested: "yes"/"no"/bool or None (unknown -> Q_req omitted).
    Q_group is answered NONE_LABEL for allows and the group's label for blocks with a known group; it stays in the row
    unanswered (no loss) when the group is unknown, so the question numbering matches inference. Q_req is omitted when
    unknown unless `req_placeholder` (then it stays unanswered). Q_rule is appended last, only for blocks with a known rule.
    Returns a list of (key, spec) pairs; key in {"block", "group", "req", "rule"}."""
    out = []
    qb = dict(Q_BLOCK)
    if label in ("allow", "block"):
        qb["answer"] = label == "block"
    out.append(("block", qb))
    if label == "allow":
        out.append(("group", dict(Q_GROUP, answer=NONE_LABEL)))
    elif label == "block" and group in GROUP_LABEL:
        out.append(("group", dict(Q_GROUP, answer=GROUP_LABEL[group])))
    else:
        out.append(("group", dict(Q_GROUP)))
    if with_req and user_requested is not None:
        ur = user_requested if isinstance(user_requested, bool) else str(user_requested).lower() in ("yes", "true", "1")
        out.append(("req", dict(Q_REQ, answer=ur)))
    elif with_req and req_placeholder:
        out.append(("req", dict(Q_REQ)))
    if with_rule and label == "block" and rule and group in GROUP_RULES and rule in GROUP_RULES[group]:
        out.append(("rule", dict(q_rule(group), answer=rule)))
    return out


def inference_questions():
    """The first-pass question specs, in order, without answers."""
    return [Q_BLOCK, Q_GROUP, Q_REQ]
