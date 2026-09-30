from __future__ import annotations

import pytest

from fabweave.core.events import EventLog
from fabweave.core.timebox import TimeBox
from fabweave.dispatch.freeze import (
    BATCH_CLASSES,
    ENGINEERING,
    HOT,
    NORMAL,
    FreezeBook,
    FreezeOverride,
    FreezeRule,
    freeze_decision,
    override_status,
)
from fabweave.dispatch.priority import DispatchCandidate
from fabweave.workflows.control import (
    freeze_decisions,
    rank_under_freeze,
    record_freeze_decision,
)


def rule(rule_id: str, version: int, priority: int, window: TimeBox, classes: frozenset[str], reason: str = "audit_hold", effective_at: int = 0) -> FreezeRule:
    return FreezeRule(rule_id, version, priority, effective_at, window, classes, reason)


def test_no_matching_rule_is_open_fail_open_boundary():
    rules = [rule("FRZ-GLOBAL", 1, 0, TimeBox(10, 20), frozenset(BATCH_CLASSES))]
    decision = freeze_decision(rules, [], NORMAL, "LOT-A", 20)
    assert decision.allowed and decision.evidence["matched_rule_ids"] == ()


def test_freeze_scope_is_per_batch_class_not_all_or_nothing():
    rules = [rule("FRZ-HOT", 1, 0, TimeBox(0, 100), frozenset({HOT, ENGINEERING}))]
    assert freeze_decision(rules, [], NORMAL, "LOT-N", 50).allowed
    denied = freeze_decision(rules, [], HOT, "LOT-H", 50)
    assert denied.reasons == ("dispatch_frozen:FRZ-HOT",)
    assert denied.evidence["winning_rule_reason"] == "audit_hold"
    assert freeze_decision(rules, [], ENGINEERING, "LOT-E", 50).reasons[0] == "dispatch_frozen:FRZ-HOT"


def test_overlapping_rules_resolve_by_priority_then_version_then_effective_time():
    rules = [
        rule("FRZ-LOW", 1, 1, TimeBox(0, 100), frozenset({NORMAL})),
        rule("FRZ-HIGH", 1, 5, TimeBox(0, 100), frozenset({NORMAL})),
        rule("FRZ-HIGH2", 1, 5, TimeBox(0, 100), frozenset({NORMAL})),
        rule("FRZ-HIGH2", 2, 5, TimeBox(0, 100), frozenset({NORMAL})),
    ]
    decision = freeze_decision(rules, [], NORMAL, "LOT-A", 50)
    assert decision.evidence["winning_rule_id"] == "FRZ-HIGH2"
    assert decision.evidence["winning_rule_version"] == 2
    assert decision.evidence["matched_rule_ids"] == ("FRZ-HIGH2", "FRZ-HIGH", "FRZ-LOW")

    newer = rule("FRZ-LATER", 1, 5, TimeBox(0, 100), frozenset({NORMAL}), effective_at=10)
    tied = [
        rule("FRZ-EARLIER", 1, 5, TimeBox(0, 100), frozenset({NORMAL}), effective_at=0),
        newer,
    ]
    # Same priority and version: the later-effective rule wins.
    assert freeze_decision(tied, [], NORMAL, "LOT-A", 50).evidence["winning_rule_id"] == "FRZ-LATER"


def test_time_boxed_override_releases_only_inside_its_window_and_scope():
    rules = [rule("FRZ-HOT", 1, 0, TimeBox(0, 100), frozenset({HOT}))]
    overrides = [
        FreezeOverride("OVR-1", "FRZ-HOT", TimeBox(40, 60), frozenset({HOT}), "SUP-1", "customer_escalation"),
    ]
    before = freeze_decision(rules, overrides, HOT, "LOT-H", 39)
    assert before.reasons == ("dispatch_frozen:FRZ-HOT",)
    released = freeze_decision(rules, overrides, HOT, "LOT-H", 50)
    assert released.allowed and released.evidence["override_id"] == "OVR-1"
    assert released.evidence["override_approver"] == "SUP-1"
    assert released.evidence["override_valid_until"] == 60
    after = freeze_decision(rules, overrides, HOT, "LOT-H", 60)
    assert after.reasons == ("dispatch_frozen:FRZ-HOT",)
    statuses = after.evidence["override_statuses"]
    assert statuses == ("OVR-1:expired",)
    # An override for a different rule/scope never releases engineering.
    rules_eng = [rule("FRZ-ENG", 1, 0, TimeBox(0, 100), frozenset({ENGINEERING}))]
    assert freeze_decision(rules_eng, overrides, ENGINEERING, "LOT-E", 50).reasons[0] == "dispatch_frozen:FRZ-ENG"


def test_override_can_be_narrowed_to_specific_lots():
    rules = [rule("FRZ-HOT", 1, 0, TimeBox(0, 100), frozenset({HOT}))]
    overrides = [
        FreezeOverride("OVR-1", "FRZ-HOT", TimeBox(0, 100), frozenset({HOT}), "SUP-1", "spare", frozenset({"LOT-A"})),
    ]
    assert freeze_decision(rules, overrides, HOT, "LOT-A", 50).allowed
    decision = freeze_decision(rules, overrides, HOT, "LOT-B", 50)
    assert decision.reasons == ("dispatch_frozen:FRZ-HOT",)
    assert decision.evidence["override_statuses"] == ("OVR-1:scope_mismatch",)


def test_revoked_and_expired_overrides_fail_closed():
    rules = [rule("FRZ-HOT", 1, 0, TimeBox(0, 100), frozenset({HOT}))]
    overrides = [
        FreezeOverride("OVR-OLD", "FRZ-HOT", TimeBox(0, 40), frozenset({HOT}), "SUP-1", "old"),
        FreezeOverride("OVR-KILL", "FRZ-HOT", TimeBox(0, 100), frozenset({HOT}), "SUP-2", "new", revoked_at=30),
    ]
    assert freeze_decision(rules, overrides, HOT, "LOT-H", 50).reasons == ("dispatch_frozen:FRZ-HOT",)
    decision = freeze_decision(rules, overrides, HOT, "LOT-H", 50)
    assert set(decision.evidence["override_statuses"]) == {"OVR-KILL:revoked", "OVR-OLD:expired"}
    assert override_status(overrides[1], HOT, "LOT-H", 20) == "active"


def test_active_overrides_choose_latest_expiring_and_report_all():
    rules = [rule("FRZ-HOT", 1, 0, TimeBox(0, 100), frozenset({HOT}))]
    overrides = [
        FreezeOverride("OVR-SHORT", "FRZ-HOT", TimeBox(0, 50), frozenset({HOT}), "SUP-1", "short"),
        FreezeOverride("OVR-LONG", "FRZ-HOT", TimeBox(0, 80), frozenset({HOT}), "SUP-2", "long"),
    ]
    decision = freeze_decision(rules, overrides, HOT, "LOT-H", 20)
    assert decision.allowed
    assert decision.evidence["override_id"] == "OVR-LONG"
    assert set(decision.evidence["active_override_ids"]) == {"OVR-SHORT", "OVR-LONG"}


def test_freeze_book_versions_overrides_revocation_and_replay():
    book = FreezeBook()
    book.publish_rule(rule("FRZ-NORM", 1, 0, TimeBox(0, 100), frozenset({NORMAL})))
    with pytest.raises(ValueError):
        book.publish_rule(rule("FRZ-NORM", 1, 0, TimeBox(0, 100), frozenset({HOT})))
    with pytest.raises(ValueError):
        book.publish_rule(rule("FRZ-NORM", 0, 0, TimeBox(0, 100), frozenset({NORMAL})))
    with pytest.raises(ValueError):
        book.grant_override(FreezeOverride("OVR-1", "FRZ-UNKNOWN", TimeBox(0, 10), frozenset({NORMAL}), "SUP-1", "x"))

    grant = FreezeOverride("OVR-1", "FRZ-NORM", TimeBox(20, 60), frozenset({NORMAL}), "SUP-1", "line_balance")
    assert book.grant_override(grant) is grant
    assert book.grant_override(grant) is grant  # idempotent
    with pytest.raises(ValueError):
        book.grant_override(FreezeOverride("OVR-1", "FRZ-NORM", TimeBox(20, 70), frozenset({NORMAL}), "SUP-1", "line_balance"))

    assert book.evaluate(NORMAL, "LOT-A", 10).reasons == ("dispatch_frozen:FRZ-NORM",)
    assert book.evaluate(NORMAL, "LOT-A", 30).allowed
    book.revoke_override("OVR-1", 40)
    assert book.evaluate(NORMAL, "LOT-A", 50).reasons == ("dispatch_frozen:FRZ-NORM",)
    # A revision can widen scope without rewriting history.
    book.publish_rule(rule("FRZ-NORM", 2, 0, TimeBox(0, 100), frozenset({NORMAL, HOT}), effective_at=0))
    assert {item.version for item in book.rule_history("FRZ-NORM")} == {1, 2}
    assert book.evaluate(HOT, "LOT-H", 50).reasons == ("dispatch_frozen:FRZ-NORM",)


def test_rule_validation_rejects_empty_scope_and_unknown_classes():
    with pytest.raises(ValueError):
        FreezeRule("FRZ-1", 1, 0, 0, TimeBox(0, 10), frozenset(), "reason")
    with pytest.raises(ValueError):
        FreezeRule("FRZ-1", 1, 0, 0, TimeBox(0, 10), frozenset({"rush"}), "reason")
    with pytest.raises(ValueError):
        freeze_decision([], [], "rush", "LOT-A", 1)


def test_rank_under_freeze_keeps_overridden_lots_and_drops_frozen_ones():
    rules = [rule("FRZ-NORM", 1, 0, TimeBox(0, 100), frozenset({NORMAL}))]
    overrides = [
        FreezeOverride("OVR-1", "FRZ-NORM", TimeBox(0, 100), frozenset({NORMAL}), "SUP-1", "ok", frozenset({"LOT-A"})),
    ]
    candidates = [
        DispatchCandidate("LOT-A", 10, 5),
        DispatchCandidate("LOT-B", 10, 5),
        DispatchCandidate("LOT-H", 1, 1),
    ]
    classes = {"LOT-A": NORMAL, "LOT-B": NORMAL, "LOT-H": HOT}
    decisions = freeze_decisions(rules, overrides, candidates, classes, 50)
    assert decisions["LOT-A"].allowed and not decisions["LOT-B"].allowed and decisions["LOT-H"].allowed
    ranked = rank_under_freeze(rules, overrides, candidates, classes, 50)
    assert [item.lot_id for item in ranked] == ["LOT-H", "LOT-A"]


def test_recorded_decisions_are_idempotent_and_replay_their_explanation():
    log = EventLog()
    rules = [rule("FRZ-HOT", 1, 0, TimeBox(0, 100), frozenset({HOT}))]
    denied = freeze_decision(rules, [], HOT, "LOT-H", 50)
    first = record_freeze_decision(log, "REQ-1", denied, 50)
    replay = record_freeze_decision(log, "REQ-1", denied, 50)
    assert first == replay and log.read("freeze:REQ-1")[0].kind == "dispatch_freeze_denied"
    assert first.decision.reasons == ("dispatch_frozen:FRZ-HOT",)
    assert first.decision.evidence["matched_rule_ids"] == ["FRZ-HOT"]

    override = FreezeOverride("OVR-1", "FRZ-HOT", TimeBox(40, 80), frozenset({HOT}), "SUP-1", "customer")
    released = freeze_decision(rules, [override], HOT, "LOT-H", 50)
    stored = record_freeze_decision(log, "REQ-2", released, 50)
    assert stored.allowed and stored.decision.evidence["override_id"] == "OVR-1"
    assert log.read("freeze:REQ-2")[0].kind == "dispatch_freeze_released"
