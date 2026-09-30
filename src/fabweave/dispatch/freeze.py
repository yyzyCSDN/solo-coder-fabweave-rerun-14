from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from fabweave.core.decision import Decision, allow, deny
from fabweave.core.ids import validate_id
from fabweave.core.timebox import TimeBox


NORMAL = "normal"
HOT = "hot"
ENGINEERING = "engineering"
BATCH_CLASSES: tuple[str, ...] = (NORMAL, HOT, ENGINEERING)


def _normalize_classes(value: Iterable[str]) -> frozenset[str]:
    classes = frozenset(value)
    if not classes:
        raise ValueError("at least one batch class is required")
    unknown = sorted(classes.difference(BATCH_CLASSES))
    if unknown:
        raise ValueError(f"unknown batch class: {', '.join(unknown)}")
    return classes


@dataclass(frozen=True, slots=True)
class FreezeRule:
    """A freeze that blocks dispatch while its window is active.

    Scope is limited to ``batch_classes`` so a freeze never has to be all-on
    or all-off.  Overlapping rules are resolved by priority, then version,
    then effective time (newest wins each tie); ``rule_id`` is only a stable
    final tie-breaker.
    """

    rule_id: str
    version: int
    priority: int
    effective_at: int
    window: TimeBox
    batch_classes: frozenset[str]
    reason: str

    def __post_init__(self) -> None:
        validate_id(self.rule_id)
        if self.version <= 0:
            raise ValueError("rule version must be positive")
        if not self.reason:
            raise ValueError("freeze rule requires a reason")
        object.__setattr__(self, "batch_classes", _normalize_classes(self.batch_classes))


@dataclass(frozen=True, slots=True)
class FreezeOverride:
    """A time-boxed authorization that releases one freeze rule.

    It is valid only inside its half-open window, so it expires automatically.
    ``lot_ids`` may narrow the grant to specific lots; an empty set covers
    every lot in ``batch_classes``.
    """

    override_id: str
    rule_id: str
    window: TimeBox
    batch_classes: frozenset[str]
    approver: str
    reason: str
    lot_ids: frozenset[str] = frozenset()
    revoked_at: int | None = None

    def __post_init__(self) -> None:
        validate_id(self.override_id)
        validate_id(self.rule_id)
        if not self.approver or not self.reason:
            raise ValueError("override requires an approver and a reason")
        object.__setattr__(self, "batch_classes", _normalize_classes(self.batch_classes))
        object.__setattr__(self, "lot_ids", frozenset(self.lot_ids))
        if self.revoked_at is not None and self.revoked_at < self.window.start:
            raise ValueError("override cannot be revoked before it is granted")


def override_status(override: FreezeOverride, batch_class: str, lot_id: str, at: int) -> str:
    if batch_class not in override.batch_classes or (override.lot_ids and lot_id not in override.lot_ids):
        return "scope_mismatch"
    if override.revoked_at is not None and at >= override.revoked_at:
        return "revoked"
    if at < override.window.start:
        return "pending"
    if at >= override.window.end:
        return "expired"
    return "active"


def _matching_rules(rules: Iterable[FreezeRule], batch_class: str, at: int) -> list[FreezeRule]:
    matched = [
        rule
        for rule in rules
        if rule.effective_at <= at and rule.window.contains(at) and batch_class in rule.batch_classes
    ]
    ordered = sorted(
        matched,
        key=lambda rule: (-rule.priority, -rule.version, -rule.effective_at, rule.rule_id),
    )
    # A raw list may carry several revisions of the same rule; keep the one
    # that precedence would pick.
    winners: dict[str, FreezeRule] = {}
    for candidate in ordered:
        winners.setdefault(candidate.rule_id, candidate)
    return list(winners.values())


def freeze_decision(
    rules: Iterable[FreezeRule],
    overrides: Iterable[FreezeOverride],
    batch_class: str,
    lot_id: str,
    at: int,
) -> Decision:
    """Evaluate one dispatch request against the freeze layer.

    Pure over immutable records, so a past allow or denial can be replayed
    later by passing the same history and an earlier ``at``.
    """

    if batch_class not in BATCH_CLASSES:
        raise ValueError("unknown batch class")
    ordered = _matching_rules(rules, batch_class, at)
    evidence: dict[str, object] = {
        "evaluated_at": at,
        "batch_class": batch_class,
        "lot_id": lot_id,
        "matched_rule_ids": tuple(rule.rule_id for rule in ordered),
    }
    if not ordered:
        return allow(**evidence)

    winner = ordered[0]
    evidence.update(
        winning_rule_id=winner.rule_id,
        winning_rule_version=winner.version,
        winning_rule_priority=winner.priority,
        winning_rule_reason=winner.reason,
    )
    considered = sorted(
        (
            (candidate, override_status(candidate, batch_class, lot_id, at))
            for candidate in overrides
            if candidate.rule_id == winner.rule_id
        ),
        key=lambda pair: pair[0].override_id,
    )
    evidence["override_statuses"] = tuple(
        f"{candidate.override_id}:{status}" for candidate, status in considered
    )
    active = [candidate for candidate, status in considered if status == "active"]
    if not active:
        return deny(f"dispatch_frozen:{winner.rule_id}", **evidence)

    active.sort(key=lambda candidate: (-candidate.window.end, candidate.override_id))
    chosen = active[0]
    evidence.update(
        override_id=chosen.override_id,
        override_approver=chosen.approver,
        override_reason=chosen.reason,
        override_valid_until=chosen.window.end,
        active_override_ids=tuple(candidate.override_id for candidate in active),
    )
    return allow(**evidence)


class FreezeBook:
    """Append-only registry of rule revisions and timed overrides."""

    def __init__(self) -> None:
        self._rules: dict[str, list[FreezeRule]] = {}
        self._overrides: dict[str, FreezeOverride] = {}

    def publish_rule(self, rule: FreezeRule) -> FreezeRule:
        history = self._rules.setdefault(rule.rule_id, [])
        if history:
            last = history[-1]
            if rule.version == last.version:
                if rule == last:
                    return last
                raise ValueError("rule version conflicts with a published revision")
            if rule.version < last.version or rule.effective_at < last.effective_at:
                raise ValueError("freeze rule revisions must move forward")
        history.append(rule)
        return rule

    def grant_override(self, override: FreezeOverride) -> FreezeOverride:
        existing = self._overrides.get(override.override_id)
        if existing is not None:
            if existing == override:
                return existing
            raise ValueError("override id was reused with different content")
        if override.rule_id not in self._rules:
            raise ValueError("override references an unknown freeze rule")
        self._overrides[override.override_id] = override
        return override

    def revoke_override(self, override_id: str, at: int) -> FreezeOverride:
        existing = self._overrides.get(override_id)
        if existing is None:
            raise LookupError("unknown override")
        if existing.revoked_at is not None:
            if existing.revoked_at == at:
                return existing
            raise ValueError("override revocation time is fixed")
        if at < existing.window.start:
            raise ValueError("override cannot be revoked before it is granted")
        revoked = replace(existing, revoked_at=at)
        self._overrides[override_id] = revoked
        return revoked

    def evaluate(self, batch_class: str, lot_id: str, at: int) -> Decision:
        return freeze_decision(self.rules(), self.overrides(), batch_class, lot_id, at)

    def rules(self) -> tuple[FreezeRule, ...]:
        return tuple(rule for history in self._rules.values() for rule in history)

    def rule_history(self, rule_id: str) -> tuple[FreezeRule, ...]:
        return tuple(self._rules.get(rule_id, ()))

    def overrides(self, rule_id: str | None = None) -> tuple[FreezeOverride, ...]:
        if rule_id is None:
            return tuple(self._overrides.values())
        return tuple(candidate for candidate in self._overrides.values() if candidate.rule_id == rule_id)
