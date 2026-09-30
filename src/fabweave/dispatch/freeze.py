from __future__ import annotations

from dataclasses import dataclass

from fabweave.core.decision import Decision, allow, deny
from fabweave.core.timebox import TimeBox


NORMAL = "NORMAL"
HOT = "HOT"
ENGINEERING = "ENGINEERING"
LOT_CLASSES = frozenset({NORMAL, HOT, ENGINEERING})

FREEZE = "freeze"
UNFREEZE = "unfreeze"
_RULE_ACTIONS = frozenset({FREEZE, UNFREEZE})


@dataclass(frozen=True, slots=True)
class FreezeRule:
    """A versioned freeze or unfreeze directive scoped to lot classes."""

    rule_id: str
    action: str
    scopes: frozenset[str]
    priority: int
    version: int
    effective_at: int

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("rule id must be non-empty")
        if self.action not in _RULE_ACTIONS:
            raise ValueError("rule action must be freeze or unfreeze")
        if not self.scopes or not self.scopes <= LOT_CLASSES:
            raise ValueError("rule scopes must be a non-empty subset of lot classes")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.version <= 0:
            raise ValueError("rule version must be positive")


@dataclass(frozen=True, slots=True)
class FreezeAuthorization:
    """A time-boxed grant that overrides a freeze for matching lots."""

    authorization_id: str
    scopes: frozenset[str]
    window: TimeBox
    granted_by: str
    granted_at: int
    lot_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.authorization_id:
            raise ValueError("authorization id must be non-empty")
        if not self.granted_by:
            raise ValueError("authorization grantor must be non-empty")
        if not self.scopes or not self.scopes <= LOT_CLASSES:
            raise ValueError("authorization scopes must be a non-empty subset of lot classes")

    def covers(self, lot_class: str, lot_id: str) -> bool:
        return lot_class in self.scopes and (not self.lot_ids or lot_id in self.lot_ids)

    def active_at(self, instant: int) -> bool:
        return self.window.contains(instant)


def resolve_rule(rules: list[FreezeRule], lot_class: str, at: int) -> FreezeRule | None:
    """Pick the winning rule for a lot class: priority, then version, then effective time."""
    matching = [rule for rule in rules if rule.effective_at <= at and lot_class in rule.scopes]
    if not matching:
        return None
    return max(matching, key=lambda rule: (rule.priority, rule.version, rule.effective_at, rule.rule_id))


def _select_authorization(
    authorizations: list[FreezeAuthorization], lot_class: str, lot_id: str, at: int
) -> FreezeAuthorization | None:
    matching = [auth for auth in authorizations if auth.active_at(at) and auth.covers(lot_class, lot_id)]
    if not matching:
        return None
    return max(matching, key=lambda auth: (auth.granted_at, auth.authorization_id))


def freeze_decision(
    lot_class: str,
    lot_id: str,
    at: int,
    rules: list[FreezeRule],
    authorizations: list[FreezeAuthorization],
) -> Decision:
    if lot_class not in LOT_CLASSES:
        return deny("unknown_lot_class", lot_class=lot_class, lot_id=lot_id, evaluated_at=at)
    evidence = {"lot_class": lot_class, "lot_id": lot_id, "evaluated_at": at}
    rule = resolve_rule(rules, lot_class, at)
    if rule is None:
        return allow(**evidence, rule_id=None)
    evidence.update(
        rule_id=rule.rule_id,
        rule_action=rule.action,
        rule_priority=rule.priority,
        rule_version=rule.version,
        rule_effective_at=rule.effective_at,
    )
    if rule.action == UNFREEZE:
        return allow(**evidence)
    override = _select_authorization(authorizations, lot_class, lot_id, at)
    if override is not None:
        return allow(
            **evidence,
            override=True,
            authorization_id=override.authorization_id,
            granted_by=override.granted_by,
            authorization_window=(override.window.start, override.window.end),
        )
    expired = tuple(
        sorted(auth.authorization_id for auth in authorizations if auth.covers(lot_class, lot_id) and auth.window.end <= at)
    )
    return deny("dispatch_frozen", **evidence, expired_authorizations=expired)
