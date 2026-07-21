"""Temporal helpers for the local context graph.

The local graph follows the same bi-temporal distinction used by Zep/Graphiti:

* ``valid_at`` / ``invalid_at`` describe when a fact was true in the world.
* ``created_at`` / ``expired_at`` describe when MiroFish learned or superseded it.

Timestamps are stored as UTC ISO-8601 strings so they work consistently in
Neo4j, Chroma metadata, JSON responses, and tests.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Tuple


UTC = timezone.utc


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse a supported timestamp value and normalize it to UTC.

    Invalid or empty values return ``None`` instead of raising. This is
    intentional: LLM-extracted temporal bounds are optional and must never
    make graph ingestion fail.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day)
    elif isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        candidate = re.sub(r"\s+UTC$", "+00:00", candidate, flags=re.IGNORECASE)
        if candidate.endswith(("Z", "z")):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def to_iso(value: Any, *, fallback: Any = None) -> Optional[str]:
    """Return a canonical UTC ISO-8601 timestamp or a parsed fallback."""
    parsed = parse_timestamp(value)
    if parsed is None:
        parsed = parse_timestamp(fallback)
    return parsed.isoformat() if parsed is not None else None


def utc_now_iso() -> str:
    """Return the current UTC transaction time."""
    return datetime.now(UTC).isoformat()


def latest_timestamp(values: Iterable[Any], *, fallback: Any = None) -> Optional[str]:
    """Return the latest valid timestamp in ``values`` as UTC ISO-8601."""
    parsed = [item for value in values if (item := parse_timestamp(value)) is not None]
    if parsed:
        return max(parsed).isoformat()
    return to_iso(fallback)


def normalize_fact(text: str) -> str:
    """Normalize fact text for deterministic exact-duplicate detection."""
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def interval_contains(
    valid_at: Any,
    invalid_at: Any,
    point_in_time: Any,
    *,
    include_unknown: bool = True,
) -> bool:
    """Return whether an event-time validity interval contains a point.

    Intervals are half-open: ``[valid_at, invalid_at)``. Unknown starts are
    treated as unbounded only when ``include_unknown`` is true.
    """
    point = parse_timestamp(point_in_time)
    if point is None:
        return False
    start = parse_timestamp(valid_at)
    end = parse_timestamp(invalid_at)
    if start is None and not include_unknown:
        return False
    return (start is None or start <= point) and (end is None or point < end)


def intervals_overlap(
    valid_at: Any,
    invalid_at: Any,
    range_start: Any,
    range_end: Any,
) -> bool:
    """Return whether a fact interval overlaps a requested time range."""
    fact_start = parse_timestamp(valid_at)
    fact_end = parse_timestamp(invalid_at)
    query_start = parse_timestamp(range_start)
    query_end = parse_timestamp(range_end)
    if query_start is not None and fact_end is not None and fact_end <= query_start:
        return False
    if query_end is not None and fact_start is not None and fact_start >= query_end:
        return False
    return True


def transaction_visible(created_at: Any, expired_at: Any, known_at: Any) -> bool:
    """Return whether a fact was present in the graph at transaction time."""
    point = parse_timestamp(known_at)
    if point is None:
        return False
    created = parse_timestamp(created_at)
    expired = parse_timestamp(expired_at)
    return (created is None or created <= point) and (expired is None or point < expired)


def fact_matches_temporal_filter(
    fact: Mapping[str, Any],
    *,
    as_of: Any = None,
    known_at: Any = None,
    start_time: Any = None,
    end_time: Any = None,
    round_from: Optional[int] = None,
    round_to: Optional[int] = None,
    include_historical: bool = False,
    include_unknown: bool = True,
    now: Any = None,
) -> bool:
    """Apply event-time, transaction-time, range, and round filters to a fact."""
    round_num = fact.get("round_num")
    try:
        fact_round = int(round_num) if round_num is not None else None
    except (TypeError, ValueError):
        fact_round = None
    if round_from is not None and fact_round is not None and fact_round < round_from:
        return False
    if round_to is not None and fact_round is not None and fact_round > round_to:
        return False

    if start_time is not None or end_time is not None:
        if not intervals_overlap(
            fact.get("valid_at"), fact.get("invalid_at"), start_time, end_time
        ):
            return False

    if as_of is not None and not interval_contains(
        fact.get("valid_at"),
        fact.get("invalid_at"),
        as_of,
        include_unknown=include_unknown,
    ):
        return False

    if known_at is not None and not transaction_visible(
        fact.get("created_at"), fact.get("expired_at"), known_at
    ):
        return False

    if as_of is None and known_at is None and not include_historical:
        current = now or utc_now_iso()
        if not interval_contains(
            fact.get("valid_at"),
            fact.get("invalid_at"),
            current,
            include_unknown=include_unknown,
        ):
            return False

    return True


def contradiction_invalidation(
    existing_valid_at: Any,
    existing_invalid_at: Any,
    new_valid_at: Any,
    new_invalid_at: Any = None,
) -> Tuple[str, Optional[str]]:
    """Choose which contradictory fact to invalidate.

    Newer event-time information supersedes older information. Backfilled facts
    are retained as history and end when the already-known newer fact begins.
    Non-overlapping intervals need no invalidation.

    Returns ``("existing", timestamp)``, ``("new", timestamp)``, or
    ``("none", None)``.
    """
    if not intervals_overlap(
        existing_valid_at, existing_invalid_at, new_valid_at, new_invalid_at
    ):
        return "none", None

    existing_start = parse_timestamp(existing_valid_at)
    new_start = parse_timestamp(new_valid_at)
    if existing_start is not None and new_start is not None and existing_start > new_start:
        return "new", existing_start.isoformat()

    boundary = new_start or parse_timestamp(new_invalid_at)
    return "existing", boundary.isoformat() if boundary is not None else None
