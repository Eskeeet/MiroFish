"""Unit tests for bi-temporal interval behavior (stdlib-only)."""

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "app" / "utils" / "temporal.py"
SPEC = importlib.util.spec_from_file_location("mirofish_temporal", MODULE_PATH)
temporal = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(temporal)


class TimestampTests(unittest.TestCase):
    def test_normalizes_z_and_naive_timestamps_to_utc(self):
        self.assertEqual(
            temporal.to_iso("2026-07-21T12:30:00Z"),
            "2026-07-21T12:30:00+00:00",
        )
        self.assertEqual(
            temporal.to_iso("2026-07-21T12:30:00"),
            "2026-07-21T12:30:00+00:00",
        )

    def test_invalid_timestamp_is_optional(self):
        self.assertIsNone(temporal.parse_timestamp("not-a-date"))
        self.assertIsNone(temporal.to_iso(None))


class ValidityIntervalTests(unittest.TestCase):
    def test_validity_interval_is_half_open(self):
        start = "2026-01-01T00:00:00Z"
        end = "2026-02-01T00:00:00Z"
        self.assertTrue(temporal.interval_contains(start, end, start))
        self.assertTrue(temporal.interval_contains(start, end, "2026-01-31T23:59:59Z"))
        self.assertFalse(temporal.interval_contains(start, end, end))

    def test_unknown_start_can_be_strict_or_unbounded(self):
        self.assertTrue(
            temporal.interval_contains(None, None, "2026-01-01T00:00:00Z")
        )
        self.assertFalse(
            temporal.interval_contains(
                None,
                None,
                "2026-01-01T00:00:00Z",
                include_unknown=False,
            )
        )

    def test_range_overlap_excludes_touching_intervals(self):
        self.assertFalse(
            temporal.intervals_overlap(
                "2026-01-01T00:00:00Z",
                "2026-02-01T00:00:00Z",
                "2026-02-01T00:00:00Z",
                "2026-03-01T00:00:00Z",
            )
        )


class ContradictionTests(unittest.TestCase):
    def test_newer_fact_invalidates_existing_fact(self):
        action, boundary = temporal.contradiction_invalidation(
            "2026-01-01T00:00:00Z",
            None,
            "2026-02-01T00:00:00Z",
        )
        self.assertEqual(action, "existing")
        self.assertEqual(boundary, "2026-02-01T00:00:00+00:00")

    def test_backfilled_fact_ends_when_newer_known_fact_begins(self):
        action, boundary = temporal.contradiction_invalidation(
            "2026-03-01T00:00:00Z",
            None,
            "2026-01-01T00:00:00Z",
        )
        self.assertEqual(action, "new")
        self.assertEqual(boundary, "2026-03-01T00:00:00+00:00")

    def test_non_overlapping_facts_do_not_invalidate_each_other(self):
        action, boundary = temporal.contradiction_invalidation(
            "2026-01-01T00:00:00Z",
            "2026-02-01T00:00:00Z",
            "2026-03-01T00:00:00Z",
        )
        self.assertEqual((action, boundary), ("none", None))


class BiTemporalFilterTests(unittest.TestCase):
    FACT = {
        "valid_at": "2026-01-01T00:00:00Z",
        "invalid_at": "2026-02-01T00:00:00Z",
        "created_at": "2026-01-10T00:00:00Z",
        "expired_at": "2026-02-10T00:00:00Z",
        "round_num": 4,
    }

    def test_fact_can_be_queried_by_event_and_transaction_time(self):
        self.assertTrue(
            temporal.fact_matches_temporal_filter(
                self.FACT,
                as_of="2026-01-20T00:00:00Z",
                known_at="2026-01-20T00:00:00Z",
            )
        )
        self.assertFalse(
            temporal.fact_matches_temporal_filter(
                self.FACT,
                as_of="2026-01-20T00:00:00Z",
                known_at="2026-02-20T00:00:00Z",
            )
        )

    def test_round_filters_form_a_time_series_slice(self):
        self.assertTrue(
            temporal.fact_matches_temporal_filter(
                self.FACT,
                round_from=3,
                round_to=5,
                include_historical=True,
            )
        )
        self.assertFalse(
            temporal.fact_matches_temporal_filter(
                self.FACT,
                round_from=5,
                include_historical=True,
            )
        )

    def test_malformed_legacy_round_does_not_break_filtering(self):
        fact = {**self.FACT, "round_num": "unknown"}
        self.assertTrue(
            temporal.fact_matches_temporal_filter(
                fact,
                round_from=3,
                include_historical=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
