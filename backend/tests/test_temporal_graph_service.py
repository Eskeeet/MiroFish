"""Behavior tests for temporal ingestion using in-memory dependency fakes."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


BACKEND = Path(__file__).parents[1]
APP = BACKEND / "app"


def _package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


_package("app", APP)
_package("app.services", APP / "services")
_package("app.utils", APP / "utils")

temporal_spec = importlib.util.spec_from_file_location(
    "app.utils.temporal", APP / "utils" / "temporal.py"
)
temporal_module = importlib.util.module_from_spec(temporal_spec)
sys.modules["app.utils.temporal"] = temporal_module
assert temporal_spec.loader is not None
temporal_spec.loader.exec_module(temporal_module)


class FakeLogger:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


for module_name, members in {
    "app.utils.embedder": {"embed_batch": lambda texts: [[0.1] for _ in texts]},
    "app.utils.logger": {"get_logger": lambda _name: FakeLogger()},
}.items():
    module = types.ModuleType(module_name)
    for name, value in members.items():
        setattr(module, name, value)
    sys.modules[module_name] = module

graph_db = types.ModuleType("app.utils.graph_db")
graph_db.run_query = lambda *_args, **_kwargs: []
graph_db.run_write = lambda *_args, **_kwargs: []
graph_db.create_relationship = lambda *_args, **_kwargs: []
graph_db.validate_rel_type = lambda value: value.upper()
sys.modules["app.utils.graph_db"] = graph_db

vector_store = types.ModuleType("app.utils.vector_store")
vector_store.update_fact_metadata = lambda *_args, **_kwargs: None
vector_store.upsert_facts = lambda *_args, **_kwargs: None
sys.modules["app.utils.vector_store"] = vector_store

service_spec = importlib.util.spec_from_file_location(
    "app.services.temporal_graph", APP / "services" / "temporal_graph.py"
)
service_module = importlib.util.module_from_spec(service_spec)
sys.modules["app.services.temporal_graph"] = service_module
assert service_spec.loader is not None
service_spec.loader.exec_module(service_module)


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def chat_json(self, **_kwargs):
        return self.response


class TemporalGraphServiceTests(unittest.TestCase):
    def setUp(self):
        self.writes = []
        self.created = []
        self.metadata_updates = []
        self.upserts = []
        self.nodes = [
            {"uuid": "alice-id", "name": "Alice"},
            {"uuid": "acme-id", "name": "Acme"},
        ]
        self.candidates = []

        def fake_query(cypher, _params=None):
            if "RETURN n.uuid AS uuid, n.name AS name" in cypher:
                return list(self.nodes)
            if "RETURN properties(r) AS props" in cypher:
                return list(self.candidates)
            return []

        service_module.run_query = fake_query
        service_module.run_write = lambda cypher, params=None: self.writes.append((cypher, params)) or []
        service_module.create_relationship = (
            lambda source, target, rel_type, props: self.created.append(
                (source, target, rel_type, props)
            )
        )
        service_module.update_fact_metadata = (
            lambda edge_id, updates: self.metadata_updates.append((edge_id, updates))
        )
        service_module.upsert_facts = lambda facts: self.upserts.extend(facts)
        service_module.embed_batch = lambda texts: [[0.1, 0.2] for _ in texts]

    def test_exact_duplicate_reuses_edge_and_adds_episode_provenance(self):
        self.candidates = [{
            "props": {
                "uuid": "existing-edge",
                "fact": "Alice works at Acme",
                "name": "WORKS_AT",
                "valid_at": "2026-01-01T00:00:00Z",
            },
            "rel_type": "WORKS_AT",
            "source_uuid": "alice-id",
            "source_name": "Alice",
            "target_uuid": "acme-id",
            "target_name": "Acme",
        }]

        result = service_module.TemporalGraphService.ingest_relationships(
            "graph-1",
            [{
                "source": "Alice",
                "target": "Acme",
                "type": "WORKS_AT",
                "fact": "  Alice works at Acme  ",
            }],
            episode_uuid="episode-2",
            reference_time="2026-02-01T00:00:00Z",
            valid_edge_types={"WORKS_AT"},
        )

        self.assertEqual(result.reused_edge_ids, ["existing-edge"])
        self.assertEqual(result.created_count, 0)
        self.assertFalse(self.created)
        self.assertTrue(any("r.episodes" in cypher for cypher, _ in self.writes))

    def test_newer_contradiction_expires_old_edge_and_creates_new_edge(self):
        self.candidates = [{
            "props": {
                "uuid": "old-edge",
                "fact": "Alice works at Acme as an engineer",
                "name": "WORKS_AT",
                "valid_at": "2026-01-01T00:00:00Z",
                "invalid_at": None,
            },
            "rel_type": "WORKS_AT",
            "source_uuid": "alice-id",
            "source_name": "Alice",
            "target_uuid": "acme-id",
            "target_name": "Acme",
        }]
        llm = FakeLLM({
            "resolutions": [{
                "new_index": 0,
                "duplicate_edge_uuid": None,
                "contradicted_edge_uuids": ["old-edge"],
            }]
        })

        result = service_module.TemporalGraphService.ingest_relationships(
            "graph-1",
            [{
                "source": "Alice",
                "target": "Acme",
                "type": "WORKS_AT",
                "fact": "Alice works at Acme as a director",
                "valid_at": "2026-03-01T00:00:00Z",
            }],
            episode_uuid="episode-3",
            reference_time="2026-03-01T00:00:00Z",
            round_num=8,
            valid_edge_types={"WORKS_AT"},
            llm=llm,
        )

        self.assertEqual(result.invalidated_edge_ids, ["old-edge"])
        self.assertEqual(result.created_count, 1)
        self.assertEqual(self.created[0][3]["round_num"], 8)
        self.assertEqual(self.created[0][3]["episodes"], ["episode-3"])
        self.assertTrue(any(edge_id == "old-edge" for edge_id, _ in self.metadata_updates))
        self.assertEqual(len(self.upserts), 1)

    def test_non_overlapping_recurrence_creates_a_new_edge(self):
        self.candidates = [{
            "props": {
                "uuid": "past-edge",
                "fact": "Alice works at Acme",
                "name": "WORKS_AT",
                "valid_at": "2024-01-01T00:00:00Z",
                "invalid_at": "2025-01-01T00:00:00Z",
            },
            "rel_type": "WORKS_AT",
            "source_uuid": "alice-id",
            "source_name": "Alice",
            "target_uuid": "acme-id",
            "target_name": "Acme",
        }]

        result = service_module.TemporalGraphService.ingest_relationships(
            "graph-1",
            [{
                "source": "Alice",
                "target": "Acme",
                "type": "WORKS_AT",
                "fact": "Alice works at Acme",
                "valid_at": "2026-01-01T00:00:00Z",
                "round_num": 11,
            }],
            episode_uuid="episode-recurrence",
            reference_time="2026-01-01T00:00:00Z",
            round_num=99,
            valid_edge_types={"WORKS_AT"},
        )

        self.assertEqual(result.created_count, 1)
        self.assertFalse(result.reused_edge_ids)
        self.assertEqual(self.created[0][3]["round_num"], 11)

    def test_explicit_event_end_is_stored_as_inactive_history(self):
        service_module.TemporalGraphService.ingest_relationships(
            "graph-1",
            [{
                "source": "Alice",
                "target": "Acme",
                "type": "WORKS_AT",
                "fact": "Alice worked at Acme during 2025",
                "valid_at": "2025-01-01T00:00:00Z",
                "invalid_at": "2026-01-01T00:00:00Z",
            }],
            episode_uuid="episode-history",
            reference_time="2026-02-01T00:00:00Z",
            valid_edge_types={"WORKS_AT"},
        )

        self.assertIsNotNone(self.created[0][3]["expired_at"])

    def test_invalid_event_interval_is_skipped(self):
        result = service_module.TemporalGraphService.ingest_relationships(
            "graph-1",
            [{
                "source": "Alice",
                "target": "Acme",
                "type": "WORKS_AT",
                "fact": "Alice worked at Acme",
                "valid_at": "2026-02-01T00:00:00Z",
                "invalid_at": "2026-01-01T00:00:00Z",
            }],
            episode_uuid="episode-invalid-interval",
            reference_time="2026-02-01T00:00:00Z",
            valid_edge_types={"WORKS_AT"},
        )

        self.assertEqual(result.skipped_count, 1)
        self.assertFalse(self.created)

    def test_timeline_rejects_invalid_query_ranges(self):
        with self.assertRaisesRegex(ValueError, "ISO-8601"):
            service_module.TemporalGraphService.get_timeline(
                "graph-1", start_time="not-a-date"
            )
        with self.assertRaisesRegex(ValueError, "earlier"):
            service_module.TemporalGraphService.get_timeline(
                "graph-1",
                start_time="2026-02-01T00:00:00Z",
                end_time="2026-01-01T00:00:00Z",
            )


if __name__ == "__main__":
    unittest.main()
