# Local Graph Architecture (Zep Replacement)

## Status

The runtime Zep dependency has been removed. MiroFish now uses:

- **Neo4j** for entities, relationships, raw episodes, and temporal history
- **ChromaDB** for fact and entity-summary vector retrieval
- **Sentence Transformers** for multilingual local embeddings
- The configured **OpenAI-compatible LLM** for extraction, semantic
  deduplication, and contradiction classification

Legacy module and class names remain as dependency-free compatibility shims so
older callers do not break. They resolve to local implementations and do not
import `zep-cloud`.

## Responsibility mapping

| Former responsibility | Local implementation |
|---|---|
| Graph construction | `services/local_graph_builder.py` |
| Entity reading | `services/local_entity_reader.py` |
| Simulation memory updates | `services/local_graph_memory_updater.py` |
| Retrieval/report tools | `services/local_tools.py` |
| Episode, dedupe, contradiction, and time logic | `services/temporal_graph.py` |
| Timestamp and interval primitives | `utils/temporal.py` |
| Graph persistence | `utils/graph_db.py` |
| Vector persistence | `utils/vector_store.py` |

## Temporal model

Facts are bi-temporal:

| Property | Meaning |
|---|---|
| `valid_at` | Event time when the fact began to be true |
| `invalid_at` | Event time when it stopped being true |
| `created_at` | Transaction time when MiroFish learned it |
| `expired_at` | Transaction time when the graph learned it was no longer current or it was superseded |
| `reference_time` | Source episode time used to resolve relative dates |
| `round_num` | Simulation round that produced this particular fact |
| `episodes` | IDs of all raw episodes supporting the fact |

Intervals are half-open: `[valid_at, invalid_at)`. Matching Graphiti, a fact
extracted with an explicit event-time end is retained as history and marked
inactive in the transaction-time view at ingestion.

### Ingestion rules

1. Store the raw input as an `Episode` before extraction.
2. Extract `valid_at`, `invalid_at`, and `round_num` with each relationship.
3. Reuse exact or semantic duplicates only when their intervals overlap, then
   append the new episode ID.
4. Use the LLM to identify semantic contradictions among relevant facts.
5. If a newer event contradicts an older one, close the older fact at the new
   fact's `valid_at` and set its transaction `expired_at`.
6. If an older event arrives out of order, retain it as backfilled history and
   close it at the already-known newer fact's `valid_at`.
7. Synchronize invalidation metadata to ChromaDB; Neo4j remains the source of
   truth during retrieval.

The simulation activity prompt carries an exact timestamp and round on every
source line. Extracted facts therefore preserve per-activity rounds even when a
write batch spans multiple rounds.

## Provenance schema

```text
(:Episode {
  uuid, graph_id, content, source, source_description,
  created_at, valid_at, round_num, platform, episode_metadata,
  processed, entity_edges
})-[:MENTIONS]->(:Entity)

(:Entity)-[:RELATION {
  uuid, graph_id, name, fact,
  created_at, expired_at,
  valid_at, invalid_at, reference_time,
  round_num, episode_source, episodes
}]->(:Entity)
```

`entity_edges` contains the derived relationship UUIDs. `episodes` supports
many-to-one provenance when repeated evidence resolves to an existing fact.

## Retrieval behavior

Semantic search first retrieves a larger candidate set from ChromaDB, hydrates
those edge IDs from Neo4j, and then applies event-time, transaction-time, range,
and round filters. This prevents stale vector metadata from returning an
invalidated fact as current.

Available HTTP reads:

| Endpoint | Important parameters |
|---|---|
| `/api/graph/timeline/<graph_id>` | `start_time`, `end_time`, `entity_uuid`, `relation_type`, `round_from`, `round_to` |
| `/api/graph/snapshot/<graph_id>` | required `as_of`, optional `known_at` |
| `/api/graph/episodes/<graph_id>` | `start_time`, `end_time`, `limit` |
| `/api/graph/data/<graph_id>` | `as_of`, `known_at`, `include_historical` |

ReportAgent exposes the same capability through `timeline_search`.

## Compatibility and migration

- `graph_builder.py`, `zep_entity_reader.py`, `zep_graph_memory_updater.py`, and
  `zep_tools.py` are thin local aliases for older imports.
- Startup creates Episode constraints/indexes and backfills temporal fields on
  legacy local relationships.
- Both `pyproject.toml` and `requirements.txt` declare all direct local graph
  dependencies. `zep-cloud` is absent from both and from `uv.lock`.
- Existing local graph IDs remain readable. Legacy relationships without
  temporal fields receive conservative timestamps and provenance defaults.

## Verification

Temporal unit tests live under `backend/tests/` and cover timestamp
normalization, half-open intervals, event/transaction-time snapshots, round
slices, exact duplicate provenance, recurrence, contradiction invalidation,
backfilled facts, and explicit historical intervals.
