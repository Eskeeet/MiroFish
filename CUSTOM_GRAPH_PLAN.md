# Replacing Zep with a Custom Local Knowledge Graph

## Overview

Replace Zep Cloud with a fully local stack: **Neo4j** (graph DB) + **ChromaDB** (vector store) + **sentence-transformers** (embeddings) + **LLM-driven entity extraction**. All five Zep roles are replaced while preserving the existing public interfaces via aliases, so callers in `api/` require zero changes.

---

## Technology Stack

| Component | Tool | Why |
|---|---|---|
| Graph DB | Neo4j Community (Docker) | `neo4j` driver already in venv; Cypher handles temporal properties; ACID transactions |
| Vector store | ChromaDB (embedded) | No server needed; stores fact metadata alongside embeddings |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Already installed via `sentence-transformers`; handles Chinese + English; CPU-only |
| Entity extraction | Existing LLM API | Called explicitly with ontology as context — replaces Zep's opaque server-side extraction |

**New deps:** `chromadb`, `rank-bm25`
**Remove:** `zep-cloud`

---

## Zep Role → Replacement Mapping

### 1. Graph Building (`graph_builder.py` → `local_graph_builder.py`)

Zep received chunked text + ontology and auto-extracted entities/relationships server-side.

**New pipeline:**
- Chunk text (same 500-char / 50-overlap, no change)
- Per batch of 3 chunks, call LLM with structured prompt including ontology schema:
  ```json
  {
    "entities": [{"name": "...", "type": "...", "attributes": {}, "summary": "..."}],
    "relationships": [{"source": "...", "target": "...", "type": "...", "fact": "...", "confidence": "explicit|inferred"}]
  }
  ```
- Filter out `confidence: "inferred"` relationships
- Deduplicate entities: MERGE by name in Neo4j (case-insensitive); if cosine similarity > 0.85 with existing node name, treat as same entity
- Write nodes/edges to Neo4j, embed fact text → upsert to ChromaDB
- After all chunks, one final LLM pass to generate `node.summary` per entity

### 2. Entity Reading (`zep_entity_reader.py` → `local_entity_reader.py`)

**New implementation:** Pure Cypher queries.
```cypher
MATCH (n:Entity:<Type> {graph_id: $g})
OPTIONAL MATCH (n)-[r]->(m)
RETURN n, collect(r), collect(m)
```
Same output dataclasses (`EntityNode`, `FilteredEntities`) — no callers change.

### 3. Graph Memory During Simulation (`zep_graph_memory_updater.py` → `local_graph_memory_updater.py`)

Queue/thread/batch architecture is **identical**. Only `_send_batch_activities()` changes:
- **Lazy extraction (recommended):** Store raw `Episode` nodes in Neo4j immediately (durable), run LLM extraction at end-of-round boundaries → fewer LLM calls
- Temporal conflict: if a new fact contradicts an active relationship, `SET r.invalid_at = now` and create a new relationship

### 4. Search Tools (`zep_tools.py` → `local_tools.py`)

| Tool | New Implementation |
|---|---|
| `InsightForge` | LLM decomposes query → parallel ChromaDB sub-queries → 2-hop Neo4j traversal for top entities |
| `PanoramaSearch` | `MATCH (n {graph_id}) RETURN n` + edges split by `invalid_at IS NULL` (active) vs not (historical) |
| `Interview` | Already uses OASIS runner — only entity reader alias needed |
| `quick_search` | Direct ChromaDB `.query()` with cosine similarity |
| Base `search_graph` | ChromaDB semantic (60%) + BM25 (40%) hybrid, filtered by `graph_id` metadata |

### 5. Temporal Tracking

Explicit Neo4j relationship properties — written by your code, not inferred by Zep:
```
valid_at:    datetime when fact was created
invalid_at:  set explicitly when a conflicting fact is extracted
expired_at:  manually retired facts
round_num:   which simulation round produced this fact
```

---

## Data Models

### Neo4j Node Schema
```
(:Entity:<EntityType> {
    uuid:        STRING,   // primary key
    graph_id:    STRING,   // project scope
    name:        STRING,
    summary:     STRING,   // LLM-generated, updated over time
    attributes:  STRING,   // JSON-encoded dict
    created_at:  DATETIME,
    updated_at:  DATETIME
})
```

### Neo4j Relationship Schema
```
-[:RELATIONSHIP_TYPE {
    uuid:           STRING,
    graph_id:       STRING,
    fact:           STRING,   // human-readable: "Alice works at MIT"
    name:           STRING,
    attributes:     STRING,   // JSON-encoded dict
    created_at:     DATETIME,
    valid_at:       DATETIME,
    invalid_at:     DATETIME, // null = still valid
    expired_at:     DATETIME,
    round_num:      INTEGER,  // simulation round, -1 for static graph
    episode_source: STRING
}]->
```

### ChromaDB Document Schema
One collection per graph (`graph_{graph_id}`), or single collection with `graph_id` metadata filter (preferred for scale):
```json
{
    "id": "<edge_uuid>",
    "document": "<fact text>",
    "metadata": {
        "graph_id":    "...",
        "edge_uuid":   "...",
        "source_name": "Alice",
        "target_name": "MIT",
        "rel_type":    "WORKS_FOR",
        "valid_at":    "2026-03-14T10:00:00",
        "invalid_at":  "",
        "round_num":   -1,
        "doc_type":    "edge_fact"
    }
}
```
Node summaries are also indexed with `doc_type: "node_summary"`.

### Graph Metadata File
Stored at `backend/uploads/projects/<project_id>/graph_metadata.json`:
```json
{
    "graph_id":    "mirofish_xxx",
    "name":        "...",
    "ontology":    {},
    "graph_backend": "local",
    "created_at":  "...",
    "node_count":  0,
    "edge_count":  0,
    "entity_types": []
}
```

---

## Files to Create

| File | Role |
|---|---|
| `backend/app/utils/graph_db.py` | Neo4j driver singleton + `run_query(cypher, params)` |
| `backend/app/utils/vector_store.py` | ChromaDB persistent client, `upsert_facts()`, `semantic_search()` |
| `backend/app/utils/embedder.py` | SentenceTransformer singleton (warm at startup), `embed(text)` |
| `backend/app/services/local_graph_builder.py` | LLM NER + Neo4j writes + ChromaDB upserts |
| `backend/app/services/local_entity_reader.py` | Cypher-based reader, same public interface as `ZepEntityReader` |
| `backend/app/services/local_graph_memory_updater.py` | Same queue architecture, local backend |
| `backend/app/services/local_tools.py` | ChromaDB + Neo4j search tools |

---

## Files to Modify

| File | Change |
|---|---|
| `services/zep_entity_reader.py` | Add `ZepEntityReader = LocalEntityReader` alias |
| `services/graph_builder.py` | Add `GraphBuilderService = LocalGraphBuilderService` alias |
| `services/zep_graph_memory_updater.py` | Alias to `LocalGraphMemoryUpdater` / `LocalGraphMemoryManager` |
| `services/zep_tools.py` | Alias to `LocalToolsService` |
| `services/oasis_profile_generator.py` | Remove `from zep_cloud.client import Zep` import + `self.zep_client` |
| `config.py` | Add `NEO4J_URI/USER/PASSWORD`, `CHROMA_PERSIST_DIR`; remove `ZEP_API_KEY` |
| `docker-compose.yml` | Add Neo4j `5-community` service |
| `backend/requirements.txt` | Add `chromadb`, `rank-bm25`; remove `zep-cloud` |
| `.env.example` | Replace `ZEP_API_KEY` with Neo4j vars |

---

## Implementation Order

1. **Config + utilities** — `config.py`, `graph_db.py`, `vector_store.py`, `embedder.py`
2. **Entity reader** — `local_entity_reader.py` → alias in `zep_entity_reader.py`
3. **Graph builder** — `local_graph_builder.py` → alias in `graph_builder.py`
4. **Memory updater** — `local_graph_memory_updater.py` → alias in `zep_graph_memory_updater.py`
5. **Search tools** — `local_tools.py` → alias in `zep_tools.py`
6. **Cleanup** — fix `oasis_profile_generator.py`, update docker-compose, remove zep-cloud
7. **Test** — end-to-end graph build → simulation → report

---

## Gotchas

- **Existing Zep projects:** Add `graph_backend` field to project metadata. Show error for `"zep"` projects — they must be rebuilt.
- **Neo4j dynamic rel types:** Cypher doesn't allow parameterized `[:$type]`. Validate rel type strings against the ontology before string-interpolating into queries.
- **LLM extraction quality:** Use `temperature=0.1`, JSON mode, and instruct: *"only extract facts explicitly stated in the text."*
- **Model cold start:** Initialize `SentenceTransformer` singleton at app startup in `__init__.py`, not lazily.
- **ChromaDB scale:** Use a single collection with `graph_id` as metadata filter rather than one collection per graph.
- **`invalid_at` conflict during search:** Neo4j snapshot isolation means a mid-update search sees relationships as active until commit — correct behaviour.
