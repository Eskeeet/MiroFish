"""Bi-temporal fact and episode storage for the local context graph.

This module supplies the behavior that Zep/Graphiti previously provided:
episode provenance, event-time validity, transaction-time expiry, semantic
deduplication/contradiction resolution, and historical snapshots.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from ..utils.embedder import embed_batch
from ..utils.graph_db import create_relationship, run_query, run_write, validate_rel_type
from ..utils.logger import get_logger
from ..utils.temporal import (
    contradiction_invalidation,
    fact_matches_temporal_filter,
    intervals_overlap,
    normalize_fact,
    parse_timestamp,
    to_iso,
    utc_now_iso,
)
from ..utils.vector_store import update_fact_metadata, upsert_facts


logger = get_logger("mirofish.temporal_graph")


def _query_timestamp(value: Any, name: str):
    """Parse an optional query timestamp, rejecting malformed user input."""
    if value in (None, ""):
        return None
    parsed = parse_timestamp(value)
    if parsed is None:
        raise ValueError(f"{name} must be an ISO-8601 timestamp")
    return parsed


@dataclass
class TemporalIngestResult:
    created_edge_ids: List[str] = field(default_factory=list)
    reused_edge_ids: List[str] = field(default_factory=list)
    invalidated_edge_ids: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    skipped_count: int = 0

    @property
    def created_count(self) -> int:
        return len(self.created_edge_ids)


class TemporalGraphService:
    """Read and write the temporal portion of a local MiroFish graph."""

    @staticmethod
    def create_episode(
        graph_id: str,
        content: str,
        *,
        reference_time: Any = None,
        source: str = "text",
        source_description: str = "",
        name: str = "",
        round_num: int = -1,
        platform: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
        episode_uuid: Optional[str] = None,
    ) -> str:
        """Persist raw source data before deriving facts from it."""
        episode_uuid = episode_uuid or uuid.uuid4().hex
        created_at = utc_now_iso()
        valid_at = to_iso(reference_time, fallback=created_at)
        props = {
            "uuid": episode_uuid,
            "graph_id": graph_id,
            "name": name or f"episode_{episode_uuid[:8]}",
            "content": content,
            # ``text`` is retained for compatibility with the first local schema.
            "text": content,
            "source": source,
            "source_description": source_description,
            "created_at": created_at,
            "valid_at": valid_at,
            "round_num": int(round_num),
            "platform": platform,
            "episode_metadata": json.dumps(dict(metadata or {}), ensure_ascii=False),
            "processed": False,
            "entity_edges": [],
        }
        run_write(
            """
            MERGE (ep:Episode {uuid: $uuid})
            SET ep += $props
            """,
            {"uuid": episode_uuid, "props": props},
        )
        return episode_uuid

    @staticmethod
    def complete_episode(
        episode_uuid: str,
        *,
        edge_ids: Sequence[str],
        entity_ids: Sequence[str],
    ) -> None:
        """Mark an episode processed and attach fact/entity provenance."""
        unique_edges = list(dict.fromkeys(edge_ids))
        unique_entities = list(dict.fromkeys(entity_ids))
        run_write(
            """
            MATCH (ep:Episode {uuid: $episode_uuid})
            SET ep.processed = true, ep.entity_edges = $edge_ids
            WITH ep
            UNWIND $entity_ids AS entity_uuid
            MATCH (n:Entity {uuid: entity_uuid})
            MERGE (ep)-[:MENTIONS]->(n)
            """,
            {
                "episode_uuid": episode_uuid,
                "edge_ids": unique_edges,
                "entity_ids": unique_entities,
            },
        )

    @staticmethod
    def _candidate_edges(graph_id: str, entity_ids: Sequence[str]) -> List[Dict[str, Any]]:
        if not entity_ids:
            return []
        return run_query(
            """
            MATCH (s:Entity {graph_id: $graph_id})-[r]->(t:Entity {graph_id: $graph_id})
            WHERE r.expired_at IS NULL
              AND (s.uuid IN $entity_ids OR t.uuid IN $entity_ids)
            RETURN properties(r) AS props, type(r) AS rel_type,
                   s.uuid AS source_uuid, s.name AS source_name,
                   t.uuid AS target_uuid, t.name AS target_name
            LIMIT 500
            """,
            {"graph_id": graph_id, "entity_ids": list(entity_ids)},
        )

    @staticmethod
    def _resolve_with_llm(
        new_facts: Sequence[Dict[str, Any]],
        candidates: Sequence[Dict[str, Any]],
        llm: Any,
    ) -> Dict[int, Dict[str, Any]]:
        """Resolve semantic duplicates and contradictions in one LLM call."""
        if not new_facts or not candidates or llm is None:
            return {}

        candidate_payload = []
        valid_candidate_ids: Set[str] = set()
        candidate_by_id: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            props = candidate.get("props", {})
            edge_uuid = str(props.get("uuid", ""))
            if not edge_uuid:
                continue
            valid_candidate_ids.add(edge_uuid)
            candidate_by_id[edge_uuid] = candidate
            candidate_payload.append({
                "edge_uuid": edge_uuid,
                "source": candidate.get("source_name", ""),
                "target": candidate.get("target_name", ""),
                "relation_type": props.get("name", candidate.get("rel_type", "")),
                "fact": props.get("fact", ""),
                "valid_at": props.get("valid_at"),
                "invalid_at": props.get("invalid_at"),
            })

        new_payload = [
            {
                "new_index": index,
                "source": fact["source_name"],
                "target": fact["target_name"],
                "relation_type": fact["rel_type"],
                "fact": fact["fact"],
                "valid_at": fact.get("valid_at"),
                "invalid_at": fact.get("invalid_at"),
            }
            for index, fact in enumerate(new_facts)
        ]
        system_prompt = """You resolve facts in a temporal knowledge graph.
For every NEW_FACT, identify:
1. A semantic duplicate: the same claim with the same material details. Numeric,
   date, location, status, or qualifier differences are not duplicates. The
   same relationship recurring in a non-overlapping interval is not a duplicate.
2. Contradictions: existing claims that cannot be true during the same time
   interval. Separate events at different times and merely related facts are not
   contradictions.

Return only JSON:
{"resolutions": [{"new_index": 0, "duplicate_edge_uuid": null,
"contradicted_edge_uuids": []}]}
Use only edge UUIDs from EXISTING_FACTS and include one resolution per new fact."""
        try:
            response = llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "EXISTING_FACTS:\n"
                            + json.dumps(candidate_payload, ensure_ascii=False)
                            + "\n\nNEW_FACTS:\n"
                            + json.dumps(new_payload, ensure_ascii=False)
                        ),
                    },
                ],
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning(f"时态事实冲突解析失败，仅使用精确去重: {exc}")
            return {}

        resolved: Dict[int, Dict[str, Any]] = {}
        for item in response.get("resolutions", []) if isinstance(response, dict) else []:
            try:
                new_index = int(item.get("new_index"))
            except (TypeError, ValueError):
                continue
            if new_index < 0 or new_index >= len(new_facts):
                continue
            duplicate_value = item.get("duplicate_edge_uuid")
            duplicate = str(duplicate_value) if duplicate_value is not None else None
            new_fact = new_facts[new_index]
            duplicate_candidate = candidate_by_id.get(str(duplicate))
            if (
                duplicate not in valid_candidate_ids
                or duplicate_candidate is None
                or duplicate_candidate.get("source_uuid") != new_fact["source_uuid"]
                or duplicate_candidate.get("target_uuid") != new_fact["target_uuid"]
                or not intervals_overlap(
                    duplicate_candidate.get("props", {}).get("valid_at"),
                    duplicate_candidate.get("props", {}).get("invalid_at"),
                    new_fact.get("valid_at"),
                    new_fact.get("invalid_at"),
                )
            ):
                duplicate = None
            contradicted = [
                str(edge_id)
                for raw_edge_id in item.get("contradicted_edge_uuids", [])
                if (edge_id := str(raw_edge_id))
                if edge_id in valid_candidate_ids
                and edge_id != duplicate
                and (
                    candidate_by_id[edge_id].get("source_uuid")
                    in (new_fact["source_uuid"], new_fact["target_uuid"])
                    or candidate_by_id[edge_id].get("target_uuid")
                    in (new_fact["source_uuid"], new_fact["target_uuid"])
                )
            ]
            resolved[new_index] = {
                "duplicate_edge_uuid": duplicate,
                "contradicted_edge_uuids": list(dict.fromkeys(contradicted)),
            }
        return resolved

    @classmethod
    def ingest_relationships(
        cls,
        graph_id: str,
        relationships: Iterable[Mapping[str, Any]],
        *,
        episode_uuid: str,
        reference_time: Any,
        round_num: int = -1,
        valid_edge_types: Optional[Set[str]] = None,
        llm: Any = None,
    ) -> TemporalIngestResult:
        """Resolve and persist extracted relationships with temporal history."""
        result = TemporalIngestResult()
        node_records = run_query(
            "MATCH (n:Entity {graph_id: $g}) RETURN n.uuid AS uuid, n.name AS name",
            {"g": graph_id},
        )
        names = {
            str(record.get("name", "")).casefold(): {
                "uuid": record.get("uuid", ""),
                "name": record.get("name", ""),
            }
            for record in node_records
            if record.get("name") and record.get("uuid")
        }

        specs: List[Dict[str, Any]] = []
        seen_new: Set[tuple] = set()
        reference_iso = to_iso(reference_time, fallback=utc_now_iso())
        for relationship in relationships:
            source_name = str(
                relationship.get("source") or relationship.get("source_entity_name") or ""
            ).strip()
            target_name = str(
                relationship.get("target") or relationship.get("target_entity_name") or ""
            ).strip()
            rel_type = str(
                relationship.get("type") or relationship.get("relation_type") or "RELATED_TO"
            ).strip()
            fact = str(relationship.get("fact") or "").strip()
            source = names.get(source_name.casefold())
            target = names.get(target_name.casefold())
            if (
                not source
                or not target
                or source["uuid"] == target["uuid"]
                or not fact
                or (valid_edge_types and rel_type not in valid_edge_types)
            ):
                result.skipped_count += 1
                continue

            invalid_at = to_iso(relationship.get("invalid_at"))
            valid_at = to_iso(relationship.get("valid_at"))
            if valid_at is None and invalid_at is None:
                valid_at = reference_iso
            valid_time = parse_timestamp(valid_at)
            invalid_time = parse_timestamp(invalid_at)
            if valid_time is not None and invalid_time is not None and invalid_time <= valid_time:
                logger.warning(
                    "跳过无效事实时间区间: valid_at=%s invalid_at=%s fact=%s",
                    valid_at,
                    invalid_at,
                    fact,
                )
                result.skipped_count += 1
                continue
            try:
                fact_round_num = int(relationship.get("round_num", round_num))
            except (TypeError, ValueError):
                fact_round_num = int(round_num)
            key = (
                source["uuid"],
                target["uuid"],
                validate_rel_type(rel_type),
                normalize_fact(fact),
                valid_at,
                invalid_at,
                fact_round_num,
            )
            if key in seen_new:
                continue
            seen_new.add(key)
            specs.append({
                "source_uuid": source["uuid"],
                "source_name": source["name"],
                "target_uuid": target["uuid"],
                "target_name": target["name"],
                "rel_type": rel_type,
                "safe_type": validate_rel_type(rel_type),
                "fact": fact,
                "valid_at": valid_at,
                "invalid_at": invalid_at,
                "round_num": fact_round_num,
            })

        if not specs:
            cls.complete_episode(episode_uuid, edge_ids=[], entity_ids=[])
            return result

        entity_ids = list({
            entity_id
            for spec in specs
            for entity_id in (spec["source_uuid"], spec["target_uuid"])
        })
        candidates = cls._candidate_edges(graph_id, entity_ids)
        candidate_by_id = {
            str(candidate.get("props", {}).get("uuid")): candidate
            for candidate in candidates
            if candidate.get("props", {}).get("uuid")
        }

        # Deterministic fast path for exact facts and endpoints.
        resolutions: Dict[int, Dict[str, Any]] = {}
        unresolved: List[Dict[str, Any]] = []
        unresolved_indices: List[int] = []
        for index, spec in enumerate(specs):
            duplicate = next((
                candidate
                for candidate in candidates
                if candidate.get("source_uuid") == spec["source_uuid"]
                and candidate.get("target_uuid") == spec["target_uuid"]
                and normalize_fact(candidate.get("props", {}).get("fact", ""))
                == normalize_fact(spec["fact"])
                and intervals_overlap(
                    candidate.get("props", {}).get("valid_at"),
                    candidate.get("props", {}).get("invalid_at"),
                    spec.get("valid_at"),
                    spec.get("invalid_at"),
                )
            ), None)
            if duplicate:
                resolutions[index] = {
                    "duplicate_edge_uuid": duplicate["props"]["uuid"],
                    "contradicted_edge_uuids": [],
                }
            else:
                unresolved.append(spec)
                unresolved_indices.append(index)

        semantic_resolutions = cls._resolve_with_llm(unresolved, candidates, llm)
        for local_index, resolution in semantic_resolutions.items():
            resolutions[unresolved_indices[local_index]] = resolution

        created_at = utc_now_iso()
        facts_to_embed: List[Dict[str, Any]] = []
        all_edge_ids: List[str] = []

        for index, spec in enumerate(specs):
            resolution = resolutions.get(index, {})
            duplicate_id = resolution.get("duplicate_edge_uuid")
            contradicted_ids = resolution.get("contradicted_edge_uuids", [])
            if duplicate_id:
                run_write(
                    """
                    MATCH ()-[r {uuid: $edge_uuid}]->()
                    SET r.episodes = CASE
                        WHEN $episode_uuid IN coalesce(r.episodes, []) THEN coalesce(r.episodes, [])
                        ELSE coalesce(r.episodes, []) + $episode_uuid
                    END
                    """,
                    {"edge_uuid": duplicate_id, "episode_uuid": episode_uuid},
                )
                update_fact_metadata(duplicate_id, {"episodes": episode_uuid})
                result.reused_edge_ids.append(duplicate_id)
                result.facts.append(spec["fact"])
                all_edge_ids.append(duplicate_id)
                continue

            new_invalid_at = spec.get("invalid_at")
            # Match Graphiti's transaction-time behavior: facts extracted as
            # already ended are stored as history and immediately marked
            # inactive in the graph's current transaction-time view.
            new_expired_at = created_at if new_invalid_at else None
            for contradicted_id in contradicted_ids:
                candidate = candidate_by_id.get(contradicted_id)
                if not candidate:
                    continue
                old_props = candidate.get("props", {})
                action, boundary = contradiction_invalidation(
                    old_props.get("valid_at"),
                    old_props.get("invalid_at"),
                    spec.get("valid_at"),
                    spec.get("invalid_at"),
                )
                if action == "existing":
                    boundary = boundary or spec.get("valid_at") or reference_iso
                    run_write(
                        """
                        MATCH ()-[r {uuid: $edge_uuid}]->()
                        SET r.invalid_at = $invalid_at, r.expired_at = $expired_at
                        """,
                        {
                            "edge_uuid": contradicted_id,
                            "invalid_at": boundary,
                            "expired_at": created_at,
                        },
                    )
                    update_fact_metadata(contradicted_id, {
                        "invalid_at": boundary,
                        "expired_at": created_at,
                        "is_current": False,
                    })
                    result.invalidated_edge_ids.append(contradicted_id)
                elif action == "new":
                    new_invalid_at = boundary
                    new_expired_at = created_at

            edge_uuid = uuid.uuid4().hex
            edge_props = {
                "uuid": edge_uuid,
                "graph_id": graph_id,
                "name": spec["rel_type"],
                "fact": spec["fact"],
                "created_at": created_at,
                "valid_at": spec.get("valid_at"),
                "invalid_at": new_invalid_at,
                "expired_at": new_expired_at,
                "reference_time": reference_iso,
                "round_num": spec["round_num"],
                "episode_source": episode_uuid,
                "episodes": [episode_uuid],
            }
            create_relationship(
                spec["source_uuid"],
                spec["target_uuid"],
                spec["safe_type"],
                edge_props,
            )
            facts_to_embed.append({
                "id": edge_uuid,
                "text": spec["fact"],
                "metadata": {
                    "graph_id": graph_id,
                    "edge_uuid": edge_uuid,
                    "source_uuid": spec["source_uuid"],
                    "source_name": spec["source_name"],
                    "target_uuid": spec["target_uuid"],
                    "target_name": spec["target_name"],
                    "rel_type": spec["rel_type"],
                    "created_at": created_at,
                    "reference_time": reference_iso or "",
                    "valid_at": spec.get("valid_at") or "",
                    "invalid_at": new_invalid_at or "",
                    "expired_at": new_expired_at or "",
                    "round_num": spec["round_num"],
                    "episode_uuid": episode_uuid,
                    "episodes": episode_uuid,
                    "is_current": new_expired_at is None,
                    "doc_type": "edge_fact",
                },
            })
            result.created_edge_ids.append(edge_uuid)
            result.facts.append(spec["fact"])
            all_edge_ids.append(edge_uuid)

        if facts_to_embed:
            embeddings = embed_batch([item["text"] for item in facts_to_embed])
            for item, embedding in zip(facts_to_embed, embeddings):
                item["embedding"] = embedding
            upsert_facts(facts_to_embed)

        cls.complete_episode(
            episode_uuid,
            edge_ids=all_edge_ids,
            entity_ids=entity_ids,
        )
        return result

    @staticmethod
    def _edge_record_to_dict(record: Mapping[str, Any]) -> Dict[str, Any]:
        props = dict(record.get("props", {}) or {})
        return {
            "uuid": props.get("uuid", ""),
            "name": props.get("name", record.get("rel_type", "")),
            "fact": props.get("fact", ""),
            "source_node_uuid": record.get("source_uuid", ""),
            "target_node_uuid": record.get("target_uuid", ""),
            "source_node_name": record.get("source_name", ""),
            "target_node_name": record.get("target_name", ""),
            "created_at": props.get("created_at"),
            "reference_time": props.get("reference_time"),
            "valid_at": props.get("valid_at"),
            "invalid_at": props.get("invalid_at"),
            "expired_at": props.get("expired_at"),
            "round_num": props.get("round_num", -1),
            "episodes": props.get("episodes") or (
                [props["episode_source"]] if props.get("episode_source") else []
            ),
        }

    @classmethod
    def get_timeline(
        cls,
        graph_id: str,
        *,
        entity_uuid: Optional[str] = None,
        relation_type: Optional[str] = None,
        start_time: Any = None,
        end_time: Any = None,
        as_of: Any = None,
        known_at: Any = None,
        round_from: Optional[int] = None,
        round_to: Optional[int] = None,
        include_historical: bool = True,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        start_time = _query_timestamp(start_time, "start_time")
        end_time = _query_timestamp(end_time, "end_time")
        as_of = _query_timestamp(as_of, "as_of")
        known_at = _query_timestamp(known_at, "known_at")
        if start_time is not None and end_time is not None and start_time >= end_time:
            raise ValueError("start_time must be earlier than end_time")
        if round_from is not None and round_to is not None and round_from > round_to:
            raise ValueError("round_from must be less than or equal to round_to")
        records = run_query(
            """
            MATCH (s:Entity {graph_id: $graph_id})-[r]->(t:Entity {graph_id: $graph_id})
            WHERE ($entity_uuid IS NULL OR s.uuid = $entity_uuid OR t.uuid = $entity_uuid)
              AND ($relation_type IS NULL OR r.name = $relation_type OR type(r) = $relation_type)
            RETURN properties(r) AS props, type(r) AS rel_type,
                   s.uuid AS source_uuid, s.name AS source_name,
                   t.uuid AS target_uuid, t.name AS target_name
            """,
            {
                "graph_id": graph_id,
                "entity_uuid": entity_uuid,
                "relation_type": relation_type,
            },
        )
        facts = [cls._edge_record_to_dict(record) for record in records]
        facts = [
            fact
            for fact in facts
            if fact_matches_temporal_filter(
                fact,
                as_of=as_of,
                known_at=known_at,
                start_time=start_time,
                end_time=end_time,
                round_from=round_from,
                round_to=round_to,
                include_historical=include_historical,
            )
        ]
        facts.sort(
            key=lambda item: parse_timestamp(item.get("valid_at"))
            or parse_timestamp(item.get("created_at"))
            or parse_timestamp("1970-01-01T00:00:00Z"),
            reverse=True,
        )
        return facts[: max(1, min(int(limit), 1000))]

    @staticmethod
    def list_episodes(
        graph_id: str,
        *,
        start_time: Any = None,
        end_time: Any = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        start = _query_timestamp(start_time, "start_time")
        end = _query_timestamp(end_time, "end_time")
        if start is not None and end is not None and start >= end:
            raise ValueError("start_time must be earlier than end_time")
        records = run_query(
            """
            MATCH (ep:Episode {graph_id: $graph_id})
            RETURN properties(ep) AS props
            ORDER BY ep.valid_at DESC, ep.created_at DESC
            LIMIT $limit
            """,
            {"graph_id": graph_id, "limit": max(1, min(int(limit), 1000))},
        )
        episodes = []
        for record in records:
            props = dict(record.get("props", {}) or {})
            valid_at = parse_timestamp(props.get("valid_at"))
            if start is not None and valid_at is not None and valid_at < start:
                continue
            if end is not None and valid_at is not None and valid_at >= end:
                continue
            if props.get("episode_metadata"):
                try:
                    props["episode_metadata"] = json.loads(props["episode_metadata"])
                except (TypeError, json.JSONDecodeError):
                    pass
            episodes.append(props)
        return episodes

    @staticmethod
    def migrate_legacy_temporal_data(graph_id: Optional[str] = None) -> None:
        """Backfill transaction/provenance fields created by older local builds."""
        now = utc_now_iso()
        run_write(
            """
            MATCH ()-[r]->()
            WHERE r.graph_id IS NOT NULL
              AND ($graph_id IS NULL OR r.graph_id = $graph_id)
            SET r.created_at = coalesce(r.created_at, r.valid_at, $now),
                r.valid_at = coalesce(r.valid_at, r.created_at, $now),
                r.reference_time = coalesce(r.reference_time, r.valid_at, r.created_at, $now),
                r.expired_at = CASE
                    WHEN r.invalid_at IS NOT NULL THEN coalesce(r.expired_at, r.invalid_at, $now)
                    ELSE r.expired_at
                END,
                r.episodes = CASE
                    WHEN r.episodes IS NOT NULL THEN r.episodes
                    WHEN r.episode_source IS NOT NULL THEN [r.episode_source]
                    ELSE []
                END
            """,
            {"graph_id": graph_id, "now": now},
        )
