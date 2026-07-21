"""
本地实体读取与过滤服务
使用Neo4j替代Zep Cloud，接口与ZepEntityReader完全兼容
"""

import json
from typing import Dict, Any, List, Optional, Set

from ..utils.logger import get_logger
from ..utils.graph_db import run_query
from ..utils.temporal import fact_matches_temporal_filter
from .graph_models import EntityNode, FilteredEntities

logger = get_logger('mirofish.local_entity_reader')


class LocalEntityReader:
    """
    本地实体读取服务（Neo4j后端）
    接口与 ZepEntityReader 完全兼容
    """

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """获取图谱的所有节点"""
        logger.info(f"获取图谱 {graph_id} 的所有节点...")

        records = run_query(
            "MATCH (n:Entity {graph_id: $graph_id}) "
            "RETURN properties(n) AS props, labels(n) AS labels",
            {"graph_id": graph_id},
        )

        nodes_data = []
        for record in records:
            props = record["props"]
            labels = record["labels"]
            attrs = {}
            if props.get("attributes"):
                try:
                    attrs = json.loads(props["attributes"])
                except Exception:
                    attrs = {}
            nodes_data.append({
                "uuid": props.get("uuid", ""),
                "name": props.get("name", ""),
                "labels": labels,
                "summary": props.get("summary", ""),
                "attributes": attrs,
            })

        logger.info(f"共获取 {len(nodes_data)} 个节点")
        return nodes_data

    def get_all_edges(
        self,
        graph_id: str,
        *,
        include_historical: bool = False,
        as_of: Any = None,
        known_at: Any = None,
    ) -> List[Dict[str, Any]]:
        """获取图谱的所有边"""
        logger.info(f"获取图谱 {graph_id} 的所有边...")

        records = run_query(
            """
            MATCH (s:Entity {graph_id: $graph_id})-[r]->(t:Entity {graph_id: $graph_id})
            RETURN properties(r) AS props, type(r) AS rel_type,
                   s.uuid AS source_uuid, t.uuid AS target_uuid
            """,
            {"graph_id": graph_id},
        )

        edges_data = []
        for record in records:
            props = record["props"]
            edge_data = {
                "uuid": props.get("uuid", ""),
                "name": props.get("name", record["rel_type"]),
                "fact": props.get("fact", ""),
                "source_node_uuid": record["source_uuid"],
                "target_node_uuid": record["target_uuid"],
                "attributes": {},
                "valid_at": props.get("valid_at"),
                "invalid_at": props.get("invalid_at"),
                "expired_at": props.get("expired_at"),
                "created_at": props.get("created_at"),
                "reference_time": props.get("reference_time"),
                "round_num": props.get("round_num", -1),
                "episodes": props.get("episodes") or (
                    [props["episode_source"]] if props.get("episode_source") else []
                ),
            }
            if fact_matches_temporal_filter(
                edge_data,
                as_of=as_of,
                known_at=known_at,
                include_historical=include_historical,
            ):
                edges_data.append(edge_data)

        logger.info(f"共获取 {len(edges_data)} 条边")
        return edges_data

    def get_node_edges(
        self,
        node_uuid: str,
        *,
        include_historical: bool = False,
        as_of: Any = None,
        known_at: Any = None,
    ) -> List[Dict[str, Any]]:
        """获取指定节点的所有相关边"""
        try:
            records = run_query(
                """
                MATCH (n:Entity {uuid: $uuid})-[r]-(m:Entity)
                RETURN properties(r) AS props, type(r) AS rel_type,
                       startNode(r).uuid AS start_uuid,
                       endNode(r).uuid AS end_uuid
                """,
                {"uuid": node_uuid},
            )

            edges_data = []
            for record in records:
                props = record["props"]
                edge_data = {
                    "uuid": props.get("uuid", ""),
                    "name": props.get("name", record["rel_type"]),
                    "fact": props.get("fact", ""),
                    "source_node_uuid": record["start_uuid"],
                    "target_node_uuid": record["end_uuid"],
                    "attributes": {},
                    "valid_at": props.get("valid_at"),
                    "invalid_at": props.get("invalid_at"),
                    "expired_at": props.get("expired_at"),
                    "created_at": props.get("created_at"),
                    "reference_time": props.get("reference_time"),
                    "round_num": props.get("round_num", -1),
                    "episodes": props.get("episodes") or (
                        [props["episode_source"]] if props.get("episode_source") else []
                    ),
                }
                if fact_matches_temporal_filter(
                    edge_data,
                    as_of=as_of,
                    known_at=known_at,
                    include_historical=include_historical,
                ):
                    edges_data.append(edge_data)
            return edges_data
        except Exception as e:
            logger.warning(f"获取节点 {node_uuid} 的边失败: {str(e)}")
            return []

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> FilteredEntities:
        """筛选出符合预定义实体类型的节点"""
        logger.info(f"开始筛选图谱 {graph_id} 的实体...")

        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)
        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []
        node_map = {n["uuid"]: n for n in all_nodes}

        filtered_entities = []
        entity_types_found: Set[str] = set()

        for node in all_nodes:
            labels = node.get("labels", [])
            custom_labels = [l for l in labels if l not in ["Entity", "Node"]]

            if not custom_labels:
                continue

            if defined_entity_types:
                matching_labels = [l for l in custom_labels if l in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)

            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node["summary"],
                attributes=node["attributes"],
            )

            if enrich_with_edges:
                related_edges = []
                related_node_uuids: Set[str] = set()

                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])

                entity.related_edges = related_edges
                entity.related_nodes = [
                    {
                        "uuid": node_map[uid]["uuid"],
                        "name": node_map[uid]["name"],
                        "labels": node_map[uid]["labels"],
                        "summary": node_map[uid].get("summary", ""),
                    }
                    for uid in related_node_uuids
                    if uid in node_map
                ]

            filtered_entities.append(entity)

        logger.info(
            f"筛选完成: 总节点 {total_count}, 符合条件 {len(filtered_entities)}, "
            f"实体类型: {entity_types_found}"
        )
        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    def get_entity_with_context(
        self,
        graph_id: str,
        entity_uuid: str,
    ) -> Optional[EntityNode]:
        """获取单个实体及其完整上下文"""
        try:
            records = run_query(
                "MATCH (n:Entity {graph_id: $graph_id, uuid: $uuid}) "
                "RETURN properties(n) AS props, labels(n) AS labels",
                {"graph_id": graph_id, "uuid": entity_uuid},
            )
            if not records:
                return None

            props = records[0]["props"]
            labels = records[0]["labels"]

            attrs = {}
            if props.get("attributes"):
                try:
                    attrs = json.loads(props["attributes"])
                except Exception:
                    attrs = {}

            edges = self.get_node_edges(entity_uuid)
            all_nodes = self.get_all_nodes(graph_id)
            node_map = {n["uuid"]: n for n in all_nodes}

            related_edges = []
            related_node_uuids: Set[str] = set()
            for edge in edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])

            related_nodes = [
                {
                    "uuid": node_map[uid]["uuid"],
                    "name": node_map[uid]["name"],
                    "labels": node_map[uid]["labels"],
                    "summary": node_map[uid].get("summary", ""),
                }
                for uid in related_node_uuids
                if uid in node_map
            ]

            return EntityNode(
                uuid=props.get("uuid", ""),
                name=props.get("name", ""),
                labels=labels,
                summary=props.get("summary", ""),
                attributes=attrs,
                related_edges=related_edges,
                related_nodes=related_nodes,
            )
        except Exception as e:
            logger.error(f"获取实体 {entity_uuid} 失败: {str(e)}")
            return None

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
        enrich_with_edges: bool = True,
    ) -> List[EntityNode]:
        """获取指定类型的所有实体"""
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges,
        )
        return result.entities
