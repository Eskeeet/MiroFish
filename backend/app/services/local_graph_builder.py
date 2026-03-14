"""
本地图谱构建服务
使用 LLM + Neo4j + ChromaDB 替代 Zep Cloud
"""

import os
import uuid
import json
import time
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from ..utils.graph_db import run_query, run_write, create_relationship, ensure_constraints, validate_rel_type
from ..utils.vector_store import upsert_facts, delete_graph_facts
from ..utils.embedder import embed_batch
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .text_processor import TextProcessor

logger = get_logger('mirofish.local_graph_builder')

# 图谱元数据目录
_GRAPHS_DIR = os.path.join(os.path.dirname(__file__), '../../uploads/graphs')


def _graphs_dir() -> str:
    os.makedirs(_GRAPHS_DIR, exist_ok=True)
    return _GRAPHS_DIR


def _meta_path(graph_id: str) -> str:
    return os.path.join(_graphs_dir(), f"{graph_id}.json")


def _load_meta(graph_id: str) -> Optional[Dict[str, Any]]:
    path = _meta_path(graph_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_meta(graph_id: str, meta: Dict[str, Any]):
    with open(_meta_path(graph_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Entity Extractor
# ---------------------------------------------------------------------------

class EntityExtractor:
    """LLM驱动的实体与关系提取器"""

    SYSTEM_PROMPT = """你是一个专业的知识图谱构建助手。
你的任务是从给定文本中提取实体和关系，严格遵循提供的本体（Ontology）schema。

规则：
1. 只提取文本中明确陈述的事实，不要推断或假设
2. 实体类型必须严格匹配本体中定义的 entity_types
3. 关系类型必须严格匹配本体中定义的 edge_types
4. 如果关系类型不在本体中，使用最接近的已有类型或忽略
5. 每个关系必须有对应的 source 和 target 实体
6. fact 字段必须是人类可读的完整陈述句

以 JSON 格式输出，结构为：
{
  "entities": [
    {"name": "实体名称", "type": "实体类型", "attributes": {}, "summary": "一句话描述"}
  ],
  "relationships": [
    {"source": "源实体名称", "target": "目标实体名称", "type": "关系类型", "fact": "完整的事实陈述句"}
  ]
}"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm = llm_client or LLMClient()

    def extract(self, text: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """从文本中提取实体和关系"""
        entity_types = [e["name"] for e in ontology.get("entity_types", [])]
        edge_types = [e["name"] for e in ontology.get("edge_types", [])]

        prompt = f"""本体定义：
实体类型: {json.dumps(entity_types, ensure_ascii=False)}
关系类型: {json.dumps(edge_types, ensure_ascii=False)}

待处理文本：
{text}

请提取上述文本中的实体和关系，严格按照本体类型。"""

        try:
            result = self._llm.chat_json(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            if isinstance(result, dict):
                return result
        except Exception as e:
            logger.warning(f"实体提取失败: {e}")
        return {"entities": [], "relationships": []}

    def generate_summaries(self, entities: List[Dict[str, Any]], all_facts: List[str]) -> Dict[str, str]:
        """
        为实体生成摘要。
        Returns: {entity_name: summary}
        """
        if not entities or not all_facts:
            return {}

        facts_text = "\n".join(f"- {f}" for f in all_facts[:50])
        entity_names = [e["name"] for e in entities]

        prompt = f"""以下是一组事实：
{facts_text}

请为以下每个实体生成一句简洁的中文摘要（20-50字），描述其在这些事实中的角色：
{json.dumps(entity_names, ensure_ascii=False)}

以 JSON 输出，格式为 {{实体名称: 摘要}}"""

        try:
            result = self._llm.chat_json(
                messages=[
                    {"role": "system", "content": "你是一个知识图谱摘要生成助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            if isinstance(result, dict):
                return result
        except Exception as e:
            logger.warning(f"摘要生成失败: {e}")
        return {}


# ---------------------------------------------------------------------------
# Graph Builder Service
# ---------------------------------------------------------------------------

class LocalGraphBuilderService:
    """
    本地图谱构建服务
    使用 LLM 提取实体/关系，存储到 Neo4j，向量索引到 ChromaDB
    """

    def __init__(self):
        self.task_manager = TaskManager()
        self._extractor = EntityExtractor()

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3,
    ) -> str:
        """异步构建图谱，返回任务ID"""
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            },
        )

        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size),
            daemon=True,
        )
        thread.start()
        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
    ):
        """图谱构建工作线程"""
        try:
            self.task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=5, message="开始构建图谱...")

            # 1. 确保 Neo4j 索引存在
            ensure_constraints()

            # 2. 创建图谱
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(task_id, progress=10, message=f"图谱已创建: {graph_id}")

            # 3. 存储本体
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(task_id, progress=15, message="本体已设置")

            # 4. 分块
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(task_id, progress=20, message=f"文本已分割为 {total_chunks} 个块")

            # 5. 分批提取并写入
            all_facts: List[str] = []
            batch_ids = self.add_text_batches(
                graph_id,
                chunks,
                ontology,
                batch_size,
                progress_callback=lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.55),  # 20-75%
                    message=msg,
                ),
                collected_facts=all_facts,
            )

            # 6. 生成节点摘要
            self.task_manager.update_task(task_id, progress=78, message="生成实体摘要...")
            self._generate_node_summaries(graph_id, all_facts)

            # 7. 获取图谱信息
            self.task_manager.update_task(task_id, progress=92, message="获取图谱信息...")
            graph_info = self._get_graph_info(graph_id)

            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
            })

        except Exception as e:
            import traceback
            self.task_manager.fail_task(task_id, f"{str(e)}\n{traceback.format_exc()}")

    def create_graph(self, name: str) -> str:
        """创建图谱，返回 graph_id"""
        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"
        meta = {
            "graph_id": graph_id,
            "name": name,
            "graph_backend": "local",
            "ontology": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "node_count": 0,
            "edge_count": 0,
            "entity_types": [],
        }
        _save_meta(graph_id, meta)
        logger.info(f"图谱已创建: {graph_id}")
        return graph_id

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """将本体存储到元数据文件"""
        meta = _load_meta(graph_id) or {}
        meta["ontology"] = ontology
        meta["entity_types"] = [e["name"] for e in ontology.get("entity_types", [])]
        _save_meta(graph_id, meta)

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        ontology: Dict[str, Any],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
        collected_facts: Optional[List[str]] = None,
    ) -> List[str]:
        """
        分批提取实体/关系并写入 Neo4j + ChromaDB。
        Returns: batch ID 列表（与 Zep episode_uuid 列表对应）
        """
        meta = _load_meta(graph_id) or {}
        valid_entity_types = {e["name"] for e in ontology.get("entity_types", [])}
        valid_edge_types = {e["name"] for e in ontology.get("edge_types", [])}

        total_chunks = len(chunks)
        batch_ids = []
        total_nodes_created = 0
        total_edges_created = 0

        # 缓存已有实体名称（用于去重）
        existing_names: Dict[str, str] = {}  # name_lower -> uuid
        existing_records = run_query(
            "MATCH (n:Entity {graph_id: $g}) RETURN n.uuid AS uuid, n.name AS name",
            {"g": graph_id},
        )
        for rec in existing_records:
            if rec["name"]:
                existing_names[rec["name"].lower()] = rec["uuid"]

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            if progress_callback:
                progress = (i + len(batch_chunks)) / total_chunks
                progress_callback(
                    f"处理第 {batch_num}/{total_batches} 批数据 ({len(batch_chunks)} 块)...",
                    progress,
                )

            batch_id = f"batch_{graph_id}_{batch_num}"
            facts_to_embed: List[Dict[str, Any]] = []

            for chunk in batch_chunks:
                extracted = self._extractor.extract(chunk, ontology)
                entities = extracted.get("entities", [])
                relationships = extracted.get("relationships", [])

                # --- 写入实体节点 ---
                chunk_entity_map: Dict[str, str] = {}  # name -> uuid
                for ent in entities:
                    ent_name = (ent.get("name") or "").strip()
                    ent_type = (ent.get("type") or "").strip()
                    if not ent_name or ent_type not in valid_entity_types:
                        continue

                    name_key = ent_name.lower()
                    if name_key in existing_names:
                        ent_uuid = existing_names[name_key]
                    else:
                        ent_uuid = uuid.uuid4().hex
                        existing_names[name_key] = ent_uuid
                        total_nodes_created += 1

                    attrs_json = json.dumps(ent.get("attributes", {}), ensure_ascii=False)
                    cypher = (
                        f"MERGE (n:Entity:{ent_type} {{uuid: $uuid}}) "
                        "ON CREATE SET n.created_at = $updated_at "
                        "SET n.graph_id = $graph_id, "
                        "n.name = $name, "
                        "n.summary = $summary, "
                        "n.attributes = $attributes, "
                        "n.updated_at = $updated_at "
                        "RETURN n.uuid AS uuid"
                    )
                    run_write(cypher, {
                        "uuid": ent_uuid,
                        "graph_id": graph_id,
                        "name": ent_name,
                        "summary": (ent.get("summary") or ""),
                        "attributes": attrs_json,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    })
                    chunk_entity_map[ent_name] = ent_uuid

                # --- 写入关系边 ---
                for rel in relationships:
                    src_name = (rel.get("source") or "").strip()
                    tgt_name = (rel.get("target") or "").strip()
                    rel_type = (rel.get("type") or "").strip()
                    fact = (rel.get("fact") or "").strip()

                    if not src_name or not tgt_name or not fact:
                        continue
                    if rel_type not in valid_edge_types:
                        continue

                    src_uuid = chunk_entity_map.get(src_name) or existing_names.get(src_name.lower())
                    tgt_uuid = chunk_entity_map.get(tgt_name) or existing_names.get(tgt_name.lower())
                    if not src_uuid or not tgt_uuid:
                        continue

                    rel_uuid = uuid.uuid4().hex
                    now = datetime.now(timezone.utc).isoformat()
                    safe_type = validate_rel_type(rel_type)
                    create_relationship(src_uuid, tgt_uuid, safe_type, {
                        "uuid": rel_uuid,
                        "graph_id": graph_id,
                        "name": rel_type,
                        "fact": fact,
                        "valid_at": now,
                        "invalid_at": None,
                        "expired_at": None,
                        "round_num": -1,
                        "episode_source": batch_id,
                    })
                    total_edges_created += 1

                    facts_to_embed.append({
                        "id": rel_uuid,
                        "text": fact,
                        "metadata": {
                            "graph_id": graph_id,
                            "edge_uuid": rel_uuid,
                            "source_name": src_name,
                            "target_name": tgt_name,
                            "rel_type": rel_type,
                            "valid_at": now,
                            "invalid_at": "",
                            "round_num": -1,
                            "doc_type": "edge_fact",
                        },
                    })
                    if collected_facts is not None:
                        collected_facts.append(fact)

            # --- 批量嵌入并写入 ChromaDB ---
            if facts_to_embed:
                texts = [f["text"] for f in facts_to_embed]
                embeddings = embed_batch(texts)
                for item, emb in zip(facts_to_embed, embeddings):
                    item["embedding"] = emb
                upsert_facts(facts_to_embed)

            batch_ids.append(batch_id)

        # 更新元数据统计
        meta = _load_meta(graph_id) or {}
        meta["node_count"] = meta.get("node_count", 0) + total_nodes_created
        meta["edge_count"] = meta.get("edge_count", 0) + total_edges_created
        _save_meta(graph_id, meta)

        logger.info(f"图谱 {graph_id} 构建完成: {total_nodes_created} 节点, {total_edges_created} 边")
        return batch_ids

    def _generate_node_summaries(self, graph_id: str, all_facts: List[str]):
        """为图谱中的实体节点生成摘要（如已有摘要则跳过）"""
        records = run_query(
            "MATCH (n:Entity {graph_id: $g}) WHERE n.summary = '' OR n.summary IS NULL "
            "RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels",
            {"g": graph_id},
        )
        if not records:
            return

        entities = [{"name": r["name"], "uuid": r["uuid"]} for r in records]
        summaries = self._extractor.generate_summaries(entities, all_facts)

        for ent in entities:
            summary = summaries.get(ent["name"], "")
            if summary:
                run_write(
                    "MATCH (n:Entity {uuid: $uuid}) SET n.summary = $summary",
                    {"uuid": ent["uuid"], "summary": summary},
                )

    def _get_graph_info(self, graph_id: str) -> "GraphInfo":
        from .graph_builder import GraphInfo
        node_count = run_query(
            "MATCH (n:Entity {graph_id: $g}) RETURN count(n) AS cnt",
            {"g": graph_id},
        )[0]["cnt"]
        edge_count = run_query(
            "MATCH (:Entity {graph_id: $g})-[r]->(:Entity {graph_id: $g}) RETURN count(r) AS cnt",
            {"g": graph_id},
        )[0]["cnt"]
        label_records = run_query(
            "MATCH (n:Entity {graph_id: $g}) RETURN labels(n) AS labels",
            {"g": graph_id},
        )
        entity_types = set()
        for rec in label_records:
            for lbl in rec["labels"]:
                if lbl not in ("Entity", "Node"):
                    entity_types.add(lbl)

        meta = _load_meta(graph_id) or {}
        meta["node_count"] = node_count
        meta["edge_count"] = edge_count
        meta["entity_types"] = list(entity_types)
        _save_meta(graph_id, meta)

        return GraphInfo(
            graph_id=graph_id,
            node_count=node_count,
            edge_count=edge_count,
            entity_types=list(entity_types),
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """获取完整图谱数据（供前端可视化）"""
        node_records = run_query(
            "MATCH (n:Entity {graph_id: $g}) "
            "RETURN properties(n) AS props, labels(n) AS labels",
            {"g": graph_id},
        )
        nodes_data = []
        node_map: Dict[str, str] = {}
        for rec in node_records:
            props = rec["props"]
            labels = rec["labels"]
            node_map[props.get("uuid", "")] = props.get("name", "")
            attrs = {}
            try:
                attrs = json.loads(props.get("attributes", "{}"))
            except Exception:
                pass
            nodes_data.append({
                "uuid": props.get("uuid", ""),
                "name": props.get("name", ""),
                "labels": labels,
                "summary": props.get("summary", ""),
                "attributes": attrs,
                "created_at": props.get("created_at"),
            })

        edge_records = run_query(
            "MATCH (s:Entity {graph_id: $g})-[r]->(t:Entity {graph_id: $g}) "
            "RETURN properties(r) AS props, type(r) AS rel_type, "
            "s.uuid AS source_uuid, t.uuid AS target_uuid",
            {"g": graph_id},
        )
        edges_data = []
        for rec in edge_records:
            props = rec["props"]
            edges_data.append({
                "uuid": props.get("uuid", ""),
                "name": props.get("name", rec["rel_type"]),
                "fact": props.get("fact", ""),
                "fact_type": props.get("name", rec["rel_type"]),
                "source_node_uuid": rec["source_uuid"],
                "target_node_uuid": rec["target_uuid"],
                "source_node_name": node_map.get(rec["source_uuid"], ""),
                "target_node_name": node_map.get(rec["target_uuid"], ""),
                "attributes": {},
                "created_at": props.get("created_at"),
                "valid_at": props.get("valid_at"),
                "invalid_at": props.get("invalid_at"),
                "expired_at": props.get("expired_at"),
                "episodes": [],
            })

        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    def delete_graph(self, graph_id: str):
        """删除图谱（Neo4j + ChromaDB + 元数据文件）"""
        run_write(
            "MATCH (n:Entity {graph_id: $g}) DETACH DELETE n",
            {"g": graph_id},
        )
        delete_graph_facts(graph_id)
        meta_path = _meta_path(graph_id)
        if os.path.exists(meta_path):
            os.remove(meta_path)
        logger.info(f"图谱 {graph_id} 已删除")
