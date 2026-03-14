"""
本地图谱检索工具服务
使用 Neo4j + ChromaDB + BM25 替代 Zep Cloud 的语义搜索
接口与 ZepToolsService 完全兼容
"""

import json
import time
import re
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.graph_db import run_query
from ..utils.vector_store import semantic_search
from ..utils.embedder import embed

from dataclasses import dataclass, field

logger = get_logger('mirofish.local_tools')


@dataclass
class SearchResult:
    """搜索结果"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {"facts": self.facts, "edges": self.edges, "nodes": self.nodes,
                "query": self.query, "total_count": self.total_count}

    def to_text(self) -> str:
        text_parts = [f"搜索查询: {self.query}", f"找到 {self.total_count} 条相关信息"]
        if self.facts:
            text_parts.append("\n### 相关事实:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")
        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """节点信息"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"uuid": self.uuid, "name": self.name, "labels": self.labels,
                "summary": self.summary, "attributes": self.attributes}

    def to_text(self) -> str:
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "未知类型")
        return f"实体: {self.name} (类型: {entity_type})\n摘要: {self.summary}"


@dataclass
class EdgeInfo:
    """边信息"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"uuid": self.uuid, "name": self.name, "fact": self.fact,
                "source_node_uuid": self.source_node_uuid, "target_node_uuid": self.target_node_uuid,
                "source_node_name": self.source_node_name, "target_node_name": self.target_node_name,
                "created_at": self.created_at, "valid_at": self.valid_at,
                "invalid_at": self.invalid_at, "expired_at": self.expired_at}

    def to_text(self, include_temporal: bool = False) -> str:
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"关系: {source} --[{self.name}]--> {target}\n事实: {self.fact}"
        if include_temporal:
            valid_at = self.valid_at or "未知"
            invalid_at = self.invalid_at or "至今"
            base_text += f"\n时效: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (已过期: {self.expired_at})"
        return base_text

    @property
    def is_expired(self) -> bool:
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """深度洞察检索结果"""
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "simulation_requirement": self.simulation_requirement,
                "sub_queries": self.sub_queries, "semantic_facts": self.semantic_facts,
                "entity_insights": self.entity_insights, "relationship_chains": self.relationship_chains,
                "total_facts": self.total_facts, "total_entities": self.total_entities,
                "total_relationships": self.total_relationships}

    def to_text(self) -> str:
        text_parts = [f"## 未来预测深度分析", f"分析问题: {self.query}",
                      f"预测场景: {self.simulation_requirement}",
                      f"\n### 预测数据统计",
                      f"- 相关预测事实: {self.total_facts}条",
                      f"- 涉及实体: {self.total_entities}个",
                      f"- 关系链: {self.total_relationships}条"]
        if self.sub_queries:
            text_parts.append(f"\n### 分析的子问题")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")
        if self.semantic_facts:
            text_parts.append(f"\n### 【关键事实】(请在报告中引用这些原文)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        if self.entity_insights:
            text_parts.append(f"\n### 【核心实体】")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', '未知')}** ({entity.get('type', '实体')})")
                if entity.get('summary'):
                    text_parts.append(f"  摘要: \"{entity.get('summary')}\"")
        if self.relationship_chains:
            text_parts.append(f"\n### 【关系链】")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")
        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """广度搜索结果"""
    query: str
    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)
    historical_facts: List[str] = field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "all_nodes": [n.to_dict() for n in self.all_nodes],
                "all_edges": [e.to_dict() for e in self.all_edges],
                "active_facts": self.active_facts, "historical_facts": self.historical_facts,
                "total_nodes": self.total_nodes, "total_edges": self.total_edges,
                "active_count": self.active_count, "historical_count": self.historical_count}

    def to_text(self) -> str:
        text_parts = [f"## 广度搜索结果（未来全景视图）", f"查询: {self.query}",
                      f"\n### 统计信息", f"- 总节点数: {self.total_nodes}",
                      f"- 总边数: {self.total_edges}",
                      f"- 当前有效事实: {self.active_count}条",
                      f"- 历史/过期事实: {self.historical_count}条"]
        if self.active_facts:
            text_parts.append(f"\n### 【当前有效事实】(模拟结果原文)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        if self.historical_facts:
            text_parts.append(f"\n### 【历史/过期事实】(演变过程记录)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        if self.all_nodes:
            text_parts.append(f"\n### 【涉及实体】")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "实体")
                text_parts.append(f"- **{node.name}** ({entity_type})")
        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """单个Agent的采访结果"""
    agent_name: str
    agent_role: str
    agent_bio: str
    question: str
    response: str
    key_quotes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"agent_name": self.agent_name, "agent_role": self.agent_role,
                "agent_bio": self.agent_bio, "question": self.question,
                "response": self.response, "key_quotes": self.key_quotes}

    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        text += f"_简介: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**关键引言:**\n"
            for quote in self.key_quotes:
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '').strip()
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                skip = any(f'问题{d}' in clean_quote for d in '123456789')
                if skip:
                    continue
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('。', 80)
                    clean_quote = clean_quote[:dot_pos + 1] if dot_pos > 0 else clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """采访结果"""
    interview_topic: str
    interview_questions: List[str]
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    interviews: List[AgentInterview] = field(default_factory=list)
    selection_reasoning: str = ""
    summary: str = ""
    total_agents: int = 0
    interviewed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"interview_topic": self.interview_topic,
                "interview_questions": self.interview_questions,
                "selected_agents": self.selected_agents,
                "interviews": [i.to_dict() for i in self.interviews],
                "selection_reasoning": self.selection_reasoning,
                "summary": self.summary, "total_agents": self.total_agents,
                "interviewed_count": self.interviewed_count}

    def to_text(self) -> str:
        text_parts = ["## 深度采访报告",
                      f"**采访主题:** {self.interview_topic}",
                      f"**采访人数:** {self.interviewed_count} / {self.total_agents} 位模拟Agent",
                      "\n### 采访对象选择理由",
                      self.selection_reasoning or "（自动选择）",
                      "\n---", "\n### 采访实录"]
        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### 采访 #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("（无采访记录）\n\n---")
        text_parts.append("\n### 采访摘要与核心观点")
        text_parts.append(self.summary or "（无摘要）")
        return "\n".join(text_parts)


class LocalToolsService:
    """
    本地图谱检索工具服务（Neo4j + ChromaDB 后端）
    接口与 ZepToolsService 完全兼容
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm_client = llm_client

    @property
    def llm(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    # -----------------------------------------------------------------------
    # 基础数据访问方法（Neo4j + ChromaDB）
    # -----------------------------------------------------------------------

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ) -> SearchResult:
        """
        混合语义搜索（ChromaDB语义 60% + BM25关键词 40%）
        """
        logger.info(f"图谱搜索: graph_id={graph_id}, query={query[:50]}...")

        facts: List[str] = []
        edges: List[Dict[str, Any]] = []
        nodes: List[Dict[str, Any]] = []

        try:
            # --- 语义搜索 via ChromaDB ---
            query_emb = embed(query)
            chroma_results = semantic_search(
                query_embedding=query_emb,
                graph_id=graph_id,
                n_results=limit * 3,
            )

            # --- BM25 关键词评分 ---
            keywords = [w.strip() for w in query.lower().replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]

            def bm25_score(text: str) -> float:
                if not text:
                    return 0.0
                t = text.lower()
                if query.lower() in t:
                    return 1.0
                matches = sum(1 for kw in keywords if kw in t)
                return matches / max(len(keywords), 1) * 0.6

            # --- 混合评分 ---
            scored: List[Dict[str, Any]] = []
            for item in chroma_results:
                sem = item["score"]
                bm = bm25_score(item["document"])
                hybrid = 0.6 * sem + 0.4 * bm
                scored.append({**item, "hybrid": hybrid})

            scored.sort(key=lambda x: x["hybrid"], reverse=True)
            top = scored[:limit]

            seen_facts: set = set()
            for item in top:
                meta = item["metadata"]
                doc_type = meta.get("doc_type", "edge_fact")
                if doc_type == "edge_fact":
                    fact = item["document"]
                    if fact and fact not in seen_facts:
                        facts.append(fact)
                        seen_facts.add(fact)
                    edges.append({
                        "uuid": meta.get("edge_uuid", ""),
                        "name": meta.get("rel_type", ""),
                        "fact": fact,
                        "source_node_uuid": meta.get("source_uuid", ""),
                        "target_node_uuid": meta.get("target_uuid", ""),
                    })
                elif doc_type == "node_summary":
                    node_name = meta.get("node_name", "")
                    summary = item["document"]
                    if summary:
                        facts.append(f"[{node_name}]: {summary}")
                    nodes.append({
                        "uuid": meta.get("node_uuid", ""),
                        "name": node_name,
                        "labels": [],
                        "summary": summary,
                    })

        except Exception as e:
            logger.warning(f"向量搜索失败，降级为关键词搜索: {e}")
            return self._keyword_search(graph_id, query, limit, scope)

        logger.info(f"搜索完成: 找到 {len(facts)} 条相关事实")
        return SearchResult(facts=facts, edges=edges, nodes=nodes, query=query, total_count=len(facts))

    def _keyword_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ) -> SearchResult:
        """关键词匹配降级搜索"""
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]

        def match_score(text: str) -> int:
            if not text:
                return 0
            t = text.lower()
            if query_lower in t:
                return 100
            return sum(10 for kw in keywords if kw in t)

        facts: List[str] = []
        edges_result: List[Dict[str, Any]] = []
        nodes_result: List[Dict[str, Any]] = []

        if scope in ("edges", "both"):
            all_edges = self.get_all_edges(graph_id)
            scored = sorted(
                [(match_score(e.fact) + match_score(e.name), e) for e in all_edges if match_score(e.fact) + match_score(e.name) > 0],
                key=lambda x: x[0], reverse=True
            )
            for _, edge in scored[:limit]:
                if edge.fact:
                    facts.append(edge.fact)
                edges_result.append({"uuid": edge.uuid, "name": edge.name, "fact": edge.fact,
                                      "source_node_uuid": edge.source_node_uuid, "target_node_uuid": edge.target_node_uuid})

        if scope in ("nodes", "both"):
            all_nodes = self.get_all_nodes(graph_id)
            scored = sorted(
                [(match_score(n.name) + match_score(n.summary), n) for n in all_nodes if match_score(n.name) + match_score(n.summary) > 0],
                key=lambda x: x[0], reverse=True
            )
            for _, node in scored[:limit]:
                nodes_result.append({"uuid": node.uuid, "name": node.name, "labels": node.labels, "summary": node.summary})
                if node.summary:
                    facts.append(f"[{node.name}]: {node.summary}")

        return SearchResult(facts=facts, edges=edges_result, nodes=nodes_result, query=query, total_count=len(facts))

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """获取图谱的所有节点"""
        logger.info(f"获取图谱 {graph_id} 的所有节点...")
        records = run_query(
            "MATCH (n:Entity {graph_id: $g}) RETURN properties(n) AS props, labels(n) AS labels",
            {"g": graph_id},
        )
        result = []
        for rec in records:
            props = rec["props"]
            attrs = {}
            try:
                attrs = json.loads(props.get("attributes", "{}"))
            except Exception:
                pass
            result.append(NodeInfo(
                uuid=props.get("uuid", ""),
                name=props.get("name", ""),
                labels=rec["labels"],
                summary=props.get("summary", ""),
                attributes=attrs,
            ))
        logger.info(f"获取到 {len(result)} 个节点")
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """获取图谱的所有边（含时间信息）"""
        logger.info(f"获取图谱 {graph_id} 的所有边...")
        records = run_query(
            """
            MATCH (s:Entity {graph_id: $g})-[r]->(t:Entity {graph_id: $g})
            RETURN properties(r) AS props, type(r) AS rel_type,
                   s.uuid AS s_uuid, t.uuid AS t_uuid,
                   s.name AS s_name, t.name AS t_name
            """,
            {"g": graph_id},
        )
        result = []
        for rec in records:
            props = rec["props"]
            ei = EdgeInfo(
                uuid=props.get("uuid", ""),
                name=props.get("name", rec["rel_type"]),
                fact=props.get("fact", ""),
                source_node_uuid=rec["s_uuid"],
                target_node_uuid=rec["t_uuid"],
                source_node_name=rec["s_name"],
                target_node_name=rec["t_name"],
            )
            if include_temporal:
                ei.created_at = props.get("created_at")
                ei.valid_at = props.get("valid_at")
                ei.invalid_at = props.get("invalid_at")
                ei.expired_at = props.get("expired_at")
            result.append(ei)
        logger.info(f"获取到 {len(result)} 条边")
        return result

    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """获取单个节点详情"""
        records = run_query(
            "MATCH (n:Entity {uuid: $uuid}) RETURN properties(n) AS props, labels(n) AS labels",
            {"uuid": node_uuid},
        )
        if not records:
            return None
        props = records[0]["props"]
        attrs = {}
        try:
            attrs = json.loads(props.get("attributes", "{}"))
        except Exception:
            pass
        return NodeInfo(
            uuid=props.get("uuid", ""),
            name=props.get("name", ""),
            labels=records[0]["labels"],
            summary=props.get("summary", ""),
            attributes=attrs,
        )

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """获取节点相关的所有边"""
        try:
            records = run_query(
                """
                MATCH (n:Entity {uuid: $uuid})-[r]-(m:Entity {graph_id: $g})
                RETURN properties(r) AS props, type(r) AS rel_type,
                       startNode(r).uuid AS s_uuid, endNode(r).uuid AS t_uuid,
                       startNode(r).name AS s_name, endNode(r).name AS t_name
                """,
                {"uuid": node_uuid, "g": graph_id},
            )
            result = []
            for rec in records:
                props = rec["props"]
                result.append(EdgeInfo(
                    uuid=props.get("uuid", ""),
                    name=props.get("name", rec["rel_type"]),
                    fact=props.get("fact", ""),
                    source_node_uuid=rec["s_uuid"],
                    target_node_uuid=rec["t_uuid"],
                    source_node_name=rec["s_name"],
                    target_node_name=rec["t_name"],
                    valid_at=props.get("valid_at"),
                    invalid_at=props.get("invalid_at"),
                    expired_at=props.get("expired_at"),
                ))
            return result
        except Exception as e:
            logger.warning(f"获取节点边失败: {e}")
            return []

    def get_entities_by_type(self, graph_id: str, entity_type: str) -> List[NodeInfo]:
        """按类型获取实体"""
        all_nodes = self.get_all_nodes(graph_id)
        return [n for n in all_nodes if entity_type in n.labels]

    def get_entity_summary(self, graph_id: str, entity_name: str) -> Dict[str, Any]:
        """获取实体关系摘要"""
        search_result = self.search_graph(graph_id=graph_id, query=entity_name, limit=20)
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = next((n for n in all_nodes if n.name.lower() == entity_name.lower()), None)
        related_edges = self.get_node_edges(graph_id, entity_node.uuid) if entity_node else []
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges),
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """获取图谱统计信息"""
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)
        entity_types: Dict[str, int] = {}
        for node in nodes:
            for lbl in node.labels:
                if lbl not in ("Entity", "Node"):
                    entity_types[lbl] = entity_types.get(lbl, 0) + 1
        relation_types: Dict[str, int] = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1
        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types,
        }

    def get_simulation_context(self, graph_id: str, simulation_requirement: str, limit: int = 30) -> Dict[str, Any]:
        """获取模拟相关的上下文信息"""
        search_result = self.search_graph(graph_id=graph_id, query=simulation_requirement, limit=limit)
        stats = self.get_graph_statistics(graph_id)
        all_nodes = self.get_all_nodes(graph_id)
        entities = [
            {"name": n.name, "type": next((l for l in n.labels if l not in ("Entity", "Node")), "实体"), "summary": n.summary}
            for n in all_nodes
            if any(l not in ("Entity", "Node") for l in n.labels)
        ]
        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities),
        }

    # -----------------------------------------------------------------------
    # 核心检索工具（与 ZepToolsService 逻辑完全相同，调用本地数据方法）
    # -----------------------------------------------------------------------

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5,
    ) -> InsightForgeResult:
        """深度洞察检索 - 多子查询并行语义搜索 + 实体洞察 + 关系链"""
        logger.info(f"InsightForge 深度洞察检索: {query[:50]}...")

        result = InsightForgeResult(query=query, simulation_requirement=simulation_requirement, sub_queries=[])

        sub_queries = self._generate_sub_queries(query, simulation_requirement, report_context, max_sub_queries)
        result.sub_queries = sub_queries

        all_facts: List[str] = []
        all_edges_data: List[Dict[str, Any]] = []
        seen_facts: set = set()

        for sq in sub_queries:
            sr = self.search_graph(graph_id=graph_id, query=sq, limit=15, scope="edges")
            for f in sr.facts:
                if f not in seen_facts:
                    all_facts.append(f)
                    seen_facts.add(f)
            all_edges_data.extend(sr.edges)

        main_sr = self.search_graph(graph_id=graph_id, query=query, limit=20, scope="edges")
        for f in main_sr.facts:
            if f not in seen_facts:
                all_facts.append(f)
                seen_facts.add(f)

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)

        entity_uuids = set()
        for ed in all_edges_data:
            if isinstance(ed, dict):
                if ed.get("source_node_uuid"):
                    entity_uuids.add(ed["source_node_uuid"])
                if ed.get("target_node_uuid"):
                    entity_uuids.add(ed["target_node_uuid"])

        entity_insights = []
        node_map: Dict[str, NodeInfo] = {}
        for node_uuid in entity_uuids:
            if not node_uuid:
                continue
            try:
                node = self.get_node_detail(node_uuid)
                if node:
                    node_map[node_uuid] = node
                    entity_type = next((l for l in node.labels if l not in ("Entity", "Node")), "实体")
                    related_facts = [f for f in all_facts if node.name.lower() in f.lower()]
                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts,
                    })
            except Exception as e:
                logger.debug(f"获取节点 {node_uuid} 失败: {e}")

        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)

        relationship_chains: List[str] = []
        _empty_node = NodeInfo("", "", [], "", {})
        for ed in all_edges_data:
            if isinstance(ed, dict):
                src = node_map.get(ed.get("source_node_uuid", ""), _empty_node).name or ed.get("source_node_uuid", "")[:8]
                tgt = node_map.get(ed.get("target_node_uuid", ""), _empty_node).name or ed.get("target_node_uuid", "")[:8]
                chain = f"{src} --[{ed.get('name', '')}]--> {tgt}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)

        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)

        logger.info(f"InsightForge完成: {result.total_facts}条事实, {result.total_entities}个实体, {result.total_relationships}条关系")
        return result

    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5,
    ) -> List[str]:
        """使用LLM将问题分解为多个子问题"""
        system_prompt = """你是一个专业的问题分析专家。将复杂问题分解为多个可以在模拟世界中独立观察的子问题。
返回JSON格式：{"sub_queries": ["子问题1", "子问题2", ...]}"""
        user_prompt = (
            f"模拟需求背景：{simulation_requirement}\n"
            + (f"报告上下文：{report_context[:500]}\n" if report_context else "")
            + f"\n请将以下问题分解为{max_queries}个子问题：\n{query}"
        )
        try:
            response = self.llm.chat_json(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.3,
            )
            return [str(sq) for sq in response.get("sub_queries", [])[:max_queries]]
        except Exception as e:
            logger.warning(f"生成子问题失败: {e}，使用默认子问题")
            return [query, f"{query} 的主要参与者", f"{query} 的原因和影响", f"{query} 的发展过程"][:max_queries]

    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50,
    ) -> PanoramaResult:
        """广度搜索 - 获取全貌，包括历史/过期内容"""
        logger.info(f"PanoramaSearch 广度搜索: {query[:50]}...")

        result = PanoramaResult(query=query)
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)

        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)

        _empty = NodeInfo("", "", [], "", {})
        active_facts: List[str] = []
        historical_facts: List[str] = []

        for edge in all_edges:
            if not edge.fact:
                continue
            is_historical = edge.is_expired or edge.is_invalid
            if is_historical:
                valid_at = edge.valid_at or "未知"
                invalid_at = edge.invalid_at or edge.expired_at or "未知"
                historical_facts.append(f"[{valid_at} - {invalid_at}] {edge.fact}")
            else:
                active_facts.append(edge.fact)

        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]

        def relevance(fact: str) -> int:
            fl = fact.lower()
            if query_lower in fl:
                return 100
            return sum(10 for kw in keywords if kw in fl)

        active_facts.sort(key=relevance, reverse=True)
        historical_facts.sort(key=relevance, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info(f"PanoramaSearch完成: {result.active_count}条有效, {result.historical_count}条历史")
        return result

    def quick_search(self, graph_id: str, query: str, limit: int = 10) -> SearchResult:
        """简单快速搜索"""
        logger.info(f"QuickSearch 简单搜索: {query[:50]}...")
        result = self.search_graph(graph_id=graph_id, query=query, limit=limit, scope="edges")
        logger.info(f"QuickSearch完成: {result.total_count}条结果")
        return result

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None,
    ) -> InterviewResult:
        """深度采访模拟Agent（调用OASIS采访API，不依赖Zep）"""
        from .simulation_runner import SimulationRunner

        logger.info(f"InterviewAgents 深度采访: {interview_requirement[:50]}...")

        result = InterviewResult(interview_topic=interview_requirement, interview_questions=custom_questions or [])

        profiles = self._load_agent_profiles(simulation_id)
        if not profiles:
            result.summary = "未找到可采访的Agent人设文件"
            return result

        result.total_agents = len(profiles)

        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents,
        )
        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning

        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents,
            )

        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])
        INTERVIEW_PROMPT_PREFIX = (
            "你正在接受一次采访。请结合你的人设、所有的过往记忆与行动，"
            "以纯文本方式直接回答以下问题。\n"
            "回复要求：\n"
            "1. 直接用自然语言回答，不要调用任何工具\n"
            "2. 不要返回JSON格式或工具调用格式\n"
            "3. 不要使用Markdown标题（如#、##、###）\n"
            "4. 按问题编号逐一回答，每个回答以「问题X：」开头（X为问题编号）\n"
            "5. 每个问题的回答之间用空行分隔\n"
            "6. 回答要有实质内容，每个问题至少回答2-3句话\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"

        try:
            interviews_request = [
                {"agent_id": agent_idx, "prompt": optimized_prompt}
                for agent_idx in selected_indices
            ]
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,
                timeout=180.0,
            )

            if not api_result.get("success", False):
                result.summary = f"采访API调用失败：{api_result.get('error', '未知错误')}"
                return result

            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}

            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "未知")
                agent_bio = agent.get("bio", "")

                twitter_response = self._clean_tool_call_response(results_dict.get(f"twitter_{agent_idx}", {}).get("response", ""))
                reddit_response = self._clean_tool_call_response(results_dict.get(f"reddit_{agent_idx}", {}).get("response", ""))

                twitter_text = twitter_response if twitter_response else "（该平台未获得回复）"
                reddit_text = reddit_response if reddit_response else "（该平台未获得回复）"
                response_text = f"【Twitter平台回答】\n{twitter_text}\n\n【Reddit平台回答】\n{reddit_text}"

                combined_responses = f"{twitter_response} {reddit_response}"
                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'问题\d+[：:]\s*', '', clean_text)
                clean_text = re.sub(r'【[^】]+】', '', clean_text)

                sentences = re.split(r'[。！？]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W，,；;：:、]+', s.strip())
                    and not s.strip().startswith(('{', '问题'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "。" for s in meaningful[:3]]

                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[，,；;：:、]', q)][:3]

                result.interviews.append(AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5],
                ))

            result.interviewed_count = len(result.interviews)

        except ValueError as e:
            result.summary = f"采访失败：{str(e)}"
            return result
        except Exception as e:
            logger.error(f"采访API调用异常: {e}")
            result.summary = f"采访过程发生错误：{str(e)}"
            return result

        if result.interviews:
            result.summary = self._generate_interview_summary(result.interviews, interview_requirement)

        logger.info(f"InterviewAgents完成: 采访了 {result.interviewed_count} 个Agent")
        return result

    # -----------------------------------------------------------------------
    # 辅助方法（与 ZepToolsService 相同）
    # -----------------------------------------------------------------------

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """读取模拟的Agent人设文件"""
        import os
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
        profiles_dir = os.path.join(sim_dir, "profiles")
        if not os.path.exists(profiles_dir):
            return []
        profiles = []
        for fname in sorted(os.listdir(profiles_dir)):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(profiles_dir, fname), "r", encoding="utf-8") as f:
                        profiles.append(json.load(f))
                except Exception:
                    pass
        return profiles

    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int,
    ):
        """使用LLM智能选择采访Agent"""
        profile_summary = json.dumps(
            [{"index": i, "name": p.get("realname", p.get("username", "")), "bio": p.get("bio", "")[:100]}
             for i, p in enumerate(profiles[:30])],
            ensure_ascii=False,
        )
        system_prompt = "你是一个采访策划专家。根据采访需求，从候选人中选择最合适的受访者。返回JSON: {\"selected_indices\": [索引列表], \"reasoning\": \"选择理由\"}"
        user_prompt = f"采访需求：{interview_requirement}\n候选Agent：{profile_summary}\n请选择最多{max_agents}个最合适的Agent（返回0-based索引）。"
        try:
            response = self.llm.chat_json(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.3,
            )
            indices = [int(i) for i in response.get("selected_indices", []) if 0 <= int(i) < len(profiles)][:max_agents]
            reasoning = response.get("reasoning", "")
        except Exception:
            indices = list(range(min(max_agents, len(profiles))))
            reasoning = "自动选择"
        selected = [profiles[i] for i in indices]
        return selected, indices, reasoning

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]],
    ) -> List[str]:
        """生成采访问题"""
        system_prompt = "你是一个专业采访策划人。生成具体、有深度的采访问题。返回JSON: {\"questions\": [\"问题1\", \"问题2\", ...]}"
        user_prompt = f"采访需求：{interview_requirement}\n背景：{simulation_requirement}\n请生成5个采访问题。"
        try:
            response = self.llm.chat_json(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.4,
            )
            return [str(q) for q in response.get("questions", [])[:5]]
        except Exception:
            return [f"你对「{interview_requirement}」有什么看法？", "这件事对你有什么影响？", "你认为未来会怎么发展？"]

    def _generate_interview_summary(self, interviews: List[AgentInterview], interview_requirement: str) -> str:
        """生成采访摘要"""
        interview_texts = "\n\n".join([
            f"{iv.agent_name}（{iv.agent_role}）：{iv.response[:300]}"
            for iv in interviews
        ])
        system_prompt = "你是一个善于总结的分析师。请根据采访内容生成一段核心观点摘要。"
        user_prompt = f"采访主题：{interview_requirement}\n\n采访内容：\n{interview_texts}\n\n请生成200字以内的摘要。"
        try:
            return self.llm.chat(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.4,
            )
        except Exception:
            return "（摘要生成失败）"
