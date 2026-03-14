"""
Neo4j图数据库连接管理
"""

import re
import threading
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver

from .logger import get_logger
from ..config import Config

logger = get_logger('mirofish.graph_db')

_driver: Optional[Driver] = None
_driver_lock = threading.Lock()


def get_driver() -> Driver:
    """获取或创建Neo4j驱动（单例）"""
    global _driver
    if _driver is None:
        with _driver_lock:
            if _driver is None:
                _driver = GraphDatabase.driver(
                    Config.NEO4J_URI,
                    auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
                )
                logger.info(f"Neo4j驱动已初始化: {Config.NEO4J_URI}")
    return _driver


def run_query(cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """执行Cypher读查询，返回记录列表"""
    driver = get_driver()
    params = params or {}
    with driver.session() as session:
        result = session.run(cypher, params)
        return [record.data() for record in result]


def run_write(cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """执行Cypher写查询（写事务）"""
    driver = get_driver()
    params = params or {}

    def _execute(tx):
        return list(tx.run(cypher, params))

    with driver.session() as session:
        records = session.execute_write(_execute)
        return [record.data() for record in records]


def validate_rel_type(rel_type: str) -> str:
    """
    验证并清理关系类型字符串，防止Cypher注入。
    只允许大写字母、数字和下划线。
    """
    clean = re.sub(r'[^A-Z0-9_]', '_', rel_type.upper())
    if not clean or clean[0].isdigit():
        clean = 'RELATED_TO'
    return clean


def create_relationship(source_uuid: str, target_uuid: str, rel_type: str, props: Dict[str, Any]):
    """
    在两个节点之间创建关系（关系类型安全插值）
    """
    safe_type = validate_rel_type(rel_type)
    cypher = f"""
        MATCH (s:Entity {{uuid: $s_uuid}})
        MATCH (t:Entity {{uuid: $t_uuid}})
        MERGE (s)-[r:{safe_type} {{uuid: $r_uuid}}]->(t)
        SET r += $props
        RETURN r
    """
    return run_write(cypher, {
        "s_uuid": source_uuid,
        "t_uuid": target_uuid,
        "r_uuid": props.get("uuid", ""),
        "props": props,
    })


def ensure_constraints():
    """确保必要的索引和约束存在"""
    try:
        run_write("CREATE CONSTRAINT entity_uuid IF NOT EXISTS FOR (n:Entity) REQUIRE n.uuid IS UNIQUE")
    except Exception as e:
        logger.debug(f"entity_uuid约束: {e}")
    try:
        run_write("CREATE INDEX entity_graph_id IF NOT EXISTS FOR (n:Entity) ON (n.graph_id)")
    except Exception as e:
        logger.debug(f"entity_graph_id索引: {e}")
    try:
        run_write("CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")
    except Exception as e:
        logger.debug(f"entity_name索引: {e}")
    logger.info("Neo4j约束和索引已确认")


def close_driver():
    """关闭Neo4j驱动"""
    global _driver
    if _driver:
        _driver.close()
        _driver = None
        logger.info("Neo4j驱动已关闭")
