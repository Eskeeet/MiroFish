"""
本地图谱记忆更新服务
将模拟中的 Agent 活动动态写入 Neo4j + ChromaDB
（取代 Zep Cloud episode 写入）

架构与 ZepGraphMemoryUpdater 完全相同：
- 相同的 AgentActivity 数据类与所有 to_episode_text() 方法（直接复用）
- 相同的 队列 / 线程 / 平台缓冲区 / BATCH_SIZE 设计
- 唯一变化：_send_batch_activities() 写入本地 Neo4j 而非 Zep
"""

import time
import threading
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from queue import Queue, Empty

from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..utils.temporal import latest_timestamp, to_iso, utc_now_iso
from .temporal_graph import TemporalGraphService

from .agent_activity import AgentActivity

logger = get_logger('mirofish.local_graph_memory_updater')


# ---------------------------------------------------------------------------
# Episode Extractor  (轻量版，用于模拟中实时提取)
# ---------------------------------------------------------------------------

_EPISODE_SYSTEM_PROMPT = """你是一个知识图谱更新助手。
根据以下 Agent 活动描述，提取出新增或变化的实体关系事实。
仅提取活动中明确体现的信息，不要推断。

以 JSON 输出：
{
  "relationships": [
    {"source": "实体A", "target": "实体B", "type": "关系类型", "fact": "完整事实句", "valid_at": "ISO-8601或null", "invalid_at": "ISO-8601或null", "round_num": 轮次整数}
  ]
}

时间规则：
- valid_at 是事实开始成立的世界时间，invalid_at 是停止成立的世界时间
- 使用活动参考时间解析“上一轮”、“昨天”等相对时间
- 每条活动前已有精确时间和轮次；事实应复制其来源活动的时间与轮次
- 同一事实跨多条活动时，使用首次明确成立的时间和最新相关活动的轮次
- 当前持续成立的事实设 valid_at 为参考时间
- 仅在活动明确表示结束/变更时设置 invalid_at，不得编造时间

如果没有有意义的新关系，输出 {"relationships": []}"""


def _extract_episode_facts(
    episode_text: str,
    graph_id: str,
    valid_edge_types: List[str],
    llm: LLMClient,
    reference_time: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """从 episode 文本中提取关系事实（LLM调用）"""
    edge_types_str = json.dumps(valid_edge_types, ensure_ascii=False)
    prompt = (
        f"可用关系类型: {edge_types_str}\n"
        f"参考时间（UTC ISO-8601）: {reference_time or utc_now_iso()}\n\n"
        f"Agent活动记录：\n{episode_text}"
    )
    try:
        result = llm.chat_json(
            messages=[
                {"role": "system", "content": _EPISODE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        if isinstance(result, dict):
            return result.get("relationships", [])
    except Exception as e:
        logger.debug(f"episode提取失败（继续）: {e}")
    return []


def _get_graph_ontology(graph_id: str) -> List[str]:
    """从元数据文件读取可用边类型"""
    try:
        from .local_graph_builder import _load_meta
        meta = _load_meta(graph_id)
        if meta and "ontology" in meta:
            return [e["name"] for e in meta["ontology"].get("edge_types", [])]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# LocalGraphMemoryUpdater
# ---------------------------------------------------------------------------

class LocalGraphMemoryUpdater:
    """
    本地图谱记忆更新器
    接口与 ZepGraphMemoryUpdater 完全兼容
    """

    BATCH_SIZE = 5
    PLATFORM_DISPLAY_NAMES = {'twitter': '世界1', 'reddit': '世界2'}
    SEND_INTERVAL = 0.5
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    def __init__(self, graph_id: str, api_key: Optional[str] = None):
        self.graph_id = graph_id
        self._llm = LLMClient()
        self._valid_edge_types: List[str] = []  # 延迟加载

        self._activity_queue: Queue = Queue()
        self._platform_buffers: Dict[str, List[AgentActivity]] = {
            'twitter': [],
            'reddit': [],
        }
        self._buffer_lock = threading.Lock()

        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        self._total_activities = 0
        self._total_sent = 0
        self._total_items_sent = 0
        self._failed_count = 0
        self._skipped_count = 0

        logger.info(f"LocalGraphMemoryUpdater 初始化: graph_id={graph_id}")

    def start(self):
        if self._running:
            return
        # 加载本体边类型
        self._valid_edge_types = _get_graph_ontology(self.graph_id)

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"LocalMemoryUpdater-{self.graph_id[:8]}",
        )
        self._worker_thread.start()
        logger.info(f"LocalGraphMemoryUpdater 已启动: graph_id={self.graph_id}")

    def stop(self):
        self._running = False
        self._flush_remaining()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)
        logger.info(
            f"LocalGraphMemoryUpdater 已停止: graph_id={self.graph_id}, "
            f"activities={self._total_activities}, sent={self._total_items_sent}, "
            f"failed={self._failed_count}, skipped={self._skipped_count}"
        )

    def add_activity(self, activity: AgentActivity):
        if activity.action_type == "DO_NOTHING":
            self._skipped_count += 1
            return
        self._activity_queue.put(activity)
        self._total_activities += 1

    def add_activity_from_dict(self, data: Dict[str, Any], platform: str):
        if "event_type" in data:
            return
        activity = AgentActivity(
            platform=platform,
            agent_id=data.get("agent_id", 0),
            agent_name=data.get("agent_name", ""),
            action_type=data.get("action_type", ""),
            action_args=data.get("action_args", {}),
            round_num=data.get("round", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )
        self.add_activity(activity)

    def _worker_loop(self):
        while self._running or not self._activity_queue.empty():
            try:
                try:
                    activity = self._activity_queue.get(timeout=1)
                    platform = activity.platform.lower()
                    with self._buffer_lock:
                        if platform not in self._platform_buffers:
                            self._platform_buffers[platform] = []
                        self._platform_buffers[platform].append(activity)
                        if len(self._platform_buffers[platform]) >= self.BATCH_SIZE:
                            batch = self._platform_buffers[platform][:self.BATCH_SIZE]
                            self._platform_buffers[platform] = self._platform_buffers[platform][self.BATCH_SIZE:]
                            self._send_batch_activities(batch, platform)
                            time.sleep(self.SEND_INTERVAL)
                except Empty:
                    pass
            except Exception as e:
                logger.error(f"工作循环异常: {e}")
                time.sleep(1)

    def _send_batch_activities(self, activities: List[AgentActivity], platform: str):
        """
        将批量活动写入本地图谱
        1. 存储 Episode 节点（持久化原始文本，用于 round-end 延迟提取）
        2. 尝试即时 LLM 提取关系 → 写入 Neo4j + ChromaDB
        """
        if not activities:
            return

        episode_texts = [
            f"[{to_iso(activity.timestamp, fallback=utc_now_iso())}] "
            f"[轮次 {activity.round_num}] {activity.to_episode_text()}"
            for activity in activities
        ]
        combined_text = "\n".join(episode_texts)
        round_num = max(a.round_num for a in activities)
        round_from = min(a.round_num for a in activities)
        now = utc_now_iso()
        reference_time = latest_timestamp(
            (activity.timestamp for activity in activities),
            fallback=now,
        )

        # --- 1. 存储 Episode 节点（轻量，不依赖 LLM）---
        try:
            episode_uuid = TemporalGraphService.create_episode(
                self.graph_id,
                combined_text,
                reference_time=reference_time,
                source="message",
                source_description="OASIS simulation activity batch",
                name=f"{platform}_round_{round_num}",
                round_num=round_num,
                platform=platform,
                metadata={
                    "activity_count": len(activities),
                    "agent_ids": list(dict.fromkeys(a.agent_id for a in activities)),
                    "round_from": round_from,
                    "round_to": round_num,
                },
            )
        except Exception as e:
            logger.warning(f"存储 Episode 节点失败: {e}")
            self._failed_count += 1
            return

        # --- 2. LLM 提取关系 ---
        for attempt in range(self.MAX_RETRIES):
            try:
                relationships = _extract_episode_facts(
                    combined_text,
                    self.graph_id,
                    self._valid_edge_types,
                    self._llm,
                    reference_time=reference_time,
                )
                self._write_relationships(
                    relationships,
                    round_num,
                    episode_uuid,
                    reference_time or now,
                )

                self._total_sent += 1
                self._total_items_sent += len(activities)
                display = self.PLATFORM_DISPLAY_NAMES.get(platform, platform)
                logger.info(f"成功处理 {len(activities)} 条{display}活动 → graph_id={self.graph_id}")
                return

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"批量处理失败 ({attempt + 1}/{self.MAX_RETRIES}): {e}")
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"批量处理失败，已重试 {self.MAX_RETRIES} 次: {e}")
                    self._failed_count += 1

    def _write_relationships(
        self,
        relationships: List[Dict[str, Any]],
        round_num: int,
        episode_uuid: str,
        reference_time: str,
    ):
        """将提取的关系写入双时态图谱，并保留 Episode 来源。"""
        return TemporalGraphService.ingest_relationships(
            self.graph_id,
            relationships,
            episode_uuid=episode_uuid,
            reference_time=reference_time,
            round_num=round_num,
            valid_edge_types=set(self._valid_edge_types),
            llm=self._llm,
        )

    def _flush_remaining(self):
        while not self._activity_queue.empty():
            try:
                activity = self._activity_queue.get_nowait()
                platform = activity.platform.lower()
                with self._buffer_lock:
                    if platform not in self._platform_buffers:
                        self._platform_buffers[platform] = []
                    self._platform_buffers[platform].append(activity)
            except Empty:
                break

        with self._buffer_lock:
            for platform, buffer in self._platform_buffers.items():
                if buffer:
                    logger.info(f"发送{self.PLATFORM_DISPLAY_NAMES.get(platform, platform)}平台剩余 {len(buffer)} 条活动")
                    self._send_batch_activities(buffer, platform)
            for platform in self._platform_buffers:
                self._platform_buffers[platform] = []

    def get_stats(self) -> Dict[str, Any]:
        with self._buffer_lock:
            buffer_sizes = {p: len(b) for p, b in self._platform_buffers.items()}
        return {
            "graph_id": self.graph_id,
            "batch_size": self.BATCH_SIZE,
            "total_activities": self._total_activities,
            "batches_sent": self._total_sent,
            "items_sent": self._total_items_sent,
            "failed_count": self._failed_count,
            "skipped_count": self._skipped_count,
            "queue_size": self._activity_queue.qsize(),
            "buffer_sizes": buffer_sizes,
            "running": self._running,
        }


# ---------------------------------------------------------------------------
# LocalGraphMemoryManager
# ---------------------------------------------------------------------------

class LocalGraphMemoryManager:
    """管理多个模拟的本地图谱记忆更新器（接口与 ZepGraphMemoryManager 完全相同）"""

    _updaters: Dict[str, LocalGraphMemoryUpdater] = {}
    _lock = threading.Lock()
    _stop_all_done = False

    @classmethod
    def create_updater(cls, simulation_id: str, graph_id: str) -> LocalGraphMemoryUpdater:
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
            updater = LocalGraphMemoryUpdater(graph_id)
            updater.start()
            cls._updaters[simulation_id] = updater
            logger.info(f"创建本地记忆更新器: simulation_id={simulation_id}, graph_id={graph_id}")
            return updater

    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[LocalGraphMemoryUpdater]:
        return cls._updaters.get(simulation_id)

    @classmethod
    def stop_updater(cls, simulation_id: str):
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
                del cls._updaters[simulation_id]

    @classmethod
    def stop_all(cls):
        if cls._stop_all_done:
            return
        cls._stop_all_done = True
        with cls._lock:
            for sim_id, updater in list(cls._updaters.items()):
                try:
                    updater.stop()
                except Exception as e:
                    logger.error(f"停止更新器失败: {sim_id}, {e}")
            cls._updaters.clear()
        logger.info("已停止所有本地记忆更新器")

    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        return {sid: u.get_stats() for sid, u in cls._updaters.items()}
