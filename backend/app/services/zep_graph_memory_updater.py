"""Compatibility exports for the local temporal graph memory updater."""

from .agent_activity import AgentActivity
from .local_graph_memory_updater import LocalGraphMemoryManager, LocalGraphMemoryUpdater


ZepGraphMemoryUpdater = LocalGraphMemoryUpdater
ZepGraphMemoryManager = LocalGraphMemoryManager

__all__ = [
    "AgentActivity",
    "ZepGraphMemoryUpdater",
    "ZepGraphMemoryManager",
    "LocalGraphMemoryUpdater",
    "LocalGraphMemoryManager",
]
