"""Compatibility exports; graph reads are handled entirely by Neo4j."""

from .graph_models import EntityNode, FilteredEntities
from .local_entity_reader import LocalEntityReader


ZepEntityReader = LocalEntityReader

__all__ = ["ZepEntityReader", "LocalEntityReader", "EntityNode", "FilteredEntities"]
