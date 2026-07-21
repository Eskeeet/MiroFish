"""Compatibility exports for the fully local graph builder."""

from .graph_models import GraphInfo
from .local_graph_builder import LocalGraphBuilderService


GraphBuilderService = LocalGraphBuilderService

__all__ = ["GraphBuilderService", "LocalGraphBuilderService", "GraphInfo"]
