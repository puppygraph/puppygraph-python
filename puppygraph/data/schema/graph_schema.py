"""PuppyGraph schema definition."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class AttributeSchema:
    """Attribute schema."""

    type: str
    name: str
    description: Optional[str] = None


@dataclass(frozen=True)
class VertexSchema:
    """Vertex schema."""

    label: str
    attributes: List[AttributeSchema] = field(default_factory=list)
    description: Optional[str] = None


@dataclass(frozen=True)
class EdgeSchema:
    """Edge schema."""

    label: str
    from_vertex_label: str
    to_vertex_label: str
    attributes: List[AttributeSchema] = field(default_factory=list)
    description: Optional[str] = None


@dataclass(frozen=True)
class PuppyGraphSchema:
    """The PuppyGraph schema."""

    vertex_schemas: List[VertexSchema] = field(default_factory=list)
    edge_schemas: List[EdgeSchema] = field(default_factory=list)
    description: Optional[str] = None
