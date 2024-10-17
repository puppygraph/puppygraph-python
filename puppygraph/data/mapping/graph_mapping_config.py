"""PuppyGraph construction config."""

from dataclasses import dataclass, field
from typing import List, Optional

from puppygraph.data.mapping.catalog_config import CatalogConfig
from puppygraph.data.mapping.graph_element_cache_config import (
    GraphElementCacheConfig,
)


@dataclass(frozen=True)
class TableSource:
    """Table source."""

    catalog_name: str
    schema_name: str
    table_name: str


@dataclass(frozen=True)
class MappedField:
    """MappedAttributes from the source table to the graph schema."""

    name: str
    type: str
    from_field: str
    description: Optional[str] = None


@dataclass(frozen=True)
class GraphElementConfig:
    """Graph Element config.

    A graph element is a vertex or an edge.
    """

    table_source: TableSource

    label: str

    # List of attributes, each attribute is a field in the source table
    attributes: List[MappedField]

    # The ID can from single field or list of fields (composite key)
    id: List[MappedField]

    # Description of the element
    description: Optional[str] = None

    # Cache config
    cache_config: Optional[GraphElementCacheConfig] = None

    # ID of the from vertex, only applicable if the element is an edge
    from_id: Optional[List[MappedField]] = None

    # ID of the to vertex, only applicable if the element is an edge
    to_id: Optional[List[MappedField]] = None

    # Label of the from vertex, only applicable if the element is an edge
    from_label: Optional[str] = None

    # Label of the to vertex, only applicable if the element is an edge
    to_label: Optional[str] = None


@dataclass(frozen=True)
class PuppyGraphMappingConfig:
    """PuppyGraph Mapping config.

    This config defines how the graph is mapped from the source tables.
    """

    # Catalogs to fetch tables from
    catalogs: List[CatalogConfig] = field(default_factory=list)

    vertices: List[GraphElementConfig] = field(default_factory=list)

    edges: List[GraphElementConfig] = field(default_factory=list)

    # description of the graph
    description: Optional[str] = None


if __name__ == "__main__":
    pass
    # print(GraphElementConfig)
    # print(PuppyGraphConstruction
