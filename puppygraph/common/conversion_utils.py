"""Utility functions for conversion."""

from typing import Any, Dict, List, Union

import dacite
from puppygraph.common.dataclass_utils import dataclass_to_camel_dict
from puppygraph.data.mapping.catalog_config import CatalogConfig, CatalogType
from puppygraph.data.mapping.database_params import (
    DatabaseParams,
    ElasticSearchParam,
    JDBCParam,
)
from puppygraph.data.mapping.datalake_params import (
    DatalakeParams,
    MetastoreParam,
    S3StorageParam,
    UnityMetastoreParam,
)
from puppygraph.data.mapping.graph_element_cache_config import (
    GraphElementCacheConfig,
)
from puppygraph.data.mapping.graph_mapping_config import (
    GraphElementConfig,
    MappedField,
    PuppyGraphMappingConfig,
    TableSource,
)


def convert_mapping_config_to_host_json(
    config: Union[PuppyGraphMappingConfig, Dict],
) -> Dict[str, Any]:
    """Converts the PuppyGraph Mapping config to the host schema JSON.

    Args:
        config: The PuppyGraph Mapping config.

    Returns:
        The host schema JSON.
    """

    def _metastore_param_to_json(metastore_param: MetastoreParam) -> Dict[str, Any]:
        json = {}
        if isinstance(metastore_param, UnityMetastoreParam):
            json = dataclass_to_camel_dict(metastore_param)
            json["type"] = "unity"
            # Rename unityCatalogName to databricksCatalogName
            if "unityCatalogName" in json:
                json["databricksCatalogName"] = json.pop("unityCatalogName")
        return json

    def _storage_param_to_json(storage_param: S3StorageParam) -> Dict[str, Any]:
        json = {}
        if isinstance(storage_param, S3StorageParam):
            json = dataclass_to_camel_dict(storage_param)
            json["type"] = "S3"
        return json

    def _datalake_params_to_json(datalake_params: DatalakeParams) -> Dict[str, Any]:
        json = {}
        if datalake_params.metastore_param is not None:
            json["metastore"] = _metastore_param_to_json(
                datalake_params.metastore_param
            )
        if datalake_params.storage_param is not None:
            json["storage"] = _storage_param_to_json(datalake_params.storage_param)
        return json

    def _database_params_to_json(database_params: DatabaseParams) -> Dict[str, Any]:
        key = ""
        if isinstance(database_params, JDBCParam):
            key = "jdbc"
        elif isinstance(database_params, ElasticSearchParam):
            key = "elasticSearch"
        return {key: dataclass_to_camel_dict(database_params)}

    def _catalog_config_to_json(catalog_config: CatalogConfig) -> Dict[str, Any]:
        """Maps the catalog config to the host catalog JSON.

        Args:
            catalog_config: The catalog configuration to map.

        Returns:
            The host catalog JSON.
        """
        catalog_json = {
            "name": catalog_config.name,
            "type": catalog_config.type.name.lower(),
        }

        if isinstance(catalog_config.params, DatalakeParams):
            catalog_json.update(_datalake_params_to_json(catalog_config.params))
        elif isinstance(catalog_config.params, DatabaseParams):
            catalog_json.update(_database_params_to_json(catalog_config.params))

        return catalog_json

    def _table_source_to_json(table_source: TableSource) -> Dict[str, Any]:
        return {
            "catalog": table_source.catalog_name,
            "schema": table_source.schema_name,
            "table": table_source.table_name,
        }

    def _mapped_id_to_json(mapped_id: List[MappedField]) -> Dict[str, Any]:
        return {
            "fields": [
                {
                    "field": mapped_field.from_field,
                    "alias": mapped_field.name,
                    "type": mapped_field.type,
                }
                for mapped_field in mapped_id
            ]
        }

    def _attribute_to_json(mapped_attribute: MappedField) -> Dict[str, Any]:
        return {
            "alias": mapped_attribute.name,
            "field": mapped_attribute.from_field,
            "type": mapped_attribute.type,
        }

    def _cache_config_to_json(cache_config: GraphElementCacheConfig) -> Dict[str, Any]:
        return dataclass_to_camel_dict(cache_config)

    def _vertex_config_to_json(vertex_config: GraphElementConfig) -> Dict[str, Any]:
        json = {
            "label": vertex_config.label,
            "oneToOne": {
                "tableSource": _table_source_to_json(
                    table_source=vertex_config.table_source
                ),
                "id": _mapped_id_to_json(mapped_id=vertex_config.id),
                "attributes": [
                    _attribute_to_json(
                        mapped_attribute=attribute,
                    )
                    for attribute in vertex_config.attributes
                ],
            },
        }
        if vertex_config.cache_config is not None:
            json["cache"] = _cache_config_to_json(vertex_config.cache_config)
        return json

    def _edge_config_to_json(
        edge_config: GraphElementConfig,
    ) -> Dict[str, Any]:

        json = {
            "label": edge_config.label,
            "fromVertex": edge_config.from_label,
            "toVertex": edge_config.to_label,
            "tableSource": _table_source_to_json(table_source=edge_config.table_source),
            "id": _mapped_id_to_json(mapped_id=edge_config.id),
            "fromId": _mapped_id_to_json(mapped_id=edge_config.from_id),
            "toId": _mapped_id_to_json(mapped_id=edge_config.to_id),
            "attributes": [
                _attribute_to_json(
                    mapped_attribute=attribute,
                )
                for attribute in edge_config.attributes
            ],
        }

        if edge_config.cache_config is not None:
            json["cache"] = _cache_config_to_json(edge_config.cache_config)
        return json

    if isinstance(config, dict):
        config = dacite.from_dict(
            data_class=PuppyGraphMappingConfig,
            data=config,
            config=dacite.Config(
                type_hooks={
                    CatalogType: lambda x: CatalogType[x.upper()],
                }
            ),
        )

    return {
        "catalogs": [
            _catalog_config_to_json(catalog_config)
            for catalog_config in config.catalogs
        ],
        "graph": {
            "vertices": [
                _vertex_config_to_json(vertex_config=vertex_config)
                for vertex_config in config.vertices
            ],
            "edges": [
                _edge_config_to_json(edge_config=edge_config)
                for edge_config in config.edges
            ],
        },
    }
