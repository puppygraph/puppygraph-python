"""Catalog configuration dataclasses and enums.

This module defines the access configurations for different types of data sources.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Union

from puppygraph.data.mapping.database_params import DatabaseParams
from puppygraph.data.mapping.datalake_params import DatalakeParams


class CatalogType(Enum):
    """Defines the source type of a catalog."""

    UNKNOWN = 0
    HIVE = 1
    ICEBERG = 2
    HUDI = 3
    DELTALAKE = 4
    MYSQL = 5
    POSTGRESQL = 6
    ELASTICSEARCH = 7
    REDSHIFT = 8
    DUCKDB = 9
    BIGQUERY = 10
    SNOWFLAKE = 11
    TRINO = 12
    VERTICA = 13
    SINGLESTORE = 14


@dataclass(frozen=True)
class CatalogConfig:
    """Defines the source of a catalog for PuppyGraph to construct the graph from."""

    name: str
    type: CatalogType
    params: Union[DatalakeParams, DatabaseParams]
