"""Defines Database parameters for a catalog."""

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass(frozen=True)
class JDBCParam:
    """JDBC connection parameters for a database."""

    username: str
    password: str
    jdbc_uri: str
    driver_class: str
    driver_url: str


@dataclass(frozen=True)
class ElasticSearchParam:
    """ElasticSearch connection parameters for a database."""

    hosts: List[str]
    username: Optional[str] = None
    password: Optional[str] = None


DatabaseParams = Union[JDBCParam, ElasticSearchParam]
