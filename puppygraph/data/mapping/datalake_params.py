"""Defines Datalake parameters for a catalog."""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class UnityMetastoreParam:
    """Unity metastore parameters."""

    token: Optional[str] = None
    host: Optional[str] = None
    unity_catalog_name: Optional[str] = None


@dataclass(frozen=True)
class S3StorageParam:
    """S3 storage parameters."""

    use_instance_profile: Optional[str] = None
    region: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    iam_role_arn: Optional[str] = None
    enable_ssl: Optional[str] = None
    endpoint: Optional[str] = None
    enable_path_style_access: Optional[str] = None


MetastoreParam = Union[UnityMetastoreParam]
StorageParam = Union[S3StorageParam]


@dataclass(frozen=True)
class DatalakeParams:
    """Datalake parameters for a catalog."""

    metastore_param: Optional[MetastoreParam] = None
    storage_param: Optional[StorageParam] = None
