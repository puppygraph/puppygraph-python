"""Config for PuppyGraph server."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PuppyGraphHostConfig:
    """PuppyGraph host configuration."""

    # IP address of the PuppyGraph host
    ip: str

    # HTTP port of the PuppyGraph server
    http_port: int = 8081

    # Cypher query port of the PuppyGraph server
    cypher_port: int = 7687

    # Gremlin query port of the PuppyGraph server
    gremlin_port: int = 8182

    # Username to authenticate with the PuppyGraph server
    username: str = "puppygraph"

    # Password to authenticate with the PuppyGraph server
    password: str = "puppygraph123"
