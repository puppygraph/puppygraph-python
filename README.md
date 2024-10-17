# PuppyGraph

This repository contains the PuppyGraph client for Python as well
as PuppyGraph's Agentic Graph RAG libraries.

## Key Features

- **Zero ETL**: Query data directly from your lakes and databases without data duplication.
- **Dynamic Schema Management**: Modify graph schemas on the fly, without needing to rebuild databases.
- **Petabyte Scalability**: Auto-sharded, distributed computation for handling vast datasets.
- **Support for Cypher and Gremlin**: Interoperable query support with robust performance.
- **AI-Native**: Ideal for Graph-RAG applications, with ultra-fast response times.

## Installation

You can install the latest version via pip:

```bash
pip install puppygraph
```

## Quick Example

### Setup the client
    
```python
    from puppygraph import PuppyGraphClient, PuppyGraphHostConfig
    client = PuppyGraphClient(PuppyGraphHostConfig("localhost"))
```

### Query the graph
```python
    # Cypher Query
    client.cypher_query("MATCH (actor:Actor)-[:ACTED_IN]->(movie:Movie) WHERE actor.name = \"Tom Hanks\"
                        "RETURN movie.title")

    # Gremlin Query 
    client.gremlin_query("g.V().hasLabel('person').has('name', 'Tom Hanks').out('ACTED_IN').values('title')")
```

### Set the schema
```python
    # A sample schema for the IMDb dataset
    client.set_schema(
    {
        "catalogs": [
            {
                "name": "imdb_catalog",
                "type": "DELTALAKE",
                "params": {
                    "metastore_param": {
                        "token": "your_token",
                        "host": "your_host",
                        "unity_catalog_name": "imdb_catalog",
                    },
                    "storage_param": {
                        "use_instance_profile": "false",
                        "region": "us-west-2",
                        "access_key": "your_access_key",
                        "secret_key": "your_secret_key",
                        "enable_ssl": "true",
                        "type": "S3",
                    },
                },
            }
        ],
        "vertices": [
            {
                "table_source": {
                    "catalog_name": "imdb_catalog",
                    "schema_name": "public",
                    "table_name": "movies",
                },
                "label": "Movie",
                "description": "A movie in the IMDb database",
                "attributes": [
                    {
                        "name": "title",
                        "from_field": "title",
                        "type": "String",
                        "description": "The title of the movie",
                    },
                    {
                        "name": "release_year",
                        "from_field": "release_year",
                        "type": "Integer",
                        "description": "The year the movie was released",
                    },
                ],
                "id": [
                    {"name": "movie_id", "from_field": "movie_id", "type": "String"}
                ],
            },
            {
                "table_source": {
                    "catalog_name": "imdb_catalog",
                    "schema_name": "public",
                    "table_name": "actors",
                },
                "label": "Actor",
                "description": "An actor who starred in movies",
                "attributes": [
                    {
                        "name": "name",
                        "from_field": "name",
                        "type": "String",
                        "description": "The name of the actor",
                    }
                ],
                "id": [
                    {"name": "actor_id", "from_field": "actor_id", "type": "String"}
                ],
            },
        ],
        "edges": [
            {
                "table_source": {
                    "catalog_name": "imdb_catalog",
                    "schema_name": "public",
                    "table_name": "acted_in",
                },
                "label": "ACTED_IN",
                "from_label": "Actor",
                "to_label": "Movie",
                "description": "An actor acted in a movie",
                "attributes": [],
                "id": [
                    {
                        "name": "acted_in_id",
                        "from_field": "acted_in_id",
                        "type": "String",
                    }
                ],
                "from_id": [
                    {"name": "actor_id", "from_field": "actor_id", "type": "String"}
                ],
                "to_id": [
                    {"name": "movie_id", "from_field": "movie_id", "type": "String"}
                ],
            }
        ],
    }
)
    
```

## About PuppyGraph

[PuppyGraph](https://www.puppygraph.com) 
is a zero-ETL graph analytics engine enabling seamless graph querying across one or multiple data sources. 
Unlike traditional graph databases, PuppyGraph connects directly to your data warehouses and lakes without requiring complex ETL pipelines, making it both cost-efficient and scalable.
