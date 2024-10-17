import pytest
from puppygraph.common.conversion_utils import convert_mapping_config_to_host_json


@pytest.mark.parametrize(
    "config_dict, expected_json",
    [
        (
            {
                "catalogs": [
                    {
                        "name": "my_catalog",
                        "type": "DELTALAKE",
                        "params": {
                            "metastore_param": {
                                "token": "my_token",
                                "host": "my_host",
                                "unity_catalog_name": "my_catalog_name",
                            },
                            "storage_param": {
                                "use_instance_profile": "false",
                                "region": "my_region",
                                "access_key": "my_access_key",
                                "secret_key": "my_secret_key",
                                "enable_ssl": "true",
                                "type": "S3",
                            },
                        },
                    }
                ],
                "vertices": [
                    {
                        "label": "Person",
                        "table_source": {
                            "catalog_name": "my_catalog",
                            "schema_name": "my_schema",
                            "table_name": "person_table",
                        },
                        "id": [{"name": "id", "type": "String", "from_field": "id"}],
                        "attributes": [
                            {
                                "name": "name",
                                "type": "String",
                                "from_field": "person_name",
                            },
                            {"name": "age", "type": "Int", "from_field": "person_age"},
                        ],
                    },
                    {
                        "label": "Location",
                        "table_source": {
                            "catalog_name": "my_catalog",
                            "schema_name": "my_schema",
                            "table_name": "location_table",
                        },
                        "id": [{"name": "id", "type": "String", "from_field": "id"}],
                        "attributes": [
                            {
                                "name": "name",
                                "type": "String",
                                "from_field": "location_name",
                            },
                            {
                                "name": "latitude",
                                "type": "Float",
                                "from_field": "location_lat",
                            },
                            {
                                "name": "longitude",
                                "type": "Float",
                                "from_field": "location_long",
                            },
                        ],
                    },
                ],
                "edges": [
                    {
                        "label": "LivesIn",
                        "from_label": "Person",
                        "to_label": "Location",
                        "table_source": {
                            "catalog_name": "my_catalog",
                            "schema_name": "my_schema",
                            "table_name": "lives_in_table",
                        },
                        "id": [{"name": "id", "type": "String", "from_field": "id"}],
                        "from_id": [
                            {
                                "name": "from_id",
                                "type": "String",
                                "from_field": "from_id",
                            }
                        ],
                        "to_id": [
                            {"name": "to_id", "type": "String", "from_field": "to_id"}
                        ],
                        "attributes": [
                            {
                                "name": "since",
                                "type": "Date",
                                "from_field": "since_date",
                            }
                        ],
                    },
                    {
                        "label": "Likes",
                        "from_label": "Person",
                        "to_label": "Person",
                        "table_source": {
                            "catalog_name": "my_catalog",
                            "schema_name": "my_schema",
                            "table_name": "likes_table",
                        },
                        "id": [{"name": "id", "type": "String", "from_field": "id"}],
                        "from_id": [
                            {
                                "name": "from_id",
                                "type": "String",
                                "from_field": "from_id",
                            }
                        ],
                        "to_id": [
                            {"name": "to_id", "type": "String", "from_field": "to_id"}
                        ],
                        "attributes": [],
                    },
                ],
            },
            {
                "catalogs": [
                    {
                        "name": "my_catalog",
                        "type": "deltalake",
                        "metastore": {
                            "type": "unity",
                            "token": "my_token",
                            "host": "my_host",
                            "databricksCatalogName": "my_catalog_name",
                        },
                        "storage": {
                            "useInstanceProfile": "false",
                            "region": "my_region",
                            "accessKey": "my_access_key",
                            "secretKey": "my_secret_key",
                            "enableSsl": "true",
                            "type": "S3",
                        },
                    },
                ],
                "graph": {
                    "vertices": [
                        {
                            "label": "Person",
                            "oneToOne": {
                                "tableSource": {
                                    "catalog": "my_catalog",
                                    "schema": "my_schema",
                                    "table": "person_table",
                                },
                                "id": {
                                    "fields": [
                                        {"field": "id", "type": "String", "alias": "id"}
                                    ]
                                },
                                "attributes": [
                                    {
                                        "alias": "name",
                                        "field": "person_name",
                                        "type": "String",
                                    },
                                    {
                                        "alias": "age",
                                        "field": "person_age",
                                        "type": "Int",
                                    },
                                ],
                            },
                        },
                        {
                            "label": "Location",
                            "oneToOne": {
                                "tableSource": {
                                    "catalog": "my_catalog",
                                    "schema": "my_schema",
                                    "table": "location_table",
                                },
                                "id": {
                                    "fields": [
                                        {"field": "id", "type": "String", "alias": "id"}
                                    ]
                                },
                                "attributes": [
                                    {
                                        "alias": "name",
                                        "field": "location_name",
                                        "type": "String",
                                    },
                                    {
                                        "alias": "latitude",
                                        "field": "location_lat",
                                        "type": "Float",
                                    },
                                    {
                                        "alias": "longitude",
                                        "field": "location_long",
                                        "type": "Float",
                                    },
                                ],
                            },
                        },
                    ],
                    "edges": [
                        {
                            "label": "LivesIn",
                            "fromVertex": "Person",
                            "toVertex": "Location",
                            "tableSource": {
                                "catalog": "my_catalog",
                                "schema": "my_schema",
                                "table": "lives_in_table",
                            },
                            "id": {
                                "fields": [
                                    {"field": "id", "type": "String", "alias": "id"}
                                ]
                            },
                            "fromId": {
                                "fields": [
                                    {
                                        "field": "from_id",
                                        "type": "String",
                                        "alias": "from_id",
                                    }
                                ]
                            },
                            "toId": {
                                "fields": [
                                    {
                                        "field": "to_id",
                                        "type": "String",
                                        "alias": "to_id",
                                    }
                                ]
                            },
                            "attributes": [
                                {
                                    "alias": "since",
                                    "field": "since_date",
                                    "type": "Date",
                                }
                            ],
                        },
                        {
                            "label": "Likes",
                            "fromVertex": "Person",
                            "toVertex": "Person",
                            "tableSource": {
                                "catalog": "my_catalog",
                                "schema": "my_schema",
                                "table": "likes_table",
                            },
                            "id": {
                                "fields": [
                                    {"field": "id", "type": "String", "alias": "id"}
                                ]
                            },
                            "fromId": {
                                "fields": [
                                    {
                                        "field": "from_id",
                                        "type": "String",
                                        "alias": "from_id",
                                    }
                                ]
                            },
                            "toId": {
                                "fields": [
                                    {
                                        "field": "to_id",
                                        "type": "String",
                                        "alias": "to_id",
                                    }
                                ]
                            },
                            "attributes": [],
                        },
                    ],
                },
            },
        )
    ],
)
def test_convert_schema_and_construction_to_host_json(config_dict, expected_json):
    """Test conversion of schema and construction to host json."""
    assert convert_mapping_config_to_host_json(config=config_dict) == expected_json
