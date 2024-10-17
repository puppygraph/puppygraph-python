import logging

from puppygraph import PuppyGraphClient, PuppyGraphHostConfig

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    cilent = PuppyGraphClient(PuppyGraphHostConfig(ip="127.0.0.1"))
    cilent.set_schema(
        {
            "catalogs": [
                {
                    "name": "pg_databricks",
                    "type": "DELTALAKE",
                    "params": {
                        "metastore_param": {
                            "token": "${ENV:DATABRICKS_TOKEN}",
                            "host": "${ENV:DATABRICKS_HOST}",
                            "unity_catalog_name": "pg_databricks",
                        },
                        "storage_param": {
                            "use_instance_profile": "false",
                            "region": "us-east-1",
                            "access_key": "${ENV:AWS_ACCESS_KEY_ID}",
                            "secret_key": "${ENV:AWS_SECRET_ACCESS_KEY}",
                            "enable_ssl": "false",
                            "type": "S3",
                        },
                    },
                }
            ],
            "vertices": [
                {
                    "table_source": {
                        "catalog_name": "pg_databricks",
                        "schema_name": "bronze",
                        "table_name": "failure_type",
                    },
                    "label": "failure_type",
                    "description": "A type of failure",
                    "attributes": [
                        {
                            "name": "failure_type_name",
                            "from_field": "failure_type_name",
                            "type": "String",
                            "description": "The name of the failure type",
                        }
                    ],
                    "id": [
                        {
                            "name": "failure_type_id",
                            "from_field": "failure_type_id",
                            "type": "String",
                        }
                    ],
                },
                {
                    "table_source": {
                        "catalog_name": "pg_databricks",
                        "schema_name": "silver",
                        "table_name": "assets",
                    },
                    "label": "asset",
                    "description": "An asset in the system",
                    "attributes": [
                        {
                            "name": "asset_id",
                            "from_field": "asset_id",
                            "type": "String",
                            "description": "The ID of the asset",
                        },
                        {
                            "name": "asset_name",
                            "from_field": "asset_name",
                            "type": "String",
                            "description": "The name of the asset",
                        },
                        {
                            "name": "asset_type",
                            "from_field": "asset_type",
                            "type": "String",
                            "description": "The type of the asset",
                        },
                        {
                            "name": "location",
                            "from_field": "location",
                            "type": "String",
                            "description": "The location of the asset",
                        },
                        {
                            "name": "acquisition_date",
                            "from_field": "acquisition_date_formatted",
                            "type": "Date",
                            "description": "The acquisition date of the asset",
                        },
                        {
                            "name": "status",
                            "from_field": "status",
                            "type": "String",
                            "description": "The status of the asset",
                        },
                    ],
                    "id": [{"name": "id", "from_field": "asset_id", "type": "String"}],
                },
                {
                    "table_source": {
                        "catalog_name": "pg_databricks",
                        "schema_name": "gold",
                        "table_name": "work_orders",
                    },
                    "label": "work_order",
                    "description": "A work order in the system",
                    "attributes": [
                        {
                            "name": "work_order_id",
                            "from_field": "work_order_id",
                            "type": "String",
                            "description": "The ID of the work order",
                        },
                        {
                            "name": "date",
                            "from_field": "date",
                            "type": "Date",
                            "description": "The date of the work order",
                        },
                        {
                            "name": "action_taken",
                            "from_field": "action_taken",
                            "type": "String",
                            "description": "The action taken for the work order",
                        },
                        {
                            "name": "technician",
                            "from_field": "technician",
                            "type": "String",
                            "description": "The technician handling the work order",
                        },
                        {
                            "name": "component_replaced_description",
                            "from_field": "component_replaced_description",
                            "type": "String",
                            "description": "Description of the component replaced",
                        },
                        {
                            "name": "component_replaced_material_num",
                            "from_field": "component_replaced_material_num",
                            "type": "String",
                            "description": "Material number of the component replaced",
                        },
                        {
                            "name": "repeated_work_order",
                            "from_field": "repeated_work_order",
                            "type": "Boolean",
                            "description": "Whether the work order is a repeated one",
                        },
                        {
                            "name": "successful_fix",
                            "from_field": "successful_fix",
                            "type": "Boolean",
                            "description": "Whether the issue was successfully fixed",
                        },
                    ],
                    "id": [
                        {"name": "id", "from_field": "work_order_id", "type": "String"}
                    ],
                },
            ],
            "edges": [
                {
                    "table_source": {
                        "catalog_name": "pg_databricks",
                        "schema_name": "silver",
                        "table_name": "troubleshooting_guide",
                    },
                    "label": "can_have_failure",
                    "from_label": "asset",
                    "to_label": "failure_type",
                    "description": "An asset can have a failure type",
                    "attributes": [
                        {
                            "name": "steps_to_follow",
                            "from_field": "steps_to_follow",
                            "type": "String",
                            "description": "Steps to follow for the failure",
                        },
                        {
                            "name": "reference_source",
                            "from_field": "reference_source",
                            "type": "String",
                            "description": "The reference source for the failure",
                        },
                        {
                            "name": "recommended_actions",
                            "from_field": "recommended_actions",
                            "type": "String",
                            "description": "The recommended actions for the failure",
                        },
                    ],
                    "id": [
                        {
                            "name": "can_have_failure_id",
                            "from_field": "reference_id",
                            "type": "String",
                        }
                    ],
                    "from_id": [
                        {"name": "asset_id", "from_field": "asset_id", "type": "String"}
                    ],
                    "to_id": [
                        {
                            "name": "failure_type_id",
                            "from_field": "failure_type",
                            "type": "String",
                        }
                    ],
                },
                {
                    "table_source": {
                        "catalog_name": "pg_databricks",
                        "schema_name": "gold",
                        "table_name": "work_orders",
                    },
                    "label": "worked_on",
                    "from_label": "work_order",
                    "to_label": "asset",
                    "description": "A work order worked on an asset",
                    "attributes": [],
                    "id": [
                        {
                            "name": "worked_on_id",
                            "from_field": "work_order_id",
                            "type": "String",
                        }
                    ],
                    "from_id": [
                        {
                            "name": "work_order_id",
                            "from_field": "work_order_id",
                            "type": "String",
                        }
                    ],
                    "to_id": [
                        {"name": "asset_id", "from_field": "asset_id", "type": "String"}
                    ],
                },
                {
                    "table_source": {
                        "catalog_name": "pg_databricks",
                        "schema_name": "gold",
                        "table_name": "work_orders",
                    },
                    "label": "related_to_failure",
                    "from_label": "work_order",
                    "to_label": "failure_type",
                    "description": "A work order is related to a failure type",
                    "attributes": [],
                    "id": [
                        {
                            "name": "related_to_failure_id",
                            "from_field": "work_order_id",
                            "type": "String",
                        }
                    ],
                    "from_id": [
                        {
                            "name": "work_order_id",
                            "from_field": "work_order_id",
                            "type": "String",
                        }
                    ],
                    "to_id": [
                        {
                            "name": "failure_type_id",
                            "from_field": "llm_failure_type",
                            "type": "String",
                        }
                    ],
                },
            ],
        }
    )
