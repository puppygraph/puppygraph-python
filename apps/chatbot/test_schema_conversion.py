import os
import unittest
from unittest.mock import patch

from backend import PuppyGraphChatbot


class SchemaConversionTest(unittest.TestCase):
    def setUp(self):
        self.chatbot = PuppyGraphChatbot.__new__(PuppyGraphChatbot)

    def test_converts_v1_schema(self):
        raw_schema = {
            "node": [
                {
                    "label": "person",
                    "id": [{"name": "id", "type": "STRING"}],
                    "attribute": [{"name": "age", "type": "INT"}],
                    "dataSourceGroup": {},
                }
            ],
            "edge": [
                {
                    "label": "knows",
                    "fromNodeLabel": "person",
                    "toNodeLabel": "person",
                    "id": [],
                    "fromKey": [{"name": "from_id", "type": "STRING"}],
                    "toKey": [{"name": "to_id", "type": "STRING"}],
                    "attribute": [{"name": "weight", "type": "DOUBLE"}],
                    "dataSourceGroup": {},
                }
            ],
            "localTable": [],
        }

        converted = self.chatbot._convert_puppygraph_schema(raw_schema)

        self.assertEqual(converted["vertices"][0]["label"], "person")
        self.assertEqual(
            converted["vertices"][0]["ids"],
            [{"name": "id", "type": "String"}],
        )
        self.assertEqual(
            converted["vertices"][0]["attributes"],
            [{"name": "age", "type": "Integer"}],
        )
        self.assertEqual(converted["edges"][0]["from"], "person")
        self.assertEqual(converted["edges"][0]["to"], "person")

    def test_still_converts_v0_schema(self):
        raw_schema = {
            "graph": {
                "vertices": [
                    {
                        "label": "person",
                        "oneToOne": {
                            "id": {
                                "fields": [
                                    {"alias": "id", "field": "person_id", "type": "String"}
                                ]
                            },
                            "attributes": [
                                {"alias": "name", "field": "full_name", "type": "String"}
                            ],
                        },
                    }
                ],
                "edges": [],
            }
        }

        converted = self.chatbot._convert_puppygraph_schema(raw_schema)

        self.assertEqual(len(converted["vertices"]), 1)
        self.assertEqual(converted["vertices"][0]["ids"][0]["name"], "id")
        self.assertEqual(
            converted["vertices"][0]["attributes"][0]["name"], "name"
        )


class EnvironmentConfigTest(unittest.TestCase):
    @patch("backend.TextToCypherRAG")
    def test_uses_environment_connection_settings(self, rag_class):
        values = {
            "PUPPYGRAPH_BOLT_URI": "bolt://graph.example:7687",
            "PUPPYGRAPH_HTTP_URI": "https://graph.example",
            "PUPPYGRAPH_USERNAME": "service-account",
            "PUPPYGRAPH_PASSWORD": "secret",
        }
        with patch.dict(os.environ, values, clear=False):
            chatbot = PuppyGraphChatbot()

        rag_class.assert_called_once()
        self.assertEqual(chatbot.puppygraph_bolt_uri, values["PUPPYGRAPH_BOLT_URI"])
        self.assertEqual(chatbot.puppygraph_http_uri, values["PUPPYGRAPH_HTTP_URI"])
        self.assertEqual(chatbot.puppygraph_username, values["PUPPYGRAPH_USERNAME"])
        self.assertEqual(chatbot.puppygraph_password, values["PUPPYGRAPH_PASSWORD"])


if __name__ == "__main__":
    unittest.main()
