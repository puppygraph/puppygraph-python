"""Keep Cypher's public result shape independent of Neo4j's Record container."""

import json
from unittest.mock import MagicMock

import pytest
from neo4j import Record, Session
from neo4j.exceptions import CypherSyntaxError
from neo4j.graph import Graph, Node, Path

from puppygraph import PuppyGraphClient, PuppyGraphHostConfig


# --- Local transport helpers ---
@pytest.fixture
def cypher_client(monkeypatch):
    """Use real client constructors and values without opening a graph connection."""
    client = PuppyGraphClient(PuppyGraphHostConfig("127.0.0.1"))
    session = MagicMock(spec=Session)
    session.__enter__.return_value = session
    monkeypatch.setattr(client._cypher_query_driver, "session", lambda: session)
    try:
        yield client, session
    finally:
        client._cypher_query_driver.close()
        client._gremlin_query_client.close()
# --- End local transport helpers ---


def test_cypher_returns_named_dictionaries(cypher_client):
    client, session = cypher_client
    session.run.return_value = [
        Record([("answer", 42), ("details", {"values": [True, None]})]),
        Record([("answer", 0), ("details", None)]),
    ]
    expected = [
        {"answer": 42, "details": {"values": [True, None]}},
        {"answer": 0, "details": None},
    ]

    rows = client.cypher_query("RETURN $answer AS answer", {"answer": 42}, 5000)

    assert all(isinstance(row, dict) for row in rows)
    assert rows == expected
    assert json.loads(json.dumps(rows)) == expected
    query, params = session.run.call_args.args
    assert query.text == "RETURN $answer AS answer"
    assert query.timeout == 5
    assert params == {"answer": 42}
    session.__exit__.assert_called_once()


def test_cypher_preserves_graph_metadata_in_nested_results(cypher_client):
    client, session = cypher_client
    node = Node(Graph(), "person:7", 7, ["Person"], {"name": "Ada"})
    record = Record([
        ("person", node),
        ("nested", {"matches": [node, None], "path": Path(node)}),
    ])
    session.run.return_value = [record]
    expected_node = {
        "type": "Node", "id": "person:7", "labels": ["Person"],
        "properties": {"name": "Ada"},
    }

    rows = client.cypher_query("MATCH (p:Person) RETURN p AS person")

    assert rows == [{
        "person": expected_node,
        "nested": {
            "matches": [expected_node, None],
            "path": {"type": "Path", "nodes": [expected_node], "relationships": []},
        },
    }]
    assert json.loads(json.dumps(rows)) == rows
    assert record["person"] is node
    assert record["nested"]["matches"][0] is node
    assert dict(node) == {"name": "Ada"}
    session.__exit__.assert_called_once()


def test_cypher_empty_results_and_syntax_errors_keep_session_cleanup(cypher_client):
    client, session = cypher_client
    session.run.return_value = []
    assert client.cypher_query("MATCH (p:Missing) RETURN p") == []
    session.__exit__.assert_called_once()

    session.reset_mock()
    error = CypherSyntaxError("invalid query")
    session.run.side_effect = error
    with pytest.raises(ValueError, match="is not valid") as raised:
        client.cypher_query("INVALID")
    assert raised.value.__cause__ is error
    session.__exit__.assert_called_once()
