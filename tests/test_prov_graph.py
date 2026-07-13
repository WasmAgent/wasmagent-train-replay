"""Tests for PROV-DM causal graph."""

from train_replay.graph.prov_graph import (
    ProvActivity,
    ProvAgent,
    ProvEntity,
    ProvGraph,
)


def _make_graph() -> ProvGraph:
    g = ProvGraph()
    g.add_agent(ProvAgent(id="rank:0:pg:default", rank=0, process_group="default"))
    g.add_activity(ProvActivity(id="act:0:all_reduce:1", label="all_reduce",
                                rank=0, process_group="default",
                                timestamp_ns=1000, collective_type="all_reduce"))
    g.add_entity(ProvEntity(id="tensor:0:1:in", digest=None, rank=0, step=1))
    g.add_entity(ProvEntity(id="tensor:0:1:out", digest=None, rank=0, step=1))
    g.used("act:0:all_reduce:1", "tensor:0:1:in")
    g.was_generated_by("tensor:0:1:out", "act:0:all_reduce:1")
    return g


def test_ancestors_found():
    g = _make_graph()
    ancestors = g.ancestors_of("tensor:0:1:out")
    assert "act:0:all_reduce:1" in ancestors


def test_leaf_entity_no_ancestors():
    g = _make_graph()
    assert g.ancestors_of("tensor:0:1:in") == []


def test_causal_subgraph():
    g = _make_graph()
    sub = g.causal_subgraph("tensor:0:1:out")
    node_ids = [n for n, _ in sub.nodes()]
    assert "act:0:all_reduce:1" in node_ids


class TestGraphDigest:
    def test_deterministic_digest(self):
        g = _make_graph()
        assert g.digest() == g.digest()

    def test_different_graphs_different_digests(self):
        g1 = _make_graph()
        g2 = ProvGraph()
        g2.add_agent(ProvAgent(id="rank:0:pg:default", rank=0, process_group="default"))
        assert g1.digest() != g2.digest()

    def test_mutation_invalidates_digest(self):
        g = _make_graph()
        original = g.digest()
        g.add_entity(ProvEntity(id="extra", digest=None, rank=0, step=99))
        assert g.digest() != original

    def test_digest_is_sha256_hex(self):
        g = _make_graph()
        d = g.digest()
        assert len(d) == 64
        int(d, 16)  # valid hex
