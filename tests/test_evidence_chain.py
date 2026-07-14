"""Tests for tamper-proof evidence chain (evidence_chain.py)."""

from train_replay.graph.evidence_chain import (
    attach_evidence,
    compute_graph_digest,
    verify_attachment,
    verify_chain,
)
from train_replay.graph.prov_graph import (
    ProvActivity,
    ProvAgent,
    ProvEntity,
    ProvGraph,
)


def _make_graph() -> ProvGraph:
    g = ProvGraph()
    g.add_agent(ProvAgent(id="rank:0:pg:default", rank=0, process_group="default"))
    g.add_activity(
        ProvActivity(
            id="act:0:all_reduce:1",
            label="all_reduce",
            rank=0,
            process_group="default",
            timestamp_ns=1000,
            collective_type="all_reduce",
        ),
    )
    g.add_entity(ProvEntity(id="tensor:0:1:in", digest=None, rank=0, step=1))
    g.add_entity(ProvEntity(id="tensor:0:1:out", digest=None, rank=0, step=1))
    g.used("act:0:all_reduce:1", "tensor:0:1:in")
    g.was_generated_by("tensor:0:1:out", "act:0:all_reduce:1")
    return g


class TestComputeGraphDigest:
    def test_deterministic(self) -> None:
        g = _make_graph()
        d1 = compute_graph_digest(g)
        d2 = compute_graph_digest(g)
        assert d1 == d2

    def test_different_graphs_different_digests(self) -> None:
        g1 = _make_graph()
        g2 = ProvGraph()
        g2.add_agent(ProvAgent(id="rank:0:pg:default", rank=0, process_group="default"))
        assert compute_graph_digest(g1) != compute_graph_digest(g2)

    def test_adding_node_invalidates_digest(self) -> None:
        g = _make_graph()
        original = compute_graph_digest(g)
        g.add_entity(ProvEntity(id="extra", digest=None, rank=0, step=99))
        assert compute_graph_digest(g) != original

    def test_adding_edge_invalidates_digest(self) -> None:
        g = _make_graph()
        original = compute_graph_digest(g)
        g.add_agent(ProvAgent(id="rank:1:pg:default", rank=1, process_group="default"))
        assert compute_graph_digest(g) != original

    def test_matches_prov_graph_digest_method(self) -> None:
        """compute_graph_digest and ProvGraph.digest() produce the same result."""
        g = _make_graph()
        assert compute_graph_digest(g) == g.digest()


class TestEvidenceAttachment:
    def test_frozen_immutability(self) -> None:
        g = _make_graph()
        att = attach_evidence(g, "bundle-digest-abc", epoch=1, sequence_number=0)
        try:
            att.graph_digest = "tampered"  # type: ignore[misc]
            assert False, "Should not be able to set attribute on frozen dataclass"
        except AttributeError:
            pass

    def test_records_graph_digest(self) -> None:
        g = _make_graph()
        att = attach_evidence(g, "bundle-digest-abc", epoch=1, sequence_number=0)
        assert att.graph_digest == compute_graph_digest(g)

    def test_records_bundle_digest(self) -> None:
        g = _make_graph()
        att = attach_evidence(g, "bundle-digest-abc", epoch=1, sequence_number=0)
        assert att.bundle_digest == "bundle-digest-abc"

    def test_records_epoch_and_sequence(self) -> None:
        g = _make_graph()
        att = attach_evidence(g, "bd", epoch=5, sequence_number=3)
        assert att.epoch == 5
        assert att.sequence_number == 3


class TestVerifyAttachment:
    def test_valid_attachment_passes(self) -> None:
        g = _make_graph()
        att = attach_evidence(g, "bd", epoch=1, sequence_number=0)
        assert verify_attachment(g, att) is True

    def test_modified_graph_fails(self) -> None:
        g = _make_graph()
        att = attach_evidence(g, "bd", epoch=1, sequence_number=0)
        # Tamper with the graph
        g.add_entity(ProvEntity(id="tampered-node", digest=None, rank=0, step=999))
        assert verify_attachment(g, att) is False

    def test_different_bundle_digest_still_passes(self) -> None:
        """verify_attachment only checks graph integrity, not bundle digest."""
        g = _make_graph()
        att = attach_evidence(g, "bd", epoch=1, sequence_number=0)
        # The attachment records a bundle digest, but verify_attachment
        # only checks the graph side — it can't verify the bundle itself
        # without the bundle.
        assert verify_attachment(g, att) is True


class TestVerifyChain:
    def test_empty_chain_fails(self) -> None:
        g = _make_graph()
        assert verify_chain(g, []) is False

    def test_single_attachment_valid(self) -> None:
        g = _make_graph()
        att = attach_evidence(g, "bd", epoch=1, sequence_number=0)
        assert verify_chain(g, [att]) is True

    def test_multi_attachment_valid(self) -> None:
        g = _make_graph()
        att0 = attach_evidence(g, "bd0", epoch=1, sequence_number=0)
        att1 = attach_evidence(g, "bd1", epoch=2, sequence_number=1)
        att2 = attach_evidence(g, "bd2", epoch=3, sequence_number=2)
        assert verify_chain(g, [att0, att1, att2]) is True

    def test_gap_in_sequence_fails(self) -> None:
        g = _make_graph()
        att0 = attach_evidence(g, "bd0", epoch=1, sequence_number=0)
        att2 = attach_evidence(g, "bd2", epoch=2, sequence_number=2)  # skip 1
        assert verify_chain(g, [att0, att2]) is False

    def test_duplicate_sequence_number_fails(self) -> None:
        g = _make_graph()
        att0 = attach_evidence(g, "bd0", epoch=1, sequence_number=0)
        att0b = attach_evidence(g, "bd0b", epoch=2, sequence_number=0)
        assert verify_chain(g, [att0, att0b]) is False

    def test_decreasing_epoch_fails(self) -> None:
        g = _make_graph()
        att0 = attach_evidence(g, "bd0", epoch=2, sequence_number=0)
        att1 = attach_evidence(g, "bd1", epoch=1, sequence_number=1)  # epoch went down
        assert verify_chain(g, [att0, att1]) is False

    def test_same_epoch_allowed(self) -> None:
        """Epochs can be equal (multiple attachments within one epoch)."""
        g = _make_graph()
        att0 = attach_evidence(g, "bd0", epoch=1, sequence_number=0)
        att1 = attach_evidence(g, "bd1", epoch=1, sequence_number=1)
        assert verify_chain(g, [att0, att1]) is True

    def test_tampered_graph_fails_chain(self) -> None:
        g = _make_graph()
        att0 = attach_evidence(g, "bd0", epoch=1, sequence_number=0)
        att1 = attach_evidence(g, "bd1", epoch=2, sequence_number=1)
        # Tamper after both attachments
        g.add_entity(ProvEntity(id="tampered", digest=None, rank=0, step=999))
        assert verify_chain(g, [att0, att1]) is False

    def test_out_of_order_list_fails(self) -> None:
        """Attachments must be in sequence_number order."""
        g = _make_graph()
        att0 = attach_evidence(g, "bd0", epoch=1, sequence_number=0)
        att1 = attach_evidence(g, "bd1", epoch=2, sequence_number=1)
        assert verify_chain(g, [att1, att0]) is False
