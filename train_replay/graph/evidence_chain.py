"""Tamper-proof evidence chain binding ProvGraph to signed EpochEvidenceBundles.

This module closes the integrity gap between the causal graph (PROV-DM) and the
signed evidence layer (AEP/Ed25519).  It provides:

- ``compute_graph_digest``: canonical SHA-256 hash of a ProvGraph's structure.
- ``EvidenceAttachment``: immutable record binding a graph state to a bundle.
- ``attach_evidence`` / ``verify_attachment`` / ``verify_chain``: core
  integrity operations.

Design goals
------------
- The graph digest covers all nodes, edges, and their attributes — any
  addition, removal, or mutation invalidates the digest.
- Attachments are frozen (hashable, immutable) so they can be stored,
  compared, or serialized without risk of post-hoc modification.
- ``verify_chain`` validates an *ordered sequence* of attachments,
  ensuring the graph evolved through each recorded state without gaps
  or tampering.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .prov_graph import ProvGraph


def compute_graph_digest(graph: ProvGraph) -> str:
    """Return a SHA-256 hex digest of the graph's canonical structure.

    The digest covers nodes, edges, and all their attributes.  It is
    stable across runs as long as the graph topology and attribute
    values are identical.
    """
    representation = graph.to_dict()
    canonical = json.dumps(representation, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class EvidenceAttachment:
    """Immutable record binding a graph state to a signed evidence bundle.

    Attributes:
        graph_digest: SHA-256 of the ProvGraph at attachment time.
        bundle_digest: SHA-256 of the EpochEvidenceBundle at attachment time.
        epoch: The epoch number of the attached bundle.
        sequence_number: Monotonically increasing attachment index within a
            graph's lifecycle (0-based).  Used by ``verify_chain`` to detect
            gaps or reordering.
    """

    graph_digest: str
    bundle_digest: str
    epoch: int
    sequence_number: int


def attach_evidence(
    graph: ProvGraph,
    bundle_digest: str,
    *,
    epoch: int = 0,
    sequence_number: int = 0,
) -> EvidenceAttachment:
    """Create an evidence attachment binding the current graph state to a bundle.

    Args:
        graph: The causal graph whose integrity is being recorded.
        bundle_digest: SHA-256 digest of the EpochEvidenceBundle being attached
            (call ``bundle.digest()`` to obtain this).
        epoch: Training epoch number.
        sequence_number: Monotonic index for chain ordering.

    Returns:
        A frozen ``EvidenceAttachment`` linking the graph state to the bundle.
    """
    return EvidenceAttachment(
        graph_digest=compute_graph_digest(graph),
        bundle_digest=bundle_digest,
        epoch=epoch,
        sequence_number=sequence_number,
    )


def verify_attachment(graph: ProvGraph, attachment: EvidenceAttachment) -> bool:
    """Check that a previously attached evidence record still matches the graph.

    Returns:
        ``True`` if the graph's current digest matches the attachment's
        recorded ``graph_digest``.  Returns ``False`` if the graph has been
        modified since the attachment was created.
    """
    return compute_graph_digest(graph) == attachment.graph_digest


def verify_chain(
    graph: ProvGraph,
    attachments: list[EvidenceAttachment],
) -> bool:
    """Verify an ordered chain of evidence attachments against the graph.

    The chain is valid if:
    1. Sequence numbers are contiguous (0, 1, 2, …) with no gaps.
    2. Epochs are non-decreasing.
    3. The last attachment's graph digest matches the current graph state.

    Note:
        Intermediate attachments are NOT checked against the current graph
        state — only the final attachment is.  Intermediate digests are
        preserved for audit trail purposes but cannot be verified without
        retaining historical graph snapshots.

    Args:
        graph: The current state of the causal graph.
        attachments: Ordered list of evidence attachments.

    Returns:
        ``True`` if the chain is structurally valid and the final digest
        matches the current graph.
    """
    if not attachments:
        return False

    for i, att in enumerate(attachments):
        if att.sequence_number != i:
            return False

    for i in range(1, len(attachments)):
        if attachments[i].epoch < attachments[i - 1].epoch:
            return False

    return verify_attachment(graph, attachments[-1])
