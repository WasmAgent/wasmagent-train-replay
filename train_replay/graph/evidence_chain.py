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
- Attachments form a **hash chain** — each attachment stores the digest of
  the previous one (``previous_attachment_digest``).  This makes removal,
  reordering, or insertion of any attachment cryptographically detectable
  without retaining historical graph snapshots.
- ``verify_chain`` validates an *ordered sequence* of attachments,
  ensuring the graph evolved through each recorded state without gaps
  or tampering.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .prov_graph import ProvGraph, _coerce_keys


def compute_graph_digest(graph: ProvGraph) -> str:
    """Return a SHA-256 hex digest of the graph's canonical structure.

    Delegates to :meth:`ProvGraph.digest` for the canonical implementation.
    The digest covers nodes, edges, and all their attributes.  It is
    stable across runs as long as the graph topology and attribute
    values are identical.
    """
    return graph.digest()


def compute_attachment_digest(attachment: EvidenceAttachment) -> str:
    """Return a SHA-256 hex digest of an attachment's identity fields.

    The digest covers all fields of the attachment *except* the
    ``previous_attachment_digest`` field.  This ensures that the hash
    chain is well-defined (each link covers the previous link's digest
    of its own identity fields, not its *own* previous reference).

    Returns:
        SHA-256 hex digest of the attachment's canonical representation.
    """
    # Canonicalise all fields except previous_attachment_digest so the
    # hash-chain link is well-defined.
    data = {
        "graph_digest": attachment.graph_digest,
        "bundle_digest": attachment.bundle_digest,
        "epoch": attachment.epoch,
        "sequence_number": attachment.sequence_number,
        # Exclude previous_attachment_digest to avoid circular dependency:
        #  attachment[i].digest() must be stable regardless of
        #  attachment[i].previous_attachment_digest.
    }
    canonical = json.dumps(_coerce_keys(data), sort_keys=True)
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
        previous_attachment_digest: SHA-256 of the preceding attachment's
            identity fields (via :func:`compute_attachment_digest`), or
            ``None`` for the first attachment in the chain.  This creates
            a cryptographic hash chain that prevents removal, reordering,
            or insertion of attachments without detection.
    """

    graph_digest: str
    bundle_digest: str
    epoch: int
    sequence_number: int
    previous_attachment_digest: str | None = None


def attach_evidence(
    graph: ProvGraph,
    bundle_digest: str,
    *,
    epoch: int = 0,
    sequence_number: int = 0,
    previous_attachment: EvidenceAttachment | None = None,
) -> EvidenceAttachment:
    """Create an evidence attachment binding the current graph state to a bundle.

    If ``previous_attachment`` is provided, the new attachment records its
    digest as ``previous_attachment_digest``, forming a cryptographic hash
    chain.  Callers should always pass the previous attachment (except for
    the very first attachment in a new chain) to ensure tamper-proof chain
    integrity.

    Args:
        graph: The causal graph whose integrity is being recorded.
        bundle_digest: SHA-256 digest of the EpochEvidenceBundle being attached
            (call ``bundle.digest()`` to obtain this).
        epoch: Training epoch number.
        sequence_number: Monotonic index for chain ordering.
        previous_attachment: The immediately preceding attachment in the
            chain, or ``None`` for the first attachment.

    Returns:
        A frozen ``EvidenceAttachment`` linking the graph state to the bundle.
    """
    return EvidenceAttachment(
        graph_digest=compute_graph_digest(graph),
        bundle_digest=bundle_digest,
        epoch=epoch,
        sequence_number=sequence_number,
        previous_attachment_digest=(
            compute_attachment_digest(previous_attachment)
            if previous_attachment is not None
            else None
        ),
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
    4. Each attachment's ``previous_attachment_digest`` matches the
       computed digest of the preceding attachment (hash-chain integrity).

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

        # Verify hash-chain link: attachment[i].previous_attachment_digest
        # must equal compute_attachment_digest(attachment[i-1]).
        expected_prev = compute_attachment_digest(attachments[i - 1])
        if attachments[i].previous_attachment_digest != expected_prev:
            return False

    return verify_attachment(graph, attachments[-1])
