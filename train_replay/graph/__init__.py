"""PROV-DM causal graph with backend-neutral collective operation abstractions.

Re-exports
----------
Core graph types (``prov_graph``):
    ProvGraph, ProvActivity, ProvEntity, ProvAgent

Backend-neutral ops (``ops``):
    Backend, CollectiveOp, OpSpec

Collision detection protocol (``collision``):
    CollisionDetector, CollisionEvent, CollisionSeverity

Evidence chain (``evidence_chain``):
    compute_graph_digest, EvidenceAttachment, attach_evidence,
    verify_attachment, verify_chain

Builder (``builder``):
    build_from_events, build_from_specs
"""

from .builder import build_from_events, build_from_specs
from .collision import CollisionDetector, CollisionEvent, CollisionSeverity
from .evidence_chain import (
    EvidenceAttachment,
    attach_evidence,
    compute_graph_digest,
    verify_attachment,
    verify_chain,
)
from .ops import Backend, CollectiveOp, OpSpec
from .prov_graph import ProvActivity, ProvAgent, ProvEntity, ProvGraph

__all__ = [
    "attach_evidence",
    "Backend",
    "build_from_events",
    "build_from_specs",
    "CollectiveOp",
    "CollisionDetector",
    "CollisionEvent",
    "CollisionSeverity",
    "compute_graph_digest",
    "EvidenceAttachment",
    "OpSpec",
    "ProvActivity",
    "ProvAgent",
    "ProvEntity",
    "ProvGraph",
    "verify_attachment",
    "verify_chain",
]
