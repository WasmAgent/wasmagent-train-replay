"""PyTorch autograd / profiler hooks for tensor-level evidence collection."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class TensorEvent:
    tensor_id: str
    op_name: str
    rank: int
    step: int
    shape: list[int] = field(default_factory=list)
    digest: str | None = None


class EvidenceProfilerHook:
    """Attach to a training loop to collect tensor-level provenance events."""

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self._events: list[TensorEvent] = []
        self._step = 0

    def on_step_begin(self) -> None:
        self._step += 1

    def record_tensor(self, op_name: str, tensor: "torch.Tensor", digest: str | None = None) -> None:
        import hashlib, struct
        if digest is None:
            try:
                flat = tensor.detach().cpu().float().numpy().tobytes()
                digest = hashlib.sha256(flat[:4096]).hexdigest()[:16]
            except Exception:
                digest = None
        self._events.append(TensorEvent(
            tensor_id=f"r{self.rank}:s{self._step}:{op_name}",
            op_name=op_name,
            rank=self.rank,
            step=self._step,
            shape=list(tensor.shape),
            digest=digest,
        ))

    @property
    def events(self) -> list[TensorEvent]:
        return list(self._events)
