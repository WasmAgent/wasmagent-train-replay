"""
wasmagent-train-replay — causal evidence layer for distributed GPU training.

Reads PyTorch Flight Recorder dumps, NCCL traces, and profiler hooks,
builds a cross-rank PROV-DM causal graph, records layered AEP evidence,
and supports deterministic replay from any epoch.
"""

__version__ = "0.1.0"
