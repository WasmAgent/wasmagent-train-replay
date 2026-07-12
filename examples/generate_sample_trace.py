"""Generate a sample Flight Recorder pickle fixture for CLI tests and demos.

Usage:
    python examples/generate_sample_trace.py

Produces examples/sample_trace.pkl containing synthetic NCCL collective
events across 4 ranks with 3 steps each.

Security note
-------------
The output is plain built-in data only (``dict``/``list``/``int``/``str``/
``None``). ``load_flight_recorder`` deserializes every dump through a
restricted unpickler that refuses any ``GLOBAL``/``REDUCE`` opcode, so this
fixture — like any genuine NCCL dump — loads without exposing the loader to
arbitrary code execution.
"""

from __future__ import annotations

import pickle
from pathlib import Path


def main() -> None:
    entries: list[dict] = []
    ranks = 4
    steps = 3

    for rank in range(ranks):
        for seq in range(steps):
            entries.append(
                {
                    "rank": rank,
                    "pg_name": "default",
                    "collective_seq": "all_reduce",
                    "p2p_src": None,
                    "p2p_dst": None,
                    "input_sizes": [[1024 * 1024]],
                    "time_created_ns": seq * 1_000_000,
                    "time_started_ns": seq * 1_000_000 + 100_000,
                    "time_finished_ns": seq * 1_000_000 + 500_000,
                    "frames": [],
                    "seq_id": seq,
                }
            )

    data = {"entries": entries}

    out = Path(__file__).resolve().parent / "sample_trace.pkl"
    with open(out, "wb") as f:
        pickle.dump(data, f)

    print(f"Generated {out} with {len(entries)} collective events across {ranks} ranks, {steps} steps each.")


if __name__ == "__main__":
    main()
