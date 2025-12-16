#!/usr/bin/env python3
"""
merge_torch_profiler_traces_gz.py

Merge multiple PyTorch profiler exported .json.gz trace files.
Useful for multi-iteration / multi-GPU / multi-rank profiling.
"""

import argparse
import gzip
import json
import pathlib
from typing import Any


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def merge_traces(trace_dir: str, output_file: str):
    trace_dir = pathlib.Path(trace_dir)
    merged: dict[str, Any] | None = None
    next_pid_base = 0

    json_files = sorted(trace_dir.glob("*.json.gz"))
    if not json_files:
        print(f"No .json.gz files found in {trace_dir}")
        return

    for idx, fn in enumerate(json_files):
        with gzip.open(fn, "rt", encoding="utf-8") as f:
            try:
                part = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse {fn}, skipping")
                continue

        if merged is None:
            merged = {k: v for k, v in part.items() if k != "traceEvents"}
            merged["traceEvents"] = []

        events = part.get("traceEvents", [])
        # Chrome Trace / Perfetto compatibility: process_name/thread_name metadata expects pid/tid
        # to remain numeric. If pid/tid are converted to strings, the UI may degrade and show
        # a less informative numeric-only hierarchy.
        # Therefore we only apply an offset to *integer* pids to avoid pid collisions across files.
        int_pids = [e.get("pid") for e in events if _is_int(e.get("pid"))]
        file_max_pid = max(int_pids) if int_pids else -1
        pid_offset = next_pid_base if file_max_pid >= 0 else 0
        next_pid_base = (
            next_pid_base + file_max_pid + 1
            if file_max_pid >= 0
            else next_pid_base
        )

        for ev in events:
            if _is_int(ev.get("pid")):
                ev["pid"] = int(ev["pid"]) + pid_offset
            # Keep tid unchanged (together with pid it forms a unique (pid, tid) key) to preserve
            # thread/stream readability.
            merged["traceEvents"].append(ev)

    if merged is None:
        print(f"No valid traces found in {trace_dir}")
        return

    # Write plain JSON
    with open(output_file, "w") as f:
        json.dump(merged, f)
    print(
        f"Merged {len(json_files)} traces into {output_file}, total events: {len(merged['traceEvents'])}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple PyTorch profiler .json.gz traces"
    )
    parser.add_argument(
        "--trace_dir",
        type=str,
        required=True,
        help="Directory containing .json.gz trace files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="merged_trace.json",
        help="Output merged JSON file",
    )
    args = parser.parse_args()

    merge_traces(args.trace_dir, args.output)
