#!/usr/bin/env python
"""Parse a profiler trace file and export per-kernel CSV + summary.

Usage:
  python scripts/run_parse.py --trace-file /path/to/trace.json.gz --output-dir /path/to/out

If --output-dir is omitted, falls back to ARTIFACT_ENV or DEFAULT_ARTIFACT_DIR from tests.utils.
"""

import os
import sys
import time
import argparse
from simulator.base_parser import BaseKernelInfoParser
from tests.utils import _write_artifact, ARTIFACT_ENV, DEFAULT_ARTIFACT_DIR


def parse_args():
    p = argparse.ArgumentParser(
        description="Parse profiler trace and generate CSV"
    )
    p.add_argument(
        "--trace-file", required=True, help="Path to .trace.json.gz file"
    )
    p.add_argument(
        "--output-dir",
        required=False,
        help="Directory to place CSV and summary artifacts",
    )
    return p.parse_args()


def main():
    args = parse_args()
    trace_file = args.trace_file
    if not os.path.exists(trace_file):
        print(
            f"[ERROR] Trace file does not exist: {trace_file}", file=sys.stderr
        )
        return 1

    out_dir = (
        args.output_dir or os.environ.get(ARTIFACT_ENV) or DEFAULT_ARTIFACT_DIR
    )
    os.makedirs(out_dir, exist_ok=True)

    parser = BaseKernelInfoParser(trace_file, enable_comm_calibration=False)

    # Basic sanity checks (non-fatal warnings)
    if not isinstance(parser.events, list) or len(parser.events) == 0:
        print("[WARN] No events parsed or events is not a list")
    if (
        not isinstance(parser.individual_info, list)
        or len(parser.individual_info) == 0
    ):
        print(
            "[WARN] No individual_info parsed or individual_info is not a list"
        )

    filtered = parser.get_aggregate_kernel_info()
    e2e_time = parser.get_kernel_e2e_time(parser.individual_info)

    # Export CSV (BaseKernelInfoParser decides filename; replicate test expectation)
    parser.save_individual_csv(out_dir)
    # Infer CSV path by replacing extension pattern as in unit test
    base_name = os.path.basename(trace_file)
    csv_name = base_name.replace(".json.gz", ".csv")
    csv_path = os.path.join(out_dir, csv_name)

    # Summary artifact
    ts = time.strftime("%Y%m%d-%H%M%S")
    summary_lines = [
        f"trace_file: {trace_file}",
        f"events_count: {len(parser.events)}",
        f"individual_info_count: {len(parser.individual_info)}",
        f"filtered_count: {len(filtered)}",
        f"kernel_e2e_time: {e2e_time}",
        f"csv_created: {csv_path if os.path.exists(csv_path) else 'MISSING'}",
        f"output_dir: {out_dir}",
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    summary_filename = f"parse_summary_{ts}.txt"
    _write_artifact(out_dir, summary_filename, summary_text)

    print("[INFO] Parse complete")
    for line in summary_lines:
        print("[INFO]", line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
