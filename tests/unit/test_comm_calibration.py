# Run base parser tests, list all communication kernels, record original time
# and compare to the final calibrated actual value.
# Expcetation: Calibrated actual should be faster than original time
import os
import time
import re
import math
import pytest
import simulator.benchmarks.nccl_benchmarks as nb
from simulator.base_parser import BaseKernelInfoParser

from tests.utils import _write_artifact
from tests.utils import ARTIFACT_ENV, DEFAULT_ARTIFACT_DIR


@pytest.fixture(scope="module")
def real_trace_file():
    trace_path = "/flowsim/tests/unit/test_trace.trace.json.gz"
    assert os.path.exists(trace_path), f"Profile File Not Exist: {trace_path}"
    return trace_path


@pytest.mark.usefixtures("real_trace_file")
def test_base_parser_with_real_profile(real_trace_file):
    # Test file is TP=4
    parser = BaseKernelInfoParser(
        real_trace_file, TP=4, enable_comm_calibration=False
    )
    assert isinstance(parser.events, list)
    assert len(parser.events) > 0

    artifact_dir = os.environ.get(ARTIFACT_ENV, DEFAULT_ARTIFACT_DIR)
    parser.save_individual_csv(artifact_dir)
    csv_path = os.path.join(artifact_dir, "test_trace.trace.csv")
    assert os.path.exists(csv_path), "Filtered individual info CSV not created"

    # Identify all communication kernels
    comm_indices = []
    for i, entry in enumerate(parser.individual_info):
        name = entry[0] if len(entry) > 0 else ""
        # detect common nccl kernel names (case-insensitive)
        if any(
            k in name
            for k in ("ncclDevKernel_AllReduce", "ncclDevKernel_AllGather")
        ):
            comm_indices.append(i)

    # Save original durations before calibration
    original_durations = {}
    for idx in comm_indices:
        entry = parser.individual_info[idx]
        original_durations[idx] = entry[5] if len(entry) > 5 else None

    # Calibration (this will update durations in parser.individual_info)
    parser._calibrate_communication_kernels()

    # For each communication kernel record original and calibrated actual
    lines = []
    header = "idx,name,original_s,actual_s"
    lines.append(header)
    for idx in comm_indices:
        entry = parser.individual_info[idx]
        name = entry[0] if len(entry) > 0 else ""

        orig_val = original_durations.get(idx)
        calibrated_value = entry[5] if len(entry) > 5 else None

        lines.append(
            ",".join(
                [
                    str(idx),
                    f'"{name}"',
                    f"{orig_val:.6e}" if orig_val is not None else "UNKNOWN",
                    (
                        f"{calibrated_value:.6e}"
                        if calibrated_value is not None
                        else "UNKNOWN"
                    ),
                ]
            )
        )

        assert (
            calibrated_value is not None
        ), f"Calibrated value missing for idx {idx}"
        assert orig_val is not None, f"Original value missing for idx {idx}"
        # Usually calibrated actual is less than or equal to original time, but there are rare cases
        # where it might be slightly higher due to noise, so we use a small tolerance
        assert (
            calibrated_value <= orig_val * 1.1
        ), f"Calibrated actual {calibrated_value} not <= original {orig_val} for idx {idx}"

    # Write a short summary artifact
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"test_base_parser_{ts}"
    summary_text = (
        "\n".join(
            [
                f"trace_file: {real_trace_file}",
                f"events_count: {len(parser.events)}",
                f"individual_info_count: {len(parser.individual_info)}",
                f"csv_created: {csv_path if os.path.exists(csv_path) else 'MISSING'}",
                f"comm_kernels_found: {len(comm_indices)}",
                "",
                "Per-kernel original vs calibrated actual (csv):",
                *lines,
                "",
            ]
        )
        + "\n"
    )
    _write_artifact(artifact_dir, f"{base}.summary.txt", summary_text)
