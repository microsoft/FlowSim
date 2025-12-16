import os
import time
import pytest
from simulator.base_parser import BaseKernelInfoParser

from tests.utils import _write_artifact
from tests.utils import ARTIFACT_ENV, DEFAULT_ARTIFACT_DIR


@pytest.fixture(scope="module")
def real_trace_file():
    trace_path = "/workloadsim/tests/unit/test_trace.trace.json.gz"
    assert os.path.exists(trace_path), f"Profile File Not Exisit: {trace_path}"
    return trace_path


def test_base_parser_with_real_profile(real_trace_file):
    parser = BaseKernelInfoParser(real_trace_file)
    # Check if events are parsed correctly
    assert isinstance(parser.events, list)
    assert len(parser.events) > 0

    # Check if individual_info is parsed correctly
    assert isinstance(parser.individual_info, list)
    assert len(parser.individual_info) > 0

    # Check get_aggregate_kernel_info function call
    filtered = parser.get_aggregate_kernel_info()
    assert isinstance(filtered, list)
    assert len(filtered) > 0

    # Check get_kernel_e2e_time function call
    e2e = parser.get_kernel_e2e_time(parser.individual_info)
    assert isinstance(e2e, (int, float))
    assert e2e > 0

    # Check save_individual_csv function call
    artifact_dir = os.environ.get(ARTIFACT_ENV, DEFAULT_ARTIFACT_DIR)
    parser.save_individual_csv(artifact_dir)
    csv_path = os.path.join(artifact_dir, "test_trace.trace.csv")
    assert os.path.exists(csv_path), "Filtered individual info CSV not created"

    # Write to artifact directory
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"test_base_parser_{ts}"

    summary_lines = [
        f"trace_file: {real_trace_file}",
        f"events_count: {len(parser.events)}",
        f"individual_info_count: {len(parser.individual_info)}",
        f"filtered_count: {len(filtered)}",
        f"kernel_e2e_time: {e2e}",
        f"csv_created: {csv_path if os.path.exists(csv_path) else 'MISSING'}",
    ]
    summary_text = "\n".join(summary_lines) + "\n"

    _write_artifact(artifact_dir, f"{base}.summary.txt", summary_text)


def test_parse_annotation_name_new_format():
    dims, dtypes, names = (
        BaseKernelInfoParser._parse_dims_and_types_from_annotation_name(
            "moe.fused|hidden_states[4x16:float32],router_logits[4x8:float32]"
        )
    )
    assert dims == [[4, 16], [4, 8]]
    assert dtypes == ["float32", "float32"]
    assert names == ["hidden_states", "router_logits"]
