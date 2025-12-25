import subprocess
import pytest
import os
import re
import time

from tests.utils import _write_artifact
from tests.utils import ARTIFACT_ENV, DEFAULT_ARTIFACT_DIR, SG_LANG_DIR

MODEL_PATH = "/flowsim/workload/models/configs/deepseek"


@pytest.mark.parametrize("tp", [1, 2])
def test_docker_image(tp):
    os.chdir(SG_LANG_DIR)
    cmd = [
        "python3",
        "-m",
        "sglang.bench_one_batch",
        "--model-path",
        MODEL_PATH,
        "--load-format",
        "dummy",
        "--tp",
        str(tp),
        "--batch",
        "1",
        "--input-len",
        "128",
        "--output-len",
        "2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    # Write the output to an artifact file
    artifact_dir = os.environ.get(ARTIFACT_ENV, DEFAULT_ARTIFACT_DIR)
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"test_docker_image_tp{tp}_{ts}"
    stdout_path = _write_artifact(
        artifact_dir, f"{base}.stdout.txt", result.stdout or ""
    )
    stderr_path = _write_artifact(
        artifact_dir, f"{base}.stderr.txt", result.stderr or ""
    )
    # Check key phrases in the output
    assert "Warmup" in output, "Missing 'Warmup' in output"
    assert "Benchmark" in output, "Missing 'Benchmark' in output"
    # Check if numbers are after "Prefill"
    prefill_match = re.search(
        r"Prefill\. latency:\s*([\d\.eE+-]+)\s*s,\s*throughput:\s*([\d\.eE+-]+)\s*token/s",
        output,
    )
    assert (
        prefill_match is not None
    ), "No latency/throughput numbers found after 'Prefill'"

    # Check if numbers are after "Decode"
    decode_match = re.search(
        r"Decode\s+\d+\. Batch size:\s*\d+,\s*latency:\s*([\d\.eE+-]+)\s*s,\s*throughput:\s*([\d\.eE+-]+)\s*token/s",
        output,
    )
    assert (
        decode_match is not None
    ), "No latency/throughput numbers found after 'Decode'"
