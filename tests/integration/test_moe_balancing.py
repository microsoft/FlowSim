import subprocess
import pytest
import os
import re
import time

from tests.utils import _write_artifact
from tests.utils import ARTIFACT_ENV, DEFAULT_ARTIFACT_DIR, SG_LANG_DIR


@pytest.mark.parametrize("tp", [8])
@pytest.mark.parametrize(
    "model_path",
    [
        "/workloadsim/workload/models/configs/deepseek",
    ],
)
def test_docker_image(tp, model_path):
    os.chdir(SG_LANG_DIR)
    cmd = [
        "python3",
        "-m",
        "sglang.bench_one_batch",
        "--model-path",
        model_path,
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
    child_env = os.environ.copy()
    child_env["SGLANG_BALANCED_MOE"] = "1"
    result = subprocess.run(cmd, capture_output=True, text=True, env=child_env)
    output = result.stdout + result.stderr
    # Write the output to an artifact file
    artifact_dir = os.environ.get(ARTIFACT_ENV, DEFAULT_ARTIFACT_DIR)
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"test_docker_image_tp{tp}_model_{os.path.basename(model_path)}_{ts}"
    stdout_path = _write_artifact(
        artifact_dir, f"{base}.stdout.txt", result.stdout or ""
    )
    stderr_path = _write_artifact(
        artifact_dir, f"{base}.stderr.txt", result.stderr or ""
    )
    # Check key phrases in the output
    assert (
        "Using balanced MoE with uniform topk ids. This is for benchmarking only!"
        in output
    ), "MoE balancing message not found in output"
