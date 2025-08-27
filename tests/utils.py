import os

ARTIFACT_ENV = "PYTEST_ARTIFACT_DIR"
DEFAULT_ARTIFACT_DIR = os.path.join(os.getcwd(), "test-artifacts")
SG_LANG_DIR = "/workloadsim/workload/framework/sglang/python/sglang"


def _write_artifact(artifact_dir: str, name: str, content: str):
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
