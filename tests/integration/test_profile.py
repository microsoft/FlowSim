import subprocess
import sys
import os
import time
import socket
import signal
import shutil


def wait_for_port(host, port, timeout=60):
    """Wait for the port to open, up to timeout seconds"""
    for _ in range(timeout):
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except Exception:
            time.sleep(1)
    return False


def test_bench_serving_predefined_len_profile():
    # Set environment variables
    env = os.environ.copy()
    profile_dir = "/workloadsim/server_profile"
    env["SGLANG_TORCH_PROFILER_DIR"] = profile_dir
    env["SGLANG_PROFILE_KERNELS"] = "1"
    env["SGLANG_PROFILE_DEBUG"] = "1"
    env["SGLANG_SET_CPU_AFFINITY"] = "1"

    # Clean profile_dir
    if os.path.exists(profile_dir):
        for f in os.listdir(profile_dir):
            fp = os.path.join(profile_dir, f)
            if os.path.isfile(fp) or os.path.islink(fp):
                os.unlink(fp)
            elif os.path.isdir(fp):
                shutil.rmtree(fp)
    else:
        os.makedirs(profile_dir)

    # Prepare server log files (Scheme A: redirect to files instead of PIPE)
    artifacts_dir = "/workloadsim/tests/test-artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    ts = int(time.time())
    server_stdout_path = os.path.join(artifacts_dir, f"server_{ts}.stdout.log")
    server_stderr_path = os.path.join(artifacts_dir, f"server_{ts}.stderr.log")
    server_stdout_f = open(server_stdout_path, "w")
    server_stderr_f = open(server_stderr_path, "w")

    # Start the server (stdout/stderr redirected to files for later inspection)
    server_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            "/workloadsim/workload/models/configs/deepseek/",
            "--load-format",
            "dummy",
            "--tp",
            "4",
            # "--data-parallel-size", "4",
            # "--enable-dp-attention",
            # "--enable-dp-lm-head",
            # "--expert-parallel-size", "4",
            "--attention-backend",
            "flashinfer",
            # "--disable-radix-cache",
            "--host",
            "0.0.0.0",
            "--port",
            "30001",
            "--disable-cuda-graph",
        ],
        cwd="/workloadsim/workload/framework/sglang/python",
        stdout=server_stdout_f,
        stderr=server_stderr_f,
        preexec_fn=os.setsid,
        env={**env, "CUDA_VISIBLE_DEVICES": "4,5,6,7"},
    )
    try:
        assert wait_for_port(
            "0.0.0.0", 30001, timeout=600
        ), "Server did not start in time"

        script = os.path.abspath(
            "/workloadsim/workload/framework/sglang/python/sglang/bench_serving.py"
        )
        args = [
            sys.executable,
            script,
            "--backend",
            "sglang",
            "--host",
            "0.0.0.0",
            "--port",
            "30001",
            "--dataset-name",
            "defined-len",
            "--prefill-decode-lens",
            "32768:8",
            "--num-prompts",
            "16",
            "--profile",
        ]
        result = subprocess.run(
            args, capture_output=True, text=True, env=env, timeout=1200
        )
        # Store logs to test-artifacts directory for debugging regardless of success
        artifacts_dir = "/workloadsim/tests/test-artifacts"
        try:
            os.makedirs(artifacts_dir, exist_ok=True)
            ts = int(time.time())
            prefix = f"bench_serving_predefined_len_{ts}"
            with open(
                os.path.join(artifacts_dir, prefix + ".stdout.log"), "w"
            ) as f_out:
                f_out.write(result.stdout)
            with open(
                os.path.join(artifacts_dir, prefix + ".stderr.log"), "w"
            ) as f_err:
                f_err.write(result.stderr)
        except Exception:
            # Do not fail test due to artifact write issues
            pass
        assert "Profiler started" in result.stdout
        assert "Profiler stopped" in result.stdout
        assert result.returncode == 0
        files = os.listdir(profile_dir)
        json_gz_files = [f for f in files if f.endswith(".trace.json.gz")]
        assert len(json_gz_files) > 0, "No .trace.json.gz profile files found"
    finally:
        try:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            server_proc.wait(timeout=10)
        except Exception:
            pass
        # Ensure server logs flushed & closed
        try:
            server_stdout_f.flush()
            server_stderr_f.flush()
        except Exception:
            pass
        try:
            server_stdout_f.close()
            server_stderr_f.close()
        except Exception:
            pass
        time.sleep(2)  # Give some time for the port to be released
