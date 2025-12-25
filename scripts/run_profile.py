#!/usr/bin/env python
"""Simplified sglang profiling script for Docker/Kubernetes.

This script launches sglang server + bench_serving for profiling.
All server and benchmark parameters are passed via --server-opts and --bench-opts.

Example:
  python scripts/run_profile.py \\
    --profile-dir /flowsim/server_profile \\
    --server-opts "--model-path /path/to/model --tp 4 --load-format dummy --host 0.0.0.0 --port 30001 --disable-cuda-graph" \\
    --bench-opts "--backend sglang --host 0.0.0.0 --port 30001 --dataset-name defined-len --num-prompts 16 --profile"
"""

import argparse
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Optional


def wait_for_port(host: str, port: int, timeout: int = 600) -> bool:
    """Wait until a TCP port becomes reachable."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except Exception:
            time.sleep(1)
    return False


def clean_dir(path: str) -> None:
    """Clean or create a directory."""
    if os.path.exists(path):
        for name in os.listdir(path):
            fp = os.path.join(path, name)
            if os.path.isfile(fp) or os.path.islink(fp):
                os.unlink(fp)
            elif os.path.isdir(fp):
                shutil.rmtree(fp)
    else:
        os.makedirs(path, exist_ok=True)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run sglang profiling workload")

    p.add_argument(
        "--profile-dir",
        default="/flowsim/server_profile",
        help="Directory where profiler traces (.trace.json.gz) will be written",
    )
    p.add_argument(
        "--log-dir",
        default="/flowsim/tests/test-artifacts",
        help="Directory to write server/client logs",
    )
    p.add_argument(
        "--server-opts",
        required=True,
        help=(
            "All options for sglang.launch_server (include --host, --port, --model-path, --tp, etc). "
            "Example: '--model-path /path --tp 1 --host 0.0.0.0 --port 30001 --disable-cuda-graph'"
        ),
    )
    p.add_argument(
        "--bench-opts",
        required=True,
        help=(
            "All options for bench_serving.py (include --backend, --host, --port, --dataset-name, --profile, etc). "
            "Example: '--backend sglang --host 0.0.0.0 --port 30001 --dataset-name defined-len --num-prompts 16 --profile'"
        ),
    )
    p.add_argument(
        "--bench-timeout",
        type=int,
        default=1200,
        help="Timeout in seconds for bench_serving.py",
    )

    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    profile_dir = args.profile_dir
    log_dir = args.log_dir

    clean_dir(profile_dir)
    os.makedirs(log_dir, exist_ok=True)

    ts = int(time.time())
    server_stdout_path = os.path.join(log_dir, f"server_{ts}.stdout.log")
    server_stderr_path = os.path.join(log_dir, f"server_{ts}.stderr.log")
    server_stdout_f = open(server_stdout_path, "w", encoding="utf-8")
    server_stderr_f = open(server_stderr_path, "w", encoding="utf-8")

    # Set profiling environment variables
    env = os.environ.copy()
    env["SGLANG_TORCH_PROFILER_DIR"] = profile_dir
    env["SGLANG_PROFILE_KERNELS"] = "1"
    env["SGLANG_PROFILE_DEBUG"] = "1"
    env["SGLANG_SET_CPU_AFFINITY"] = "1"

    # Extract host and port from server-opts for connection check
    server_args = shlex.split(args.server_opts)
    host = "0.0.0.0"
    port = 30001
    try:
        if "--host" in server_args:
            host = server_args[server_args.index("--host") + 1]
        if "--port" in server_args:
            port = int(server_args[server_args.index("--port") + 1])
    except (ValueError, IndexError):
        pass

    # Build server command
    launch_cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
    ] + server_args

    print("[INFO] Starting sglang server:", " ".join(launch_cmd), flush=True)
    preexec = getattr(os, "setsid", None)
    server_proc = subprocess.Popen(
        launch_cmd,
        cwd="/flowsim/workload/framework/sglang/python",
        stdout=server_stdout_f,
        stderr=server_stderr_f,
        preexec_fn=preexec,
        env=env,
    )

    try:
        if not wait_for_port(host, port, timeout=600):
            print(
                "[ERROR] Server did not start within timeout", file=sys.stderr
            )
            return 1

        script = os.path.abspath(
            "/flowsim/workload/framework/sglang/python/sglang/bench_serving.py"
        )

        bench_args = shlex.split(args.bench_opts)
        client_args = [sys.executable, script] + bench_args

        print(
            "[INFO] Running bench_serving:", " ".join(client_args), flush=True
        )
        result = subprocess.run(
            client_args,
            capture_output=True,
            text=True,
            env=env,
            timeout=args.bench_timeout,
        )

        ts2 = int(time.time())
        prefix = f"bench_serving_{ts2}"
        client_stdout_path = os.path.join(log_dir, prefix + ".stdout.log")
        client_stderr_path = os.path.join(log_dir, prefix + ".stderr.log")
        with open(client_stdout_path, "w", encoding="utf-8") as f_out:
            f_out.write(result.stdout)
        with open(client_stderr_path, "w", encoding="utf-8") as f_err:
            f_err.write(result.stderr)

        if result.returncode != 0:
            print(
                f"[ERROR] bench_serving exited with code {result.returncode}",
                file=sys.stderr,
            )
            return result.returncode

        files = os.listdir(profile_dir)
        json_gz_files = [f for f in files if f.endswith(".trace.json.gz")]
        if not json_gz_files:
            print(
                f"[ERROR] No .trace.json.gz files found in {profile_dir}",
                file=sys.stderr,
            )
            return 1

        print(
            f"[INFO] Profiling complete, found {len(json_gz_files)} trace file(s) in {profile_dir}",
            flush=True,
        )
        return 0
    finally:
        try:
            if server_proc.poll() is None:
                try:
                    os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
                except Exception:
                    server_proc.terminate()
            server_proc.wait(timeout=30)
        except Exception:
            pass
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
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
