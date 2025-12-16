# FlowSim: Workload Simulation Pipeline

FlowSim is a lightweight and extensible toolchain for simulating and analyzing kernel-level performance based on real-world inference workloads. It bridges high-level framework profiling and low-level hardware simulation through a three-stage workflow:

1. **Profiling**: Extracts real AI workload traces from optimized inference frameworks (e.g., [sglang](https://docs.sglang.ai/)), capturing operator-level execution from production-like workloads.
2. **Trace Translation**: Converts profiled pytorch traces into detailed kernel-level representations enriched with fine-grained tensor information such as shapes and data types.
3. **Simulation**: Feeds the translated traces into hardware-level simulators (e.g., LLMCompass) to enable GPU kernel simulation, simulator calibration, and performance prediction.

Although simulation is driven at the operator/kernel level, performance analysis and observation are performed end-to-end (E2E) to provide a holistic view of workload behavior.

FlowSim is most suitable for:

- Developers who need detailed, fine-grained end-to-end workload profiling.
- Researchers and simulator developers who require accurate kernel-level traces to calibrate and evaluate GPU performance simulators.

The project supports rapid deployment using Docker, includes scripts for environment setup and profiling, and offers flexible configuration options.

---

## Table of Contents

- [Quick Demo](#quick-demo)
- [Prerequisites](#prerequisites)
- [Docker Build and Run](#docker-build-and-run)
- [Profiling and Simulation](#profiling-and-simulation)
- [Tests](#tests)
- [Risks and limitations](#risks-and-limitations)
- [License](#license)
- [Trademarks](#trademarks)

---


## Quick Demo

This example walks through profiling, parsing, and simulating a real workload.

1. **Build and run the Docker image (on the host with GPU)**

```bash
make build-docker
make run-docker GPU_DEVICES=all MOUNT_VOLUME=y CONTAINER_NAME=flowsim-demo
```

2. **Apply FlowSim patches to the bundled sglang**

```bash
cd /workloadsim/workload/framework/sglang
git apply ../patches/hook.patch
git apply ../patches/v055.patch
cd /workloadsim
```

3. **Generate (or reuse) a trace (optionally translate to CSV + summary)**

Preferred path: run the integration profiling test to generate a fresh trace under `/workloadsim/server_profile`:

```bash
# 3.a Generate a new trace via profiling (GPU required)
pytest tests/integration/test_profile.py::test_bench_serving_predefined_len_profile

# 3.b (optional) Translate the trace to CSV + summary for offline analysis
python scripts/run_parse.py \
  --trace-file server_profile/your-trace-name-TP-0.trace.json.gz \
  --output-dir server_simulate
```

Fallback: if you cannot run profiling (e.g., no GPU), reuse the demo trace shipped with the repo instead (both for CSV translation and for step 4 simulation):

```bash
python scripts/run_parse.py \  
	--trace-file demo/deepseekv3-TP-0.trace.json.gz \  
	--output-dir server_simulate
```

These steps:

- Use sglang to produce or reuse a real profile trace under `server_profile/` or `demo/`.
- Optionally use `BaseKernelInfoParser` (via `run_parse.py`) to extract kernel-level information and write a per-kernel CSV plus a summary file into `server_simulate/`.

You can then inspect the generated artifacts in the corresponding folder.

4. **Run a lightweight simulation via the LLMCompass backend**

With a trace from step 3 (or the demo directory), you can run a small hardware-level simulation using the LLMCompass backend integration test. This test starts a local backend server and parses the trace internally before posting kernels to it; the CSV from step 3 is not required.

```bash
# Optionally point the simulator to the trace you just generated or reused
export TRACE_PATH=/workloadsim/path-to-your-trace.trace.json.gz

# Run the LLMCompass backend integration test
pytest tests/unit/test_llmcompass_backend.py::test_post_parsed_kernels_to_backend
```

If `TRACE_PATH` is not set, the test falls back to the bundled sample trace `tests/unit/test_trace.trace.json.gz`.

Together, steps 1–4 illustrate the core FlowSim workflow: **profile → parse/translate → simulate/analyze**.
---

## Prerequisites

- Recommended: Linux system
- Required: [Docker](https://docs.docker.com/get-docker/), [Make](https://www.gnu.org/software/make/) and an [NVIDIA NGC account](https://org.ngc.nvidia.com/setup/api-key) for pulling NVIDIA Docker images.

---

## Docker Build and Run

From the project root directory, build and launch the Docker container. Initializing git submodules is necessary, as the repository requires a Personal Access Token (PAT) for cloning.

```bash
make build-docker
make run-docker GPU_DEVICES=[xxx] MOUNT_VOLUME=[y/n] CONTAINER_NAME=[YourContainerName]
```

- `make build-docker`: Builds the Docker image using the provided Dockerfile. Run `git submodule update --init --recursive` first, since the LLMCompass submodule requires a PAT for initialization.
- `make run-docker GPU_DEVICES=all`: Starts the container interactively. Use `MOUNT_VOLUME=y` for development purposes to easily download trace files. By default, the container name is `workloadsim-docker`.
- `make rm-docker`: Removes the Docker container after it stops.

---

## Profiling and Simulation

### Quick Start

For a concrete end-to-end profiling setup in this repo, see `tests/integration/test_profile.py`. It demonstrates how to:

- Launch an sglang server with profiling enabled (via `sglang.launch_server` and environment variables such as `SGLANG_TORCH_PROFILER_DIR` and `SGLANG_PROFILE_KERNELS`).
- Run `python sglang/bench_serving.py --profile ...` against that server to generate `.trace.json.gz` files under `/workloadsim/server_profile`.

These trace files can then be translated and simulated following the [Quick Demo](#quick-demo) section.

For more background on profiling options and parameters, refer to the [sglang profiling documentation](https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html).

### Simulation characteristics (LLMCompass backend)

FlowSim currently integrates with [LLMCompass](https://github.com/TerrenceZhangX/LLMCompass) as a reference GPU performance simulator. In this setup:

- Each parsed kernel (from a FlowSim trace) is turned into a small JSON payload and submitted to the LLMCompass backend via its `/tasks` API.
- The backend estimates runtime characteristics per kernel (e.g., latency) under a user-specified hardware configuration.
- Results are polled asynchronously until all tasks reach a terminal state, then written as JSON artifacts for further analysis.

LLMCompass itself supports richer workflows (e.g., compiling full operator graphs, system-level roofline analysis, and running graphs on real GPUs). FlowSim focuses on the **kernel-level, trace-driven** usage: taking end-to-end traces from real inference workloads and feeding them into a calibrated backend to study per-kernel performance, compare hardware configurations, or validate simulator behavior.

### Kernel metadata and unknown kernels

After you obtain a profile trace (`*.trace.json.gz`), you will typically run the parser once to inspect kernel-level status:

```bash
python scripts/run_parse.py \
	--trace-file /workloadsim/server_profile/your-trace-name.trace.json.gz \
	--output-dir /workloadsim/server_simulate
```

During parsing, FlowSim looks up kernel metadata (e.g., tensor shapes and dtypes) in `kernels.json`. Any kernels it cannot match are written to `unknown_kernels.json` at the project root, with incomplete or `unknown` parameter descriptions.

To enrich metadata for new or unsupported kernels:

- Open `unknown_kernels.json`, locate the entries of interest, and fill in the missing information (e.g., `operation`, `params[*].role`, `example_dim`, `example_dtype`, `description`).
- Copy the completed entries into `kernels.json` to make them part of the known-kernel database.
- Re-run `scripts/run_parse.py` on your trace; those kernels should now be treated as known and will no longer appear in `unknown_kernels.json`.

Tensor shapes and dtypes for Triton kernels are surfaced via the FlowSim tracing hooks. When `SGLANG_PROFILE_KERNELS=1`, `sglang.launch_server` calls `register_kernels_for_profiling` from `sglang.srt.tracing.hook_register`, which attaches tensor metadata to PyTorch profiler labels for registered kernels. If you introduce custom Triton kernels that still appear as "unknown" after parsing, you may need to extend this registration logic and/or add corresponding entries to `kernels.json`.


---

## Risks and limitations

FlowSim was not designed or evaluated for all possible downstream purposes. Users should consider its inherent limitations when selecting use cases and must evaluate and mitigate accuracy, safety, and fairness concerns specific to each intended downstream use.

---

## License

This project is released under the MIT License. For the full license text, see the `LICENSE` file in the repository root.

---

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.

Any use of third-party trademarks or logos is subject to those third parties' trademark and brand policies.