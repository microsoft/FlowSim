# Workload Simulation Pipeline
## Overview

Workload Simulation Pipeline is a toolchain for simulating and analyzing large-scale operator performance, such as GEMM and GEMV, with a focus on real-world inference workloads. The pipeline consists of three main stages:

1. **Profiling**: Extracts real traces from existing inference frameworks (e.g., [sglang](https://docs.sglang.ai/)), capturing operator-level execution from actual workloads.
2. **Trace Translation**: Converts the profiled traces (e.g., PyTorch traces) into detailed kernel-level traces, enriching them with fine-grained information such as tensor shapes and data types.
3. **Simulation**: Feeds the translated traces into hardware simulators (e.g., LLMCompass) to enable hardware-level simulation, calibration, and prediction of system configurations.

> **Note:** While the simulation is performed at the operator level, performance analysis and observation are conducted end-to-end (E2E) to provide a holistic view of workload behavior.

The project supports rapid deployment using Docker, includes scripts for environment setup and profiling, and offers flexible configuration options.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Build and Run](#docker-build-and-run)
- [Profiling and Simulation](#profiling-and-simulation)
- [Tests](#tests)

---

## Prerequisites

- Recommended: Linux system
- Required: [Docker](https://docs.docker.com/get-docker/), [Make](https://www.gnu.org/software/make/) and [NVidia NGC Account](https://org.ngc.nvidia.com/setup/api-key) for pulling nvidia docker.

---

## Docker Build and Run

From the project root directory, build and launch the Docker container. Initializing git submodules is necessary, as the repository requires a Personal Access Token (PAT) for cloning.

```bash
make build-docker
make run-docker GPU_DEVICES=[xxx] MOUNT_VOLUME=[y/n] CONTAINER_NAME=[YourContainerName]
```

- `make build-docker`: Builds the Docker image using the provided Dockerfile. Run `git submodule update --init --recursive` first, since the LLMCompass submodule requires PAT for initialization.
- `make run-docker GPU_DEVICES=all`: Starts the container interactively. Use `MOUNT_VOLUME=y` for developing purpose to easily download trace files. By default the container name is `workloadsim-docker`
- `make rm-docker`: Remove the docker after it stops.

---

## Profiling and Simulation

### Quick Start

See the [sglang documentation](https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html) for single-node online profiling reference scripts.

---

## Tests

Tests are located in the `tests` directory. To execute tests, run `pytest <test file>` in the project root folder to generate results.