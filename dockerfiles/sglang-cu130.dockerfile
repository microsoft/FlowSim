# FlowSim on top of an upstream SGLang runtime image.
#
# Why a second dockerfile: dockerfiles/cuda12.6.dockerfile builds everything from
# nvcr.io/nvidia/pytorch:24.10-py3 (CUDA 12.6), whose kernels are compiled for
# sm_90 and below. Blackwell parts (GB200/B200, sm_100+) need CUDA 13, and
# building SGLang + FlowInfer + DeepGEMM from source for aarch64 is slow and
# fragile. Instead this image starts from the official SGLang release image,
# which already ships matching CUDA 13 / FlashInfer / DeepGEMM builds for both
# amd64 and arm64, and only layers FlowSim on top.
#
# The base image pins the same SGLang commit as the workload/framework/sglang
# submodule, and installs it in editable mode from /sgl-workspace/sglang, so the
# FlowSim tracing patch can be applied directly to that checkout.
#
# Build:
#   make build-docker-sglang
#   # or
#   docker build -t flowsim-image -f dockerfiles/sglang-cu130.dockerfile .
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.16-cu130
FROM ${SGLANG_IMAGE}

LABEL maintainer="FlowSim"

ENV DEBIAN_FRONTEND=noninteractive

# No apt/pip install step on purpose. GPU hosts frequently build without egress,
# and everything needed is already in the base image:
#   * git + patch          -> applying the FlowSim tracing patch
#   * requests/numpy/pandas/PyYAML -> the profile + trace-parsing code paths
# `perfetto` is listed in pyproject.toml but is not imported by scripts/ or
# simulator/, so it is not required to produce the kernel CSVs. Pass extra
# packages when the builder does have egress, e.g.
#   docker build --build-arg PIP_EXTRA_INSTALL="perfetto scalesim scipy" ...
ARG PIP_EXTRA_INSTALL=""
RUN if [ -n "${PIP_EXTRA_INSTALL}" ]; then \
        pip install --no-cache-dir ${PIP_EXTRA_INSTALL}; \
    fi && \
    command -v git >/dev/null && command -v patch >/dev/null && \
    python -c "import requests, numpy, pandas, yaml; print('FlowSim python deps OK')"

# Copy FlowSim itself. The sglang submodule is deliberately not copied: this
# image uses the SGLang checkout that ships in the base image.
COPY scripts /flowsim/scripts
COPY simulator /flowsim/simulator
COPY schedulers /flowsim/schedulers
COPY utils /flowsim/utils
COPY backend /flowsim/backend
COPY tests /flowsim/tests
COPY workload/models /flowsim/workload/models
COPY workload/framework/patches /flowsim/workload/framework/patches
COPY kernels.json pyproject.toml README.md /flowsim/

# Apply the FlowSim kernel-tracing hooks to the SGLang install in the base
# image. Fails the build if the base image drifts from the pinned submodule.
ARG SGLANG_SRC=/sgl-workspace/sglang
RUN cd ${SGLANG_SRC} && \
    git apply --verbose /flowsim/workload/framework/patches/hook-v0516.patch && \
    python -c "from sglang.srt.tracing.hook_register import register_kernels_for_profiling; print('FlowSim tracing hooks installed')"

WORKDIR /flowsim
ENV PYTHONPATH=/flowsim
CMD ["/bin/bash"]
