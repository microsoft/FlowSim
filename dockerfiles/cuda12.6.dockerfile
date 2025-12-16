FROM nvcr.io/nvidia/pytorch:24.10-py3

# OS:
#   - Ubuntu: 22.04
#   - OpenMPI: 4.1.7
#   - Docker Client: 20.10.8
# NVIDIA:
#   - CUDA: 12.6.2
#   - cuBLAS: 12.6.3.3
#   - cuDNN: 9.5.0.50
#   - NCCL: 2.22.3
#   - TransformerEngine 1.11
# Mellanox:
#   - OFED: 23.07-0.5.1.2
#   - HPC-X: 2.20
# Intel:
#   - mlc: v3.11

# Note: dockerfile modifed based on
# https://github.com/microsoft/superbenchmark/blob/main/dockerfile/cuda12.4.dockerfile

LABEL maintainer="WorkloadSim"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    bc \
    build-essential \
    curl \
    dmidecode \
    ffmpeg \
    git \
    iproute2 \
    jq \
    libaio-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libboost-program-options-dev \
    libcap2 \
    libcurl4-openssl-dev \
    libnuma-dev \
    libpci-dev \
    libswresample-dev \
    libtinfo5 \
    libtool \
    lshw \
    net-tools \
    openssh-client \
    openssh-server \
    pciutils \
    sudo \
    util-linux \
    vim \
    wget \
    && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

RUN pip install perfetto scalesim matplotlib seaborn scipy cuda-bindings==12.9.0

ARG NUM_MAKE_JOBS=
ARG TARGETPLATFORM
ARG TARGETARCH

# Install Docker
ENV DOCKER_VERSION=20.10.8
RUN TARGETARCH_HW=$(uname -m) && \
    wget -q https://download.docker.com/linux/static/stable/${TARGETARCH_HW}/docker-${DOCKER_VERSION}.tgz -O docker.tgz && \
    tar --extract --file docker.tgz --strip-components 1 --directory /usr/local/bin/ && \
    rm docker.tgz

# Update system config
RUN mkdir -p /root/.ssh && \
    touch /root/.ssh/authorized_keys && \
    mkdir -p /var/run/sshd && \
    sed -i "s/[# ]*PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config && \
    sed -i "s/[# ]*PermitUserEnvironment no/PermitUserEnvironment yes/" /etc/ssh/sshd_config && \
    sed -i "s/[# ]*Port.*/Port 22/" /etc/ssh/sshd_config && \
    echo "* soft nofile 1048576\n* hard nofile 1048576" >> /etc/security/limits.conf && \
    echo "root soft nofile 1048576\nroot hard nofile 1048576" >> /etc/security/limits.conf

# Install OFED
ENV OFED_VERSION=23.07-0.5.1.2
RUN TARGETARCH_HW=$(uname -m) && \
    cd /tmp && \
    wget -q https://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu22.04-${TARGETARCH_HW}.tgz && \
    tar xzf MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu22.04-${TARGETARCH_HW}.tgz && \
    MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu22.04-${TARGETARCH_HW}/mlnxofedinstall --user-space-only --without-fw-update --without-ucx-cuda --force --all && \
    rm -rf /tmp/MLNX_OFED_LINUX-${OFED_VERSION}*

# Install HPC-X
ENV HPCX_VERSION=v2.18
RUN TARGETARCH_HW=$(uname -m) && \
    cd /opt && \
    rm -rf hpcx && \
    wget https://content.mellanox.com/hpc/hpc-x/${HPCX_VERSION}/hpcx-${HPCX_VERSION}-gcc-mlnx_ofed-ubuntu22.04-cuda12-${TARGETARCH_HW}.tbz -O hpcx.tbz && \
    tar xf hpcx.tbz && \
    mv hpcx-${HPCX_VERSION}-gcc-mlnx_ofed-ubuntu22.04-cuda12-${TARGETARCH_HW} hpcx && \
    rm hpcx.tbz

# Deprecated - Installs specific to amd64 platform
# RUN if [ "$TARGETARCH" = "amd64" ]; then \
#     # Install Intel MLC
#     cd /tmp && \
#     wget -q https://downloadmirror.intel.com/793041/mlc_v3.11.tgz -O mlc.tgz && \
#     tar xzf mlc.tgz Linux/mlc && \
#     cp ./Linux/mlc /usr/local/bin/ && \
#     rm -rf ./Linux mlc.tgz && \
#     # Install AOCC compiler
#     wget https://download.amd.com/developer/eula/aocc-compiler/aocc-compiler-4.0.0_1_amd64.deb && \
#     apt install -y ./aocc-compiler-4.0.0_1_amd64.deb && \
#     rm -rf aocc-compiler-4.0.0_1_amd64.deb && \
#     # Install AMD BLIS
#     wget https://download.amd.com/developer/eula/blis/blis-4-0/aocl-blis-linux-aocc-4.0.tar.gz && \
#     tar xzf aocl-blis-linux-aocc-4.0.tar.gz && \
#     mv amd-blis /opt/AMD && \
#     rm -rf aocl-blis-linux-aocc-4.0.tar.gz; \
#     else \
#     echo "Skipping Intel MLC, AOCC and AMD Bliss installations for non-amd64 architecture: $TARGETARCH"; \
#     fi

# Install UCX v1.16.0 with multi-threading support
RUN cd /tmp && \
    wget https://github.com/openucx/ucx/releases/download/v1.16.0/ucx-1.16.0.tar.gz && \
    tar xzf ucx-1.16.0.tar.gz && \
    cd ucx-1.16.0 && \
    ./contrib/configure-release-mt --prefix=/usr/local && \
    make -j ${NUM_MAKE_JOBS} && \
    make install

ENV PATH="${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:/usr/local/mpi/lib:${LD_LIBRARY_PATH}" \
    IW_HOME=/opt/infrawise

RUN echo PATH="$PATH" > /etc/environment && \
    echo LD_LIBRARY_PATH="$LD_LIBRARY_PATH" >> /etc/environment && \
    echo IW_MICRO_PATH="$IW_MICRO_PATH" >> /etc/environment && \
    echo "source /opt/hpcx/hpcx-init.sh && hpcx_load" | tee -a /etc/bash.bashrc >> /etc/profile.d/10-hpcx.sh

# Install DeepGEMM
RUN cd /workspace && \
    git clone https://github.com/deepseek-ai/DeepGEMM && \
    cd DeepGEMM && \
    git checkout 03d0be3d2d03b6eed3c99d683c0620949a13a826 && \
    git submodule update --init --recursive && \
    python setup.py install

# Copy local workloadsim code into container
COPY . /workloadsim

WORKDIR /workloadsim
# Init sglang submodule
RUN git submodule update --init --recursive

# Install sglang
WORKDIR /workloadsim/workload/framework/sglang
RUN python3 -m pip install -e "python[all]"

# Build nccl-tests
WORKDIR /workloadsim/third_party/nccl-tests
RUN make -j ${NUM_MAKE_JOBS}

WORKDIR /workloadsim
CMD ["/bin/bash"]