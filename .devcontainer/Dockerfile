ARG NVARIANT=20.04
ARG CUDNN_VERSION=8
ARG CUDA_VERSION=11.3
ARG CUDA_VERSION_FULL=${CUDA_VERSION}.1
ARG from=nvidia/cuda:${CUDA_VERSION_FULL}-cudnn${CUDNN_VERSION}-devel-ubuntu${NVARIANT}
FROM ${from}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ARG INSTALL_ZSH="false"
ARG UPGRADE_PACKAGES="true"
ARG COMMON_SCRIPT_SOURCE="https://raw.githubusercontent.com/microsoft/vscode-dev-containers/v0.122.1/script-library/common-debian.sh"
ARG COMMON_SCRIPT_SHA="da956c699ebef75d3d37d50569b5fbd75d6363e90b3f5d228807cff1f7fa211c"

# Configure apt and install packages
RUN apt-get update && apt-get install --no-install-recommends -y \
      python3-flake8 pylint python3-bandit python3-mypy python3-pycodestyle python3-pydocstyle \
      black python3-yapf python3-autopep8 pycodestyle pydocstyle yapf3 mypy flake8 pipx \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*
  
RUN apt-get update \
    #
    # Verify git, common tools / libs installed, add/modify non-root user, optionally install zsh
    && apt-get -y install --no-install-recommends curl ca-certificates 2>&1 \
    && curl -sSL  ${COMMON_SCRIPT_SOURCE} -o /tmp/common-setup.sh \
    && ([ "${COMMON_SCRIPT_SHA}" = "dev-mode" ] || (echo "${COMMON_SCRIPT_SHA} /tmp/common-setup.sh" | sha256sum -c -)) \
    && /bin/bash /tmp/common-setup.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" \
    && rm /tmp/common-setup.sh \
    && apt-get -y install --no-install-recommends build-essential cmake cppcheck valgrind clang lldb llvm gdb 2>&1 \
    #
    # Clean ups
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install --no-install-recommends -y \
      wget curl git git-lfs \
      python3-pip python3-venv \
      libboost-all-dev \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*