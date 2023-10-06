# syntax=docker/dockerfile:1.3
# NB: at least 1.3 is needed to benefit from ARG expansion in bind mount arguments

# This Dockerfile can be built from scratch with e.g. the command
#
#     DOCKER_BUILDKIT=1 docker build . -t numba_dpex_dev
#
# The ARGs in the header of this file can be changed as needed (either by
# editing the file or with the command line).
# The build can be fairly long (up to one hour on low-end machines).
# To start a container, use e.g.:
#
#      sudo docker run --rm --name my_container -it -v /my/host/volume/:/mounted/volume --device=/dev/dri numba_dpex_dev
#
# The --device=/dev/dri option enables GPU passthrough (as long as the user
# has permissions, e.g. by being a member of the video group).
#
# The image is publicly available on dockerhub with the tag jjerphan/numba_dpex_dev:latest .

# The `ubuntu2004_intel_drivers` base ships an up-to-date installation of the
# intel runtime, including drivers and opencl and level_zero runtimes.
#
# See available versions here: https://github.com/intel/llvm/pkgs/container/llvm%2Fubuntu2204_intel_drivers
#
# TODO: make a custom build with other base images to test compatibility
# with other OSes.
ARG BASE_IMAGE_VERSION=latest-57b016e6e031950e5e5abda2d69afc355e7c7b61
ARG BASE_IMAGE_TAG=ghcr.io/intel/llvm/ubuntu2204_intel_drivers

ARG BASE=${BASE_IMAGE_TAG}:${BASE_IMAGE_VERSION}

# Various resources about the SYCL toolchain
# Environment variables with important effects on the SYCL toolchain:
# https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter
# Overview of the SYCL build instructions and environment requirements: https://intel.github.io/llvm-docs/GetStartedGuide.html
# More about sycl docker image Dockerfiles: https://github.com/intel/llvm/tree/sycl/devops/containers
# and https://github.com/intel/llvm/blob/sycl/sycl/doc/developer/DockerBKMs.md

# FIXME: The base image is missing a package required for enabling the `level_zero`
# backend, see https://github.com/intel/llvm/issues/6342
# Until it is fixed upstream we need to explicitly install the oneAPI level_zero loader.
ARG L0_LOADER_DEB_VERSION=1.14.0
ARG L0_LOADER_DEB_PLATFORM=u22.04

ARG INTEL_PYPI_URL=https://pypi.anaconda.org/intel/simple
ARG BASE_PYPI_URL=https://pypi.org/simple

# So far, some of the packages might not be compatible with python>=3.11
ARG PYTHON_VERSION=3.10.13


# This link might need to be refreshed occasionally.
# It can be found at https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=webdownload&options=online
# The installer CLI is documented here: https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-with-command-line.html#install-with-command-line_interactive

ARG ONEAPI_INSTALLER_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/992857b9-624c-45de-9701-f6445d845359
ARG ONEAPI_INSTALL_BINARY_NAME=l_BaseKit_p_2023.2.0.49397.sh

ARG ONEAPI_INSTALL_DIR=/opt/intel/oneapi

# Some build steps require a high enough version of cmake.
# Bump it if necessary.

ARG CMAKE_VERSION=3.27
ARG CMAKE_VERSION_BUILD=6


# Versions of the intel python packages

ARG DPCTL_GIT_BRANCH=0.15.1dev0
ARG DPCTL_GIT_URL=https://github.com/IntelPython/dpctl.git

ARG DPNP_GIT_BRANCH=0.13.1dev0
ARG DPNP_GIT_URL=https://github.com/IntelPython/dpnp.git

# `numba_dpex` often requires to install from arbitrary git commits, so the
# full history of the given branch will be cloned (as opposed to only fetching
# files for the most recent ref), and arbirary references on this branch can be
# selected.
ARG NUMBA_DPEX_GIT_BRANCH=main
ARG NUMBA_DPEX_GIT_REF=511241f1956eff7c916ebe876a7b3586400b5a9a
# If not blank, this will override the version that is inferred by versioneer based on
# the reference of the commit being used for the build. Use it especially when
# installing an arbitrary commit from the main branch, in which case versioneer
# resolve the version to 0.0.0 which can cause downstream dependency resolution issues
# for packages that depend on numba_dpex .
ARG NUMBA_DPEX_VERSION_OVERRIDE=0.22.0dev0
ARG NUMBA_DPEX_GIT_URL=https://github.com/IntelPython/numba-dpex.git

ARG DPCPP_LLVM_SPIRV_GIT_BRANCH=main
ARG DPCPP_LLVM_SPIRV_GIT_URL=https://github.com/IntelPython/dpcpp-llvm-spirv/

# Version of other python packages explicitly installed either within the
# build environment or the runtime environment.

ARG NUMPY_VERSION="<1.25"
ARG NUMBA_VERSION=""
ARG CYTHON_VERSION=""
ARG WHEEL_VERSION=""
ARG SCIKIT_BUILD_VERSION=""
ARG PACKAGING_VERSION=""

ARG SKBUILD_ARGS="-- -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx"

# Pyenv installer github reference
ARG PYENV_INSTALLER_REF=master

ARG TMPDIR=/tmp

FROM $BASE as l0_base
ARG TMPDIR
ARG L0_LOADER_DEB_VERSION
ARG L0_LOADER_DEB_PLATFORM
ARG L0_LOADER_DEB_ARCH=amd64
ARG L0_LOADER_DEB_NAME=level-zero_${L0_LOADER_DEB_VERSION}+${L0_LOADER_DEB_PLATFORM}_${L0_LOADER_DEB_ARCH}.deb
ARG L0_LOADER_DEVEL_DEB_NAME=level-zero-devel_${L0_LOADER_DEB_VERSION}+${L0_LOADER_DEB_PLATFORM}_${L0_LOADER_DEB_ARCH}.deb
ARG L0_LOADER_DEB_URL=https://github.com/oneapi-src/level-zero/releases/download
ARG L0_LOADER_DOWNLOAD_DIR=$TMPDIR/l0_loader
RUN mkdir -p $L0_LOADER_DOWNLOAD_DIR \
    && wget -P $L0_LOADER_DOWNLOAD_DIR $L0_LOADER_DEB_URL/v${L0_LOADER_DEB_VERSION}/$L0_LOADER_DEB_NAME \
    && wget -P $L0_LOADER_DOWNLOAD_DIR $L0_LOADER_DEB_URL/v${L0_LOADER_DEB_VERSION}/$L0_LOADER_DEVEL_DEB_NAME \
    && dpkg -i $L0_LOADER_DOWNLOAD_DIR/$L0_LOADER_DEB_NAME \
    && dpkg -i $L0_LOADER_DOWNLOAD_DIR/$L0_LOADER_DEVEL_DEB_NAME \
    && rm -Rf $L0_LOADER_DOWNLOAD_DIR


FROM l0_base AS oneapi_devel_installer
ARG TMPDIR
ARG ONEAPI_INSTALL_DIR
ARG ONEAPI_INSTALLER_URL
ARG ONEAPI_INSTALL_BINARY_NAME
ARG ONEAPI_CACHE_DIR=$TMPDIR/intel/cache
ARG ONEAPI_DOWNLOAD_DIR=$TMPDIR/intel/download
ARG ONEAPI_LOG_DIR=$TMPDIR/intel/log
ARG ONEAPI_COMPONENTS="intel.oneapi.lin.dpcpp-cpp-compiler:intel.oneapi.lin.tbb.devel:intel.oneapi.lin.mkl.devel:intel.oneapi.lin.dnnl"
RUN apt-get update --quiet && apt-get install -y --no-install-recommends gawk \
    && rm -Rf /var/lib/apt/lists/* \
    && mkdir -p $ONEAPI_INSTALL_DIR $ONEAPI_CACHE_DIR $ONEAPI_DOWNLOAD_DIR $ONEAPI_LOG_DIR \
    && wget -P $ONEAPI_DOWNLOAD_DIR $ONEAPI_INSTALLER_URL/$ONEAPI_INSTALL_BINARY_NAME \
    && chmod +x $ONEAPI_DOWNLOAD_DIR/$ONEAPI_INSTALL_BINARY_NAME \
    && $ONEAPI_DOWNLOAD_DIR/$ONEAPI_INSTALL_BINARY_NAME -a -s --eula accept \
        --action install --components $ONEAPI_COMPONENTS  \
        --install-dir $ONEAPI_INSTALL_DIR --log-dir $ONEAPI_LOG_DIR \
        --download-cache $ONEAPI_CACHE_DIR \
    && rm -Rf $TMPDIR/*


FROM l0_base AS base_config
ENV PYENV_ROOT=/opt/pyenv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
ENV LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"

# The following blocks script a convenient way of installing python
# It installs a python binary that is separate from the system binary
# and the version is easily configurable.
# The build and the environment management is managed with pyenv.

FROM base_config AS python_builder
ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive
ARG PYENV_INSTALLER_REF
ARG NUMPY_VERSION
ARG NUMBA_VERSION
ARG BASE_PYPI_URL
ARG INTEL_PYPI_URL
RUN apt-get update --quiet && apt-get install -y --no-install-recommends \
        make \
        build-essential \
        libssl-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        git \
        ca-certificates \
     && rm -rf /var/lib/apt/lists/*
RUN curl -L \
    https://raw.githubusercontent.com/pyenv/pyenv-installer/${PYENV_INSTALLER_REF}/bin/pyenv-installer \
    | bash
RUN CFLAGS="-O2 -pipe" \
    CONFIGURE_OPTS="--enable-shared" \
    # --enable-optimizations # possible performance improvements if building for local use
    pyenv install -v $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION
RUN python -m venv $VIRTUAL_ENV \
    && pip install -U pip \
    && pip install -U setuptools \
    && pip install -U -i $INTEL_PYPI_URL --extra-index-url $BASE_PYPI_URL \
        numpy${NUMPY_VERSION} numba${NUMBA_VERSION} \
    && pip cache purge


FROM base_config AS build_environment
ARG TMPDIR
ARG CMAKE_VERSION
ARG CMAKE_VERSION_BUILD
ARG CMAKE_BUILD_DIR=$TMPDIR/cmake
ARG CMAKE_INSTALL_DIR=/opt/cmake
ARG CMAKE_INSTALLER_NAME=cmake-${CMAKE_VERSION}.${CMAKE_VERSION_BUILD}-linux-x86_64.sh
ARG INTEL_PYPI_URL
ARG BASE_PYPI_URL
ARG WHEEL_VERSION
ARG CYTHON_VERSION
ARG SCIKIT_BUILD_VERSION
ARG ONEAPI_INSTALL_DIR
# Install an up-to-date cmake that is necessary for some builds
# Taken from https://askubuntu.com/a/865294
RUN apt-get update --quiet \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libtool \
        autoconf \
        unzip \
        libssl-dev \
        wget \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN apt purge -y --auto-remove cmake \
    && mkdir -p $CMAKE_BUILD_DIR $CMAKE_INSTALL_DIR \
    && cd $CMAKE_BUILD_DIR \
    && wget https://cmake.org/files/v${CMAKE_VERSION}/${CMAKE_INSTALLER_NAME} \
    && sh ${CMAKE_INSTALLER_NAME} --prefix=${CMAKE_INSTALL_DIR} --skip-license \
    && rm -Rf $CMAKE_BUILD_DIR
ENV PATH=$CMAKE_INSTALL_DIR/bin:$PATH
COPY --from=python_builder $PYENV_ROOT $PYENV_ROOT
COPY --from=python_builder $VIRTUAL_ENV $VIRTUAL_ENV
# install some common python prerequisites for dpctl, dpnp and numba-dpex builds
RUN pip install -U -i $INTEL_PYPI_URL --extra-index-url $BASE_PYPI_URL \
        wheel${WHEEL_VERSION} cython${CYTHON_VERSION} \
        scikit-build${SCIKIT_BUILD_VERSION} \
    && pip cache purge
# install the oneapi devel environment
ENV CMAKE_GENERATOR="Ninja"


# Build the four intel python packages

FROM build_environment AS dpctl_builder
SHELL ["/bin/bash", "-c"]
ARG TMPDIR
ARG ONEAPI_INSTALL_DIR
ARG DPCTL_GIT_BRANCH
ARG DPCTL_GIT_URL
ARG DPCTL_BUILD_DIR=$TMPDIR/dpctl
ARG SKBUILD_ARGS
RUN --mount=type=bind,target=$ONEAPI_INSTALL_DIR,source=$ONEAPI_INSTALL_DIR,from=oneapi_devel_installer \
    mkdir -p $DPCTL_BUILD_DIR \
    && cd $DPCTL_BUILD_DIR \
    && git clone --recursive -b $DPCTL_GIT_BRANCH --depth 1 $DPCTL_GIT_URL . \
    && source $ONEAPI_INSTALL_DIR/setvars.sh \
    && python setup.py clean --all \
    # XXX: is it needed to pass manylinux wheel build arg to the setup command ?
    && python setup.py bdist_wheel ${SKBUILD_ARGS} \
    && cp dist/dpctl*.whl $TMPDIR \
    && rm -Rf $DPCTL_BUILD_DIR

FROM build_environment AS dpnp_builder
SHELL ["/bin/bash", "-c"]
ARG TMPDIR
ARG INTEL_PYPI_URL
ARG BASE_PYPI_URL
ARG ONEAPI_INSTALL_DIR
ARG DPNP_BUILD_DIR=$TMPDIR/dpnp
ARG DPNP_GIT_BRANCH
ARG DPNP_GIT_URL
ARG SKBUILD_ARGS
RUN --mount=type=bind,target=$ONEAPI_INSTALL_DIR,source=$ONEAPI_INSTALL_DIR,from=oneapi_devel_installer \
    --mount=type=bind,target=$TMPDIR/dpctl,source=$TMPDIR,from=dpctl_builder \
    pip install -U -i $INTEL_PYPI_URL --extra-index-url $BASE_PYPI_URL $TMPDIR/dpctl/dpctl*.whl \
    && pip cache purge \
    && mkdir -p $DPNP_BUILD_DIR \
    && cd $DPNP_BUILD_DIR \
    && git clone --recursive -b $DPNP_GIT_BRANCH --depth 1 $DPNP_GIT_URL . \
    && source $ONEAPI_INSTALL_DIR/setvars.sh \
    && python setup.py clean \
    # XXX: is it needed to pass manylinux wheel build arg to the setup command ?
    && export DPL_ROOT_HINT=$ONEAPI_ROOT/dpl/latest \
    && python setup.py bdist_wheel ${SKBUILD_ARGS} -DDPCTL_MODULE_PATH=$(python -m dpctl --cmakedir) \
    && cp dist/dpnp*.whl $TMPDIR \
    && rm -Rf $DPNP_BUILD_DIR

FROM build_environment AS dpcpp_llvm_spirv_builder
SHELL ["/bin/bash", "-c"]
ARG TMPDIR
ARG ONEAPI_INSTALL_DIR
ARG DPCPP_LLVM_SPIRV_GIT_BRANCH
ARG DPCPP_LLVM_SPIRV_GIT_URL
ARG DPCPP_LLVM_SPIRV_BUILD_DIR=$TMPDIR/dpcpp_llvm_spirv
RUN --mount=type=bind,target=$ONEAPI_INSTALL_DIR,source=$ONEAPI_INSTALL_DIR,from=oneapi_devel_installer \
    mkdir -p $DPCPP_LLVM_SPIRV_BUILD_DIR \
    && cd $DPCPP_LLVM_SPIRV_BUILD_DIR \
    && git clone --recursive -b $DPCPP_LLVM_SPIRV_GIT_BRANCH --depth 1 $DPCPP_LLVM_SPIRV_GIT_URL . \
    && cd pkg \
    && python setup.py clean --all \
    && python setup.py bdist_wheel \
    && cp dist/dpcpp_llvm_spirv*.whl $TMPDIR \
    && cp $ONEAPI_INSTALL_DIR/compiler/latest/linux/bin-llvm/llvm-spirv $TMPDIR \
    && rm -Rf $DPCPP_LLVM_SPIRV_BUILD_DIR

FROM build_environment AS numba_dpex_builder
SHELL ["/bin/bash", "-c"]
ARG TMPDIR
ARG ONEAPI_INSTALL_DIR
ARG NUMBA_DPEX_BUILD_DIR=$TMPDIR/numba_dpex
ARG NUMBA_DPEX_GIT_BRANCH
ARG NUMBA_DPEX_GIT_REF
ARG NUMBA_DPEX_GIT_URL
ARG NUMBA_DPEX_VERSION_OVERRIDE
ARG INTEL_PYPI_URL
ARG BASE_PYPI_URL
RUN --mount=type=bind,target=$ONEAPI_INSTALL_DIR,source=$ONEAPI_INSTALL_DIR,from=oneapi_devel_installer \
    --mount=type=bind,target=$TMPDIR/dpctl,source=$TMPDIR,from=dpctl_builder \
    --mount=type=bind,target=$TMPDIR/dpnp,source=$TMPDIR,from=dpnp_builder \
    --mount=type=bind,target=$TMPDIR/dpcpp_llvm_spirv,source=$TMPDIR,from=dpcpp_llvm_spirv_builder \
    pip install -U -i $INTEL_PYPI_URL --extra-index-url $BASE_PYPI_URL \
        $TMPDIR/dpctl/dpctl*.whl $TMPDIR/dpnp/dpnp*.whl $TMPDIR/dpcpp_llvm_spirv/dpcpp_llvm_spirv*.whl \
    && cp $TMPDIR/dpcpp_llvm_spirv/llvm-spirv \
        $(python -c "import dpcpp_llvm_spirv as p; print(p.get_llvm_spirv_path())") \
    && pip cache purge \
    && mkdir -p $NUMBA_DPEX_BUILD_DIR \
    && cd $NUMBA_DPEX_BUILD_DIR \
    && git clone --recursive -b $NUMBA_DPEX_GIT_BRANCH $NUMBA_DPEX_GIT_URL . \
    && git checkout $NUMBA_DPEX_GIT_REF \
    && if [ -n "${NUMBA_DPEX_VERSION_OVERRIDE}" ]; then git tag -f "${NUMBA_DPEX_VERSION_OVERRIDE}"; fi \
    && source $ONEAPI_INSTALL_DIR/setvars.sh \
    # XXX: is it needed to pass manylinux wheel build arg to the setup command ?
    && python setup.py bdist_wheel \
    && cp dist/numba_dpex*.whl $TMPDIR \
    && rm -Rf $NUMBA_DPEX_BUILD_DIR

# HACK: because it's the last version on which sklearn_numba_dpex is stable on cpu, also
# build a wheel and add it to the image so it can be used for test pipeline on cpu-only
# runners.
FROM build_environment AS numba_dpex_builder_cpu_stable
SHELL ["/bin/bash", "-c"]
ARG TMPDIR
ARG ONEAPI_INSTALL_DIR
ARG NUMBA_DPEX_BUILD_DIR=$TMPDIR/cpu-stable/numba_dpex
ARG NUMBA_DPEX_GIT_BRANCH=0.21.0dev1
ARG NUMBA_DPEX_GIT_REF=0.21.0dev1
ARG NUMBA_DPEX_GIT_URL
ARG INTEL_PYPI_URL
ARG BASE_PYPI_URL
RUN --mount=type=bind,target=$ONEAPI_INSTALL_DIR,source=$ONEAPI_INSTALL_DIR,from=oneapi_devel_installer \
    --mount=type=bind,target=$TMPDIR/dpctl,source=$TMPDIR,from=dpctl_builder \
    --mount=type=bind,target=$TMPDIR/dpnp,source=$TMPDIR,from=dpnp_builder \
    --mount=type=bind,target=$TMPDIR/dpcpp_llvm_spirv,source=$TMPDIR,from=dpcpp_llvm_spirv_builder \
    pip install -U -i $INTEL_PYPI_URL --extra-index-url $BASE_PYPI_URL \
        $TMPDIR/dpctl/dpctl*.whl $TMPDIR/dpnp/dpnp*.whl $TMPDIR/dpcpp_llvm_spirv/dpcpp_llvm_spirv*.whl \
    && cp $TMPDIR/dpcpp_llvm_spirv/llvm-spirv \
        $(python -c "import dpcpp_llvm_spirv as p; print(p.get_llvm_spirv_path())") \
    && pip cache purge \
    && mkdir -p $NUMBA_DPEX_BUILD_DIR \
    && cd $NUMBA_DPEX_BUILD_DIR \
    && git clone --recursive -b $NUMBA_DPEX_GIT_BRANCH $NUMBA_DPEX_GIT_URL . \
    && git checkout $NUMBA_DPEX_GIT_REF \
    && source $ONEAPI_INSTALL_DIR/setvars.sh \
    # HACK: currently, there is an issue with the bdist_wheel configuration for numba_dpex
    # that causes missing files in the final tarball.
    # The workaround consists in triggering building steps by running `setup.py develop`
    # before running `setup.py bdist_wheel`.
    # See https://github.com/soda-inria/sklearn-numba-dpex/issues/5
    && python setup.py develop \
    # XXX: is it needed to pass manylinux wheel build arg to the setup command ?
    && python setup.py bdist_wheel \
    && cp dist/numba_dpex*.whl $TMPDIR/cpu-stable/ \
    && rm -Rf $NUMBA_DPEX_BUILD_DIR


FROM base_config AS runtime_environment
COPY --from=python_builder $PYENV_ROOT $PYENV_ROOT
COPY --from=python_builder $VIRTUAL_ENV $VIRTUAL_ENV


# Install the 4 packages and the spirv-tools binaries and headers
FROM runtime_environment AS numba_dpex_dev
ARG TMPDIR
ARG INTEL_PYPI_URL
ARG BASE_PYPI_URL
ARG PACKAGING_VERSION
RUN --mount=type=tmpfs,target=$TMPDIR \
    --mount=type=bind,target=$TMPDIR/dpctl,source=$TMPDIR,from=dpctl_builder \
    --mount=type=bind,target=$TMPDIR/dpnp,source=$TMPDIR,from=dpnp_builder \
    --mount=type=bind,target=$TMPDIR/numba_dpex,source=$TMPDIR,from=numba_dpex_builder \
    --mount=type=bind,target=$TMPDIR/cpu-stable,source=$TMPDIR/cpu-stable,from=numba_dpex_builder_cpu_stable \
    --mount=type=bind,target=$TMPDIR/dpcpp_llvm_spirv,source=$TMPDIR,from=dpcpp_llvm_spirv_builder \
    pip install -U -i $INTEL_PYPI_URL --extra-index-url $BASE_PYPI_URL \
        $TMPDIR/dpctl/dpctl*.whl $TMPDIR/dpnp/dpnp*.whl \
        $TMPDIR/dpcpp_llvm_spirv/dpcpp_llvm_spirv*.whl \
        $TMPDIR/numba_dpex/numba_dpex*.whl \
        packaging${PACKAGING_VERSION} \
        # HACK this package is not in the dependency tree of any of the top
        # packages but probably should, in the meantime we add it manually
        mkl-dpcpp \
    && pip cache purge \
    && cp $TMPDIR/dpcpp_llvm_spirv/llvm-spirv \
        $(python -c "import dpcpp_llvm_spirv as p; print(p.get_llvm_spirv_path())") \
    && mkdir -p /opt/numba_dpex/cpu-stable \
    && cp $TMPDIR/cpu-stable/numba_dpex*.whl /opt/numba_dpex/cpu-stable/
RUN apt-get update --quiet && apt-get install -y --no-install-recommends \
        spirv-tools spirv-headers \
    && rm -Rf /var/lib/apt/lists/*
CMD ["/bin/bash"]
