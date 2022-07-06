# This Dockerfile can be built from scratch with e.g the command
# docker build . -t my_numba_dpex_build
# The ARGs in the header of this file can be changed as needed (either by
# editing the file or with the command line).
# The build can be fairly long (up to one hour on low-end machines.)
# To start a container, use e.g
# sudo docker run --rm --name my_container -it -v /my/host/volume/:/mounted/volume --device=/dev/dri my_numba_dpex_build
# The parameter --device=/dev/dri enables gpu passthrough (as long as the user 
# has permissions, e.g by being a member of the video group)


# TODO: make a custom build of the base image to test with other base OS
# (e.g ubuntu2204)
# ???: actually not sure if it's needed or interesting to use the sycl nightly
# binaries at all. The nightly also ship an up-to-date installation of the
# intel runtime, including drivers and opencl and level_zero runtimes.

ARG BASE=ghcr.io/intel/llvm/sycl_ubuntu2004_nightly
ARG TMPDIR=/tmp

# Various resources about the SYCL toolchain
# Environment variables with important effects on the SYCL toolchain:
# https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter
# Overview of the SYCL build instructions and environment requirements: https://intel.github.io/llvm-docs/GetStartedGuide.html
# More about sycl docker image Dockerfiles: https://github.com/intel/llvm/tree/sycl/devops/containers
# and https://github.com/intel/llvm/blob/sycl/sycl/doc/developer/DockerBKMs.md


ARG INTEL_PYPI_URL=https://pypi.anaconda.org/intel/simple
ARG BASE_PYPI_URL=https://pypi.org/simple

# /!\ the following version must exist in intel pypi. It is recommended to
# not use version ranges, because if a higher version exists in pypi.org it
# will be preferred over the versions in the intel pypi.

ARG INTEL_NUMPY_VERSION="==1.22.3"

# TODO: This can be fixed if intel can host the whole the dependency tree
# in the INTEL_PYPI_URL instead of having to rely in pypi.org for the main part
# of the dependency tree.
# TODO: find which part of the intel basekit should be installed
# with the installer, and which part can be used from python packages, at
# build time and at run time, might enable a lighter build.

# ???: not clear if those versions should also be limited to the content
# of the intel pypi or not. (looks like not)
ARG INTEL_SCIPY_VERSION=""
ARG INTEL_NUMBA_VERSION="<0.56"


# So far, the intel numpy python package is not built for python>=3.10
ARG PYTHON_VERSION=3.9.13


# This link might need to be refreshed occasionally.
# It can be found at https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=webdownload&options=online
# The installer CLI is documented here: https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-with-command-line.html#install-with-command-line_interactive

ARG ONEAPI_INSTALLER=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/18673/l_BaseKit_p_2022.2.0.262.sh
ARG ONEAPI_INSTALL_DIR=/opt/intel/oneapi


# Some build steps requires a high enough version of cmake.
# Bump it if necessary.

ARG CMAKE_VERSION=3.23
ARG CMAKE_VERSION_BUILD=2


# Versions of the intel python packages

ARG DPCTL_GIT_BRANCH=0.13.0dev2
ARG DPCTL_GIT_URL=https://github.com/IntelPython/dpctl.git

ARG DPNP_GIT_BRANCH=master
ARG DPNP_GIT_URL=https://github.com/IntelPython/dpnp.git

ARG NUMBA_DPEX_GIT_BRANCH=0.18.0
ARG NUMBA_DPEX_GIT_URL=https://github.com/IntelPython/numba-dpex.git


FROM $BASE AS oneapi_base_toolkit_installer
ARG TMPDIR
ARG ONEAPI_INSTALL_DIR
ARG ONEAPI_INSTALLER
ARG ONEAPI_CACHE_DIR=$TMPDIR/intel/cache
ARG ONEAPI_DOWNLOAD_DIR=$TMPDIR/intel/download
ARG ONEAPI_LOG_DIR=$TMPDIR/intel/log
# TODO: are all those components necessary ?
# TODO: some of those components that are necessary for build might not be
# necessary for runtime ?
ARG ONEAPI_COMPONENTS="intel.oneapi.lin.dpcpp-ct:intel.oneapi.lin.dpcpp_dbg:intel.oneapi.lin.dpcpp-cpp-compiler:intel.oneapi.lin.tbb.devel:intel.oneapi.lin.mkl.devel"
RUN mkdir -p $ONEAPI_INSTALL_DIR $ONEAPI_CACHE_DIR $ONEAPI_DOWNLOAD_DIR ONEAPI_LOG_DIR \
    && wget -P $ONEAPI_DOWNLOAD_DIR $ONEAPI_INSTALLER \
    && chmod +x $ONEAPI_DOWNLOAD_DIR/l_BaseKit_p_2022.2.0.262.sh \
    && $ONEAPI_DOWNLOAD_DIR/l_BaseKit_p_2022.2.0.262.sh -a -s --eula accept --action install --components $ONEAPI_COMPONENTS --install-dir $ONEAPI_INSTALL_DIR --log-dir $ONEAPI_INSTALL_DIR --download-cache $ONEAPI_CACHE_DIR \
    && rm -Rf $TMPDIR/*


# The two following blocks script a convenient way of installing python
# It installs a python binary that is separate from the system binary
# and the version is easily configurable.
# The build and the environment management is managed with pyenv.

FROM $BASE AS python_base_config
ENV PYENV_ROOT="/opt/pyenv"
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
ENV LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"

FROM python_base_config AS python_builder
ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --quiet; apt-get install -y --no-install-recommends \
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
    https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer \
    | bash
RUN CFLAGS="-O2 -pipe" \
    CONFIGURE_OPTS="--enable-shared --with-computed-gotos" \
    # --enable-optimizations # possible performance improvements if building for local use
    pyenv install -v $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION
RUN python -m venv $VIRTUAL_ENV && pip install -U pip && pip install -U setuptools


# Assemble the base image that will be used to build the final base runtime image

FROM python_base_config AS host
COPY --from=python_builder $PYENV_ROOT $PYENV_ROOT
COPY --from=python_builder $VIRTUAL_ENV $VIRTUAL_ENV
COPY --from=oneapi_base_toolkit_installer $ONEAPI_INSTALL_DIR $ONEAPI_INSTALL_DIR
# Those environment variables are required by the dpnp runtime
ENV OCL_ICD_FILENAMES_RESET=1
ENV OCL_ICD_FILENAMES=libintelocl.so


# Install an up-to-date cmake that is necessary for some builds
# and also some common prerequisites for dpctl, dpnp and numba-dpex builds

FROM host AS base_builder
ARG TMPDIR
ARG CMAKE_VERSION
ARG CMAKE_VERSION_BUILD
ARG CMAKE_BUILD_DIR=$TMPDIR/cmake
ARG INTEL_PYPI_URL
ARG BASE_PYPI_URL
ARG INTEL_NUMPY_VERSION
ARG INTEL_SCIPY_VERSION
RUN  apt-get update --quiet \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libtool \
        autoconf \
        unzip \
        libssl-dev \
        wget \
        ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && apt purge -y --auto-remove cmake \
    && mkdir -p $CMAKE_BUILD_DIR \
    && cd $CMAKE_BUILD_DIR \
    && wget https://cmake.org/files/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.${CMAKE_VERSION_BUILD}.tar.gz \
    && tar -xzvf cmake-${CMAKE_VERSION}.${CMAKE_VERSION_BUILD}.tar.gz \
    && cd cmake-${CMAKE_VERSION}.${CMAKE_VERSION_BUILD} \
    && ./bootstrap \
    && make -j$(nproc) \
    && sudo make install \
    && rm -Rf $CMAKE_BUILD_DIR \
    && cd \
    && pip install -U -i $INTEL_PYPI_URL --extra-index-url $BASE_PYPI_URL "numpy${INTEL_NUMPY_VERSION}" "numba${INTEL_NUMBA_VERSION}" "scipy${INTEL_SCIPY_VERSION}" wheel cython scikit-build
ENV CMAKE_GENERATOR="Ninja"


# Build the three intel python packages

FROM base_builder AS dpctl_builder
SHELL ["/bin/bash", "-c"]
ARG TMPDIR
ARG ONEAPI_INSTALL_DIR
ARG DPCTL_GIT_BRANCH
ARG DPCTL_GIT_URL
ARG DPCTL_BUILD_DIR=$TMPDIR/dpctl
ARG SKBUILD_ARGS="-- -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx"
RUN mkdir -p $DPCTL_BUILD_DIR \
    && cd $DPCTL_BUILD_DIR \
    && git clone --recursive -b $DPCTL_GIT_BRANCH --depth 1 $DPCTL_GIT_URL . \
    && source $ONEAPI_INSTALL_DIR/setvars.sh \
    && export PATH="/opt/sycl/bin:$VIRTUAL_ENV/bin:$PATH" \
    && export LD_LIBRARY_PATH="/opt/sycl/lib:$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH" \
    && python setup.py clean --all \
    # TODO: manylinux wheel build arg ?
    && python setup.py bdist_wheel ${SKBUILD_ARGS} \
    && cp dist/dpctl*.whl $TMPDIR \
    && rm -Rf $DPCTL_BUILD_DIR

FROM base_builder AS dpnp_builder
SHELL ["/bin/bash", "-c"]
ARG TMPDIR
ARG INTEL_PYPI_URL
ARG BASE_PYPI_URL
ARG ONEAPI_INSTALL_DIR
ARG DPNP_BUILD_DIR=$TMPDIR/dpnp
ARG DPNP_GIT_BRANCH
ARG DPNP_GIT_URL
ARG DPNP_DEBUG=1
COPY --from=dpctl_builder $TMPDIR/dpctl*.whl $TMPDIR
RUN pip install -U -i $INTEL_PYPI_URL --extra-index-url $BASE_PYPI_URL $TMPDIR/dpctl*.whl \
    && mkdir -p $DPNP_BUILD_DIR \
    && cd $DPNP_BUILD_DIR \
    && git clone --recursive -b $DPNP_GIT_BRANCH --depth 1 $DPNP_GIT_URL . \
    && source $ONEAPI_INSTALL_DIR/setvars.sh \
    && export PATH="/opt/sycl/bin:$VIRTUAL_ENV/bin:$PATH" \
    && export LD_LIBRARY_PATH="/opt/sycl/lib:$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH" \
    && export DPLROOT=$ONEAPI_ROOT/dpl/latest \
    && python setup.py clean \
    && python setup.py build_clib \
    && export CC=dpcpp \
    && python setup.py build_ext \
    # TODO: manylinux wheel build arg ?
    && python setup.py bdist_wheel \
    && cp dist/dpnp*.whl $TMPDIR \
    && rm -Rf $DPNP_BUILD_DIR

FROM base_builder AS numba_dpex_builder
SHELL ["/bin/bash", "-c"]
ARG TMPDIR
ARG ONEAPI_INSTALL_DIR
ARG NUMBA_DPEX_BUILD_DIR=$TMPDIR/numba_dpex
ARG NUMBA_DPEX_GIT_BRANCH
ARG NUMBA_DPEX_GIT_URL
ARG INTEL_NUMBA_VERSION
ARG INTEL_PYPI_URL
ARG BASE_PYPI_URL
COPY --from=dpctl_builder $TMPDIR/dpctl*.whl $TMPDIR
COPY --from=dpnp_builder $TMPDIR/dpnp*.whl $TMPDIR
RUN pip install -U -i $INTEL_PYPI_URL --extra-index-url $BASE_PYPI_URL $TMPDIR/dpctl*.whl $TMPDIR/dpnp*.whl packaging \
    && rm -Rf $TMPDIR/* \
    && mkdir -p $NUMBA_DPEX_BUILD_DIR \
    && cd $NUMBA_DPEX_BUILD_DIR \
    && git clone --recursive -b $NUMBA_DPEX_GIT_BRANCH --depth 1 $NUMBA_DPEX_GIT_URL . \
    && apt-get update --quiet && apt-get install -y --no-install-recommends \
        spirv-tools spirv-headers \
    && rm -rf /var/lib/apt/lists/* \
    && source $ONEAPI_INSTALL_DIR/setvars.sh \
    && export PATH=/opt/sycl/bin:$VIRTUAL_ENV/bin:$ONEAPI_INSTALL_DIR/compiler/latest/bin-llvm:$PATH \
    && export LD_LIBRARY_PATH=/opt/sycl/lib:$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH \
    && python setup.py bdist_wheel \
    && cp dist/numba_dppy*.whl $TMPDIR


# Install the 3 packages and setup the entrypoint to source the 
# oneapi setvars.sh file on startup, and fix some other environment variables
# accordingly.
FROM host AS numba_dpex_dev_installer
ARG TMPDIR
ARG INTEL_PYPI_URL
ARG BASE_PYPI_URL
ARG INTEL_NUMPY_VERSION
ARG INTEL_SCIPY_VERSION
ARG INTEL_NUMBA_VERSION
COPY --from=dpctl_builder $TMPDIR/dpctl*.whl $TMPDIR
COPY --from=dpnp_builder $TMPDIR/dpnp*.whl $TMPDIR
COPY --from=numba_dpex_builder $TMPDIR/numba_dppy*.whl $TMPDIR
RUN apt-get update --quiet && apt-get install -y --no-install-recommends spirv-tools spirv-headers \
    && rm -Rf /var/lib/apt/lists/* \
    && pip install -U -i $INTEL_PYPI_URL --extra-index-url $BASE_PYPI_URL "numpy${INTEL_NUMPY_VERSION}" "numba${INTEL_NUMBA_VERSION}" "scipy${INTEL_SCIPY_VERSION}" $TMPDIR/dpctl*.whl $TMPDIR/dpnp*.whl $TMPDIR/numba_dppy*.whl cython packaging pytest pexpect \
    && rm -Rf $TMPDIR/*

# and squash the layers on the install step
FROM host AS numba_dpex_dev
ARG ONEAPI_INSTALL_DIR
COPY --from=numba_dpex_dev_installer / /
ENV ONEAPI_INSTALL_DIR=$ONEAPI_INSTALL_DIR
ENTRYPOINT ["/bin/bash", "-c", "source $ONEAPI_INSTALL_DIR/setvars.sh && export PATH=/opt/sycl/bin:$VIRTUAL_ENV/bin:$ONEAPI_INSTALL_DIR/compiler/latest/bin-llvm:$PATH && export LD_LIBRARY_PATH=/opt/sycl/lib:$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH && exec \"$@\"", ""]
CMD ["/bin/bash"]
