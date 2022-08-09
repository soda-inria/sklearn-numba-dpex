# This Dockerfile can be built from scratch with e.g. the command
#
#     docker build . -t numba_dpex_dev
#
# The ARGs in the header of this file can be changed as needed (either by
# editing the file or with the command line).
# The build can be fairly long (up to one hour on low-end machines.)
# To start a container, use e.g.:
#
#      sudo docker run --rm --name my_container -it -v /my/host/volume/:/mounted/volume --device=/dev/dri numba_dpex_dev
#
# The --device=/dev/dri option enables GPU passthrough (as long as the user
# has permissions, e.g. by being a member of the video group)

# The `ubuntu2004_intel_drivers` ships an up-to-date installation of the
# intel runtime, including drivers and opencl and level_zero runtimes.
#
# See available versions here: https://github.com/intel/llvm/pkgs/container/llvm%2Fubuntu2004_intel_drivers
#
# TODO: make a custom build with other base images to test compatibiliry
# with other OSes
ARG BASE_IMAGE_VERSION=latest-72882d570b860258739a83bc23609c6f6dd3002f
ARG BASE_IMAGE_TAG=ghcr.io/intel/llvm/ubuntu2004_intel_drivers

ARG BASE=${BASE_IMAGE_TAG}:${BASE_IMAGE_VERSION}

# Various resources about the SYCL toolchain
# Environment variables with important effects on the SYCL toolchain:
# https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter
# Overview of the SYCL build instructions and environment requirements: https://intel.github.io/llvm-docs/GetStartedGuide.html
# More about sycl docker image Dockerfiles: https://github.com/intel/llvm/tree/sycl/devops/containers
# and https://github.com/intel/llvm/blob/sycl/sycl/doc/developer/DockerBKMs.md


ARG INTEL_PYPI_URL=https://pypi.anaconda.org/intel/simple
ARG BASE_PYPI_URL=https://pypi.org/simple

# /!\ the following versions must exist in intel pypi. It is recommended to
# not use version ranges, because if a higher version exists in pypi.org it
# will be preferred over the versions in the intel pypi.

ARG INTEL_NUMPY_VERSION="==1.22.3"
ARG INTEL_SCIPY_VERSION="==1.7.3"

# XXX: This can be fixed if intel can host the whole the dependency tree
# in the INTEL_PYPI_URL instead of having to rely in pypi.org for the main part
# of the dependency tree.

# ???: it is not clear so far if this version should also be limited to the
# content of the intel pypi or not. (this Dockerfile currently assumes that
# it is not limited to intel pypi. The opposite assumption seems to create
# impossible build conditions.)
ARG INTEL_NUMBA_VERSION="<0.56"


# So far, the intel numpy python package is not built for python>=3.10
ARG PYTHON_VERSION=3.9.13


# This link might need to be refreshed occasionally.
# It can be found at https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=webdownload&options=online
# The installer CLI is documented here: https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-with-command-line.html#install-with-command-line_interactive

ARG ONEAPI_INSTALLER_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/18673
ARG ONEAPI_INSTALL_BINARY_NAME=l_BaseKit_p_2022.2.0.262.sh

ARG ONEAPI_INSTALL_DIR=/opt/intel/oneapi

# Some build steps requires a high enough version of cmake.
# Bump it if necessary.

ARG CMAKE_VERSION=3.23
ARG CMAKE_VERSION_BUILD=2


# Versions of the intel python packages

ARG DPCTL_GIT_BRANCH=0.13.0dev2
ARG DPCTL_GIT_URL=https://github.com/IntelPython/dpctl.git

ARG DPNP_GIT_BRANCH=0.10.0
ARG DPNP_GIT_URL=https://github.com/IntelPython/dpnp.git

ARG NUMBA_DPEX_GIT_BRANCH=0.18.0
ARG NUMBA_DPEX_GIT_URL=https://github.com/IntelPython/numba-dpex.git

# Version of other python packages explicitly installed either within the
# build environment or the runtime environment.

ARG CYTHON_VERSION=""
ARG WHEEL_VERSION=""
ARG SCIKIT_BUILD_VERSION=""
ARG PACKAGING_VERSION=""

# Pyenv installer github reference
ARG PYENV_INSTALLER_REF=master

ARG TMPDIR=/tmp

FROM $BASE AS oneapi_dpcpp_binaries_installer
ARG TMPDIR
ARG ONEAPI_INSTALL_DIR
ARG ONEAPI_INSTALLER_URL
ARG ONEAPI_INSTALL_BINARY_NAME
ARG ONEAPI_CACHE_DIR=$TMPDIR/intel/cache
ARG ONEAPI_DOWNLOAD_DIR=$TMPDIR/intel/download
ARG ONEAPI_LOG_DIR=$TMPDIR/intel/log
ARG ONEAPI_COMPONENTS="intel.oneapi.lin.dpcpp-cpp-compiler"
RUN mkdir -p $ONEAPI_INSTALL_DIR $ONEAPI_CACHE_DIR $ONEAPI_DOWNLOAD_DIR $ONEAPI_LOG_DIR \
    && wget -P $ONEAPI_DOWNLOAD_DIR $ONEAPI_INSTALLER_URL/$ONEAPI_INSTALL_BINARY_NAME \
    && chmod +x $ONEAPI_DOWNLOAD_DIR/$ONEAPI_INSTALL_BINARY_NAME \
    && $ONEAPI_DOWNLOAD_DIR/$ONEAPI_INSTALL_BINARY_NAME -a -s --eula accept \
        --action install --components $ONEAPI_COMPONENTS  \
        --install-dir $ONEAPI_INSTALL_DIR --log-dir $ONEAPI_LOG_DIR \
        --download-cache $ONEAPI_CACHE_DIR \
    && rm -Rf $TMPDIR/*


FROM oneapi_dpcpp_binaries_installer AS oneapi_devel_installer
ARG TMPDIR
ARG ONEAPI_INSTALL_DIR
ARG ONEAPI_INSTALLER_URL
ARG ONEAPI_INSTALL_BINARY_NAME
ARG ONEAPI_CACHE_DIR=$TMPDIR/intel/cache
ARG ONEAPI_DOWNLOAD_DIR=$TMPDIR/intel/download
ARG ONEAPI_LOG_DIR=$TMPDIR/intel/log
ARG ONEAPI_COMPONENTS="intel.oneapi.lin.dpcpp-cpp-compiler:intel.oneapi.lin.tbb.devel:intel.oneapi.lin.mkl.devel"
RUN mkdir -p $ONEAPI_INSTALL_DIR $ONEAPI_CACHE_DIR $ONEAPI_DOWNLOAD_DIR $ONEAPI_LOG_DIR \
    && wget -P $ONEAPI_DOWNLOAD_DIR $ONEAPI_INSTALLER_URL/$ONEAPI_INSTALL_BINARY_NAME \
    && chmod +x $ONEAPI_DOWNLOAD_DIR/$ONEAPI_INSTALL_BINARY_NAME \
    && $ONEAPI_DOWNLOAD_DIR/$ONEAPI_INSTALL_BINARY_NAME -a -s --eula accept \
        --action modify --components $ONEAPI_COMPONENTS  \
        --install-dir $ONEAPI_INSTALL_DIR --log-dir $ONEAPI_LOG_DIR \
        --download-cache $ONEAPI_CACHE_DIR \
    && rm -Rf $TMPDIR/*


FROM $BASE AS base_config
ARG ONEAPI_INSTALL_DIR
ENV PYENV_ROOT=/opt/pyenv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
ENV LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"
# Those environment variables are required by the dpnp runtime
ENV OCL_ICD_FILENAMES_RESET=1
ENV OCL_ICD_FILENAMES=libintelocl.so
# Needed by the numba-dpex runtime
ENV PATH=$ONEAPI_INSTALL_DIR/compiler/latest/linux/bin-llvm:$PATH

# The following blocks script a convenient way of installing python
# It installs a python binary that is separate from the system binary
# and the version is easily configurable.
# The build and the environment management is managed with pyenv.

FROM base_config AS python_builder
ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive
ARG PYENV_INSTALLER_REF
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
    https://github.com/pyenv/pyenv-installer/raw/${PYENV_INSTALLER_REF}/bin/pyenv-installer \
    | bash
RUN CFLAGS="-O2 -pipe" \
    CONFIGURE_OPTS="--enable-shared" \
    # --enable-optimizations # possible performance improvements if building for local use
    pyenv install -v $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION
RUN python -m venv $VIRTUAL_ENV \
    && pip install -U pip \
    && pip install -U setuptools \
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
ARG INTEL_NUMPY_VERSION
ARG INTEL_SCIPY_VERSION
ARG INTEL_NUMBA_VERSION
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
# HACK: install numpy separately first to ensure that it is fetched from the intel
# pypi.
# The proper fix is for intel to host all the dependency tree.
# see https://stackoverflow.com/questions/67253141/python-pip-priority-order-with-index-url-and-extra-index-url
RUN pip install -U -i $INTEL_PYPI_URL "numpy${INTEL_NUMPY_VERSION}" \
        "scipy${INTEL_SCIPY_VERSION}" --no-deps \
    && pip install -U -i $INTEL_PYPI_URL --extra-index-url $BASE_PYPI_URL \
        "numpy${INTEL_NUMPY_VERSION}" "numba${INTEL_NUMBA_VERSION}" \
        "scipy${INTEL_SCIPY_VERSION}" \
        wheel${WHEEL_VERSION} cython${CYTHON_VERSION} \
        scikit-build${SCIKIT_BUILD_VERSION} \
    && pip cache purge
# install the oneapi devel environment
COPY --from=oneapi_devel_installer $ONEAPI_INSTALL_DIR $ONEAPI_INSTALL_DIR
ENV CMAKE_GENERATOR="Ninja"


# Build the three intel python packages

FROM build_environment AS dpctl_builder
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
ARG DPNP_DEBUG=1
COPY --from=dpctl_builder $TMPDIR/dpctl*.whl $TMPDIR
RUN pip install -U -i $INTEL_PYPI_URL --extra-index-url $BASE_PYPI_URL $TMPDIR/dpctl*.whl \
    && pip cache purge \
    && mkdir -p $DPNP_BUILD_DIR \
    && cd $DPNP_BUILD_DIR \
    && git clone --recursive -b $DPNP_GIT_BRANCH --depth 1 $DPNP_GIT_URL . \
    && source $ONEAPI_INSTALL_DIR/setvars.sh \
    && export DPLROOT=$ONEAPI_ROOT/dpl/latest \
    && python setup.py clean \
    && python setup.py build_clib \
    && export CC=dpcpp \
    && python setup.py build_ext \
    # XXX: is it needed to pass manylinux wheel build arg to the setup command ?
    && python setup.py bdist_wheel \
    && cp dist/dpnp*.whl $TMPDIR \
    && rm -Rf $DPNP_BUILD_DIR

FROM build_environment AS numba_dpex_builder
SHELL ["/bin/bash", "-c"]
ARG TMPDIR
ARG ONEAPI_INSTALL_DIR
ARG NUMBA_DPEX_BUILD_DIR=$TMPDIR/numba_dpex
ARG NUMBA_DPEX_GIT_BRANCH
ARG NUMBA_DPEX_GIT_URL
ARG INTEL_PYPI_URL
ARG BASE_PYPI_URL
COPY --from=dpctl_builder $TMPDIR/dpctl*.whl $TMPDIR
COPY --from=dpnp_builder $TMPDIR/dpnp*.whl $TMPDIR
RUN pip install -U -i $INTEL_PYPI_URL --extra-index-url $BASE_PYPI_URL \
        $TMPDIR/dpctl*.whl $TMPDIR/dpnp*.whl \
    && pip cache purge \
    && mkdir -p $NUMBA_DPEX_BUILD_DIR \
    && cd $NUMBA_DPEX_BUILD_DIR \
    && git clone --recursive -b $NUMBA_DPEX_GIT_BRANCH --depth 1 $NUMBA_DPEX_GIT_URL . \
    && source $ONEAPI_INSTALL_DIR/setvars.sh \
    # HACK: currently, there is an issue with the bdist_wheel configuration for numba_dpex
    # that causes missing files in the final tarball.
    # The workaround consists in triggering building steps by running `setup.py develop`
    # before running `setup.py bdist_wheel`.
    # See https://github.com/ogrisel/sklearn-numba-dpex/issues/5
    && python setup.py develop \
    # XXX: is it needed to pass manylinux wheel build arg to the setup command ?
    && python setup.py bdist_wheel \
    && cp dist/numba_dppy*.whl $TMPDIR \
    && rm -Rf $NUMBA_DPEX_BUILD_DIR

FROM base_config AS runtime_environment
ARG ONEAPI_INSTALL_DIR
COPY --from=python_builder $PYENV_ROOT $PYENV_ROOT
COPY --from=python_builder $VIRTUAL_ENV $VIRTUAL_ENV
COPY --from=oneapi_dpcpp_binaries_installer $ONEAPI_INSTALL_DIR $ONEAPI_INSTALL_DIR


# Install the 3 packages and the spirv-tools binaries and headers
FROM runtime_environment AS numba_dpex_dev_installer
ARG TMPDIR
ARG INTEL_PYPI_URL
ARG BASE_PYPI_URL
ARG INTEL_NUMPY_VERSION
ARG INTEL_SCIPY_VERSION
ARG INTEL_NUMBA_VERSION
ARG PACKAGING_VERSION
COPY --from=dpctl_builder $TMPDIR/dpctl*.whl $TMPDIR
COPY --from=dpnp_builder $TMPDIR/dpnp*.whl $TMPDIR
COPY --from=numba_dpex_builder $TMPDIR/numba_dppy*.whl $TMPDIR
RUN pip install -U -i $INTEL_PYPI_URL "numpy${INTEL_NUMPY_VERSION}" \
        "scipy${INTEL_SCIPY_VERSION}" --no-deps \
    && pip install -U -i $INTEL_PYPI_URL --extra-index-url $BASE_PYPI_URL \
        "numpy${INTEL_NUMPY_VERSION}" "numba${INTEL_NUMBA_VERSION}" \
        "scipy${INTEL_SCIPY_VERSION}" $TMPDIR/dpctl*.whl $TMPDIR/dpnp*.whl \
        $TMPDIR/numba_dppy*.whl packaging${PACKAGING_VERSION} \
        # HACK this package is not in the dependency tree of any of the top
        # packages but probably should, in the meantime we add it manually
        mkl-dpcpp \
    && pip cache purge

FROM runtime_environment AS numba_dpex_dev
COPY --from=numba_dpex_dev_installer $VIRTUAL_ENV $VIRTUAL_ENV
RUN apt-get update --quiet && apt-get install -y --no-install-recommends \
        spirv-tools spirv-headers \
    && rm -Rf /var/lib/apt/lists/*
CMD ["/bin/bash"]
