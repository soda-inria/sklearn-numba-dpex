# sklearn-numba-dpex

Experimental plugin for scikit-learn to be able to run (some) estimators on Intel GPUs
via [numba-dpex](https://github.com/IntelPython/numba-dpex). Support for other GPU
constructors is also on the roadmap and depends on progress of interoperability features
of the `numba-dpex` stack.

This package requires working with the following branch of scikit-learn:

- `wip-engines` branch on https://github.com/ogrisel/scikit-learn

A step-by-step guide is provided in this README for installing `numba-dpex`, along with
the `wip-engines` branch of `scikit-learn` and this plugin from source`

🚧 TODO: package `wip-engines` and `sklearn-numba-dpex` to have everything installable
in `conda` with a single one-liner.

## List of Included Engines

- `sklearn.cluster.KMeans` for the standard LLoyd's algorithm on dense data arrays,
  including `kmeans++` support.

## Getting started:

### Step 1: Installing a `numba_dpex` environment

Getting started requires a working environment for using `numba_dpex`. Currently a
[conda install](#using-a-conda-installation) or a [docker image](#using-the-docker-image)
are available.

#### Using a conda installation

Conda does not currently support installation of the low-level runtime libraries for
GPUs, so the first part of the installation guide consists in installing those libraries
on the host system.

The second part consists in running conda commands that create the environment with
all the required packages and configuration. Note that the installation logic is a bit
complicated since it mixes packages from several conda channels `conda-forge`,
`dppy/label/dev`, and `intel`, some of which being experimental. Neither the builds nor
the channels are maintained by the `sklearn_numba_dpex` team and their level of
stability is unknown.

🚧 TODO: update the instructions to install everything from non-dev conda packages on
always up-to-date channels whenever it's available.

##### Install low-level runtime libraries for your GPU (1/2)

At this time, only Intel GPUs are supported.

###### Intel GPU runtime libraries

For Intel GPUs, two backends are available. You might want to install both of those,
and test if one gives better performances.

🚧 TODO: write a guide on how to select the device and the backend in a python script.

- **Intel Opencl for GPU**: the intel opencl runtime can be installed following
  [this link](https://github.com/intel/compute-runtime#installation-options).
  **WARNING**: for Ubuntu (confirmed for `focal` and `jammy`) the `apt`-based
  installation is broken, see https://github.com/IntelPython/dpctl/issues/887 .

   ⚠ Like whenever installing packages outside of official repositories, existing
   workarounds might make your system unstable and are not recommended outside of a
   containerized environment and/or for expert users.

   To not alter the `apt`-based version tree too much and risk other compatibility
   issues, there's a minimal workaround that consists in identifying the version
   that is officially supported by your OS (use [packages.ubuntu.com](https://packages.ubuntu.com/search?keywords=intel-opencl-icd&searchon=names))
   then download the corresponding build from the [Intel release page on github](https://github.com/intel/compute-runtime/releases)
   and adapt the following set of instructions accordingly:

   ```bash
    sudo apt-get install intel-opencl-icd  # install the whole dependency tree
    sudo apt-get remove intel-opencl-icd  # remove only the top-level package
    wget https://github.com/intel/compute-runtime/releases/download/22.14.22890/intel-opencl-icd_22.14.22890_amd64.deb
    sudo dpkg -i intel-opencl-icd_22.14.22890_amd64.deb  # replace the top-level package only
   ```

- **oneAPI level zero loader**: alternatively, or in addition, the oneAPI level zero
  backend can be used. This backend is more experimental, and is sometimes preferred
  over opencl. Source and `deb` archives are available
  [here](https://github.com/oneapi-src/level-zero/releases)

###### Give permissions to submit GPU workloads

Non-root users might lack permission to access the GPU device to submit workloads. Add
the current user to the `video` group and/or `render` group:

```bash
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER
```

##### Setup a conda environment for numba-dpex (2/2)

To setup a [conda](https://docs.conda.io/en/latest/) conda environment, run first
```bash
export CONDA_DPEX_ENV_NAME=my-dpex-env
conda create -n $CONDA_DPEX_ENV_NAME numba-dpex intel::dpcpp_linux-64 -c dppy/label/dev -c conda-forge -c intel
```
An additonal command is currently required to work around missing Intel CPU opencl
runtime activation:
```bash
conda env config vars set OCL_ICD_FILENAMES_RESET=1 OCL_ICD_FILENAMES=libintelocl.so -n $CONDA_DPEX_ENV_NAME
```

This will create an environment named `my-dpex-env` (that you can change to your
liking) containing the package `numba_dpex`, all of its dependencies, and adequate
configuration.

The corresponding development branch of scikit-learn must be installed from source and
that requires a separate conda environment. The following sequence of commands will
create the appropriate conda environment, build the scikit-learn binary, then remove
the environment:

```bash
conda activate $CONDA_DPEX_ENV_NAME
export DPEX_PYTHON_VERSION=$(python -c "import platform; print(platform.python_version())")
export DPEX_NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)")
conda create -n sklearn-dev -c conda-forge python==$(DPEX_PYTHON_VERSION) numpy==$(DPEX_NUMPY_VERSION) scipy cython joblib threadpoolctl pytest compilers
conda activate sklearn-dev
git clone https://github.com/ogrisel/scikit-learn -b wip-engines
cd scikit-learn
git checkout fdaf97b5b90e18fc63483de9455970123208c9bb
python setup.py bdist_wheel
conda activate $CONDA_DPEX_ENV_NAME
cd build/
pip install *.whl
unset DPEX_PYTHON_VERSION
unset DPEX_NUMPY_VERSION
conda deactivate
conda env remove -n sklearn-dev
cd ../../
rm -Rf scikit-learn
```

Finally, activate the environment with the command:
```bash
conda activate my-dpex-env
```

#### Using the docker image

Alternatively, a docker image is available and provides an up-to-date, one-command
install environment. You can either build it from the [Dockerfile](./docker/Dockerfile):

```bash
cd docker
DOCKER_BUILDKIT=1 docker build . -t my_tag
```

or pull the docker image from
[this publicly available repository](https://hub.docker.com/repository/docker/jjerphan/numba_dpex_dev):

```bash
docker pull jjerphan/numba_dpex_dev:latest
```

Run the container in interactive mode with your favorite docker flags, for example:

```bash
docker run --name my_container_name -it -v /my/host/volume/:/mounted/volume --device=/dev/dri my_tag
```

where `my_tag` would be `jjerphan/numba_dpex_dev:latest` if you pulled from the
repository.

⚠ The flag `--device=/dev/dri` is **mandatory** to enable the gpu within the container,
also the user starting the `docker run` command must have access to the gpu, see
[Give permissions to submit GPU workloads](#give-permissions-to-submit-gpu-workloads).

Unless using the flag `--rm` when starting a container, you can restart it after it was
exited, with the command:

```bash
sudo docker start -a -i my_container_name
```

Once you have loaded into the container, follow those instructions to install the
`wip-engines` branch of scikit-learn:

```bash
git clone https://github.com/ogrisel/scikit-learn -b wip-engines
cd scikit-learn
git checkout fdaf97b5b90e18fc63483de9455970123208c9bb
pip install -e .
cd ..
```

### Step 2: Check the installation of the environment was successfull

Once the environment you just installed with one of those two methods is activated,
you can inspect the available hardware:

```bash
python -c "import dpctl; print(dpctl.get_devices())"
```

this should print a list of available devices, including `cpu` and `gpu` devices, once
for each available backends (`opencl`, `level_zero`,...).

### Step 3: install this plugin

FIXME: currently, non-editable mode installation does not work.

When loaded into your `numba_dpex` + `scikit-learn` environment from previous steps,
run:

```bash
git clone https://github.com/soda-inria/sklearn-numba-dpex
cd sklearn-numba-dpex
pip install -e .
```

## Intended usage

See the `sklearn_numba_dpex/kmeans/tests` folder for example usage.

🚧 TODO: write some examples here instead.

### Running the tests

To run the tests run the following from the root of the `sklearn_numba_dpex` repository:

```bash
pytest sklearn_numba_dpex
```

To run the `scikit-learn` tests with the `sklearn_numba_dpex` engine you can run the
following:

```bash
SKLEARN_NUMBA_DPEX_TESTING_MODE=1 pytest --sklearn-engine-provider sklearn_numba_dpex --pyargs sklearn.cluster.tests.test_k_means
```

(change the `--pyargs` option accordingly to select other test suites).

The `--sklearn-engine-provider sklearn_numba_dpex` option offered by the sklearn pytest
plugin will automatically activate the `sklearn_numba_dpex` engine for all tests.

Tests covering unsupported features (that trigger
`sklearn.exceptions.FeatureNotCoveredByPluginError`) will be automatically marked as
_xfailed_.

### Running the benchmarks

Repeat the pip installation step exposed in [step 3](#step-3-install-this-plugin) with
the following edit:

```bash
pip install -e .[benchmark]
```

(i.e adding the _benchmark_ extra-require), followed by:

```bash
cd benckmark
python ./kmeans.py
```

to run a benchmark for different k-means implementations and print a short summary of
the performance.

Some parameters in the `__main__` section of the file `./benchmark/kmeans.py` are
exposed for quick edition (`n_clusters`, `max_iter`, `skip_slow`, ...).

### Notes about the preferred floating point precision (float32)

In many machine learning applications, operations using single-precision (float32)
floating point data require twice as less memory that double-precision (float64), are
regarded as faster, accurate enough and more suitable for GPU compute. Besides, most
GPUs used in machine learning projects are significantly faster with float32 than with
double-precision (float64) floating point data.

To leverage the full potential of GPU execution, it's strongly advised to use a float32
data type.

By default, unless specified otherwise numpy array are created with type float64, so be
especially careful to the type whenever the loader does not explicitly document the
type nor expose a type option.

Transforming NumPy arrays from float64 to float32 is also possible using
[`numpy.ndarray.astype`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html),
although it is less recommended to prevent avoidable data copies. `numpy.ndarray.astype`
can be used as follows:

```python
X = my_data_loader()
X_float32 = X.astype(float32)
my_gpu_compute(X_float32)
```
