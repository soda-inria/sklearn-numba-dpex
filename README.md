# sklearn-numba-dpex

Experimental plugin for scikit-learn to be able to run (some estimators) on Intel GPUs
via [numba-dpex](https://github.com/IntelPython/numba-dpex). Support for other GPU
constructors is also on the roadmap and depends on progress of interoperability features
of the `numba-dpex` stack.

This package requires working with the following branch of scikit-learn:

- `wip-engines` branch on https://github.com/ogrisel/scikit-learn

## List of Included Engines

- `sklearn.cluster.KMeans` for the standard LLoyd's algorithm on dense data arrays,
  including `kmeans++` support.

  No implementation of k-means for sparse input data or for the "Elkan" algorithm is
  included at the moment or planned to be.

## Getting started:

### Step 1: Installing a `numba_dpex` environment

Getting started requires a working environment for using `numba_dpex`. Currently a
[conda install](#using-a-conda-installation) or a
[docker image](#using-the-docker-image) are available.

#### Using a conda installation

Conda does not currently support installation of the low-level runtime libraries for
GPUs, so the first part of the installation guide consists in installing those libraries
on the host system.

The second part consists in running conda commands that create the environment with
all the required packages and configuration. Note that while the installation is a bit
complicated since it mixes packages from several conda channels `conda-forge`,
`dppy/label/dev`, and `intel`, some of which being experimental, neither the builds nor
the channels are maintained by the `sklearn_numba_dpex` and their level of stability is
unknown.

TODO: update the instructions to install everything from non-dev conda packages on
always up-to-date channels whenever it's available.

##### Install low-level runtime libraries for your GPU (1/2)

At this time, only Intel GPUs are supported.

###### Intel GPU runtime libraries

For Intel GPUs, two backends are available. You might want to install both of those,
and test if one gives better performances.

TODO: write a guide on how to select the device and the backend in a python script.

- **intel opencl for gpu**: the intel opencl runtime can be installed following
  [this link](https://github.com/intel/compute-runtime#installation-options) . For apt
  based linux distributions, for example it can be installed using the package manager:
  ```
  $ apt-get install intel-opencl-icd
  ```
- **oneAPI level zero loader**: alternatively, or in addition, the oneAPI level zero
  backend can be used. This backend is more experimental, and is sometimes preferred
  over opencl. Source and `deb` archives are available
  [here](https://github.com/oneapi-src/level-zero/releases)

###### Give permissions to submit GPU workloads

Non-root users might lack permission to access the GPU device to submit workloads. Add
those users to the `video` group and/or `render` group:

```
$ sudo usermod -a -G video my_username
$ sudo usermod -a -G render my_username
```

where `my_username` should be the username of your current session, or any other user
you want to give permissions to.

##### Setup a conda environment for numba-dpex (2/2)

The following [conda](https://docs.conda.io/en/latest/) commands:

```
$ conda create -n my-dpex-env numba-dpex -c conda-forge -c dppy/label/dev -c intel
# The following command is currently required to work around missing Intel CPU opencl
# runtime activation
$ conda env config vars set OCL_ICD_FILENAMES_RESET=1 OCL_ICD_FILENAMES=libintelocl.so -n my-dpex-env
```

will create an environment named `my-dpex-env` (that you can change to your liking)
containing the package `numba_dpex`, all of its dependencies, and adequate
configuration.

Activate the environment with the command:
```
$ conda activate my-dpex-env
```

#### Using the docker image

Alternatively, a docker image is available and provides an up-to-date, one-command
install environment. You can either build it from the [Dockerfile](./docker/Dockerfile):

```
$ cd docker
$ DOCKER_BUILDKIT=1 docker build . -t my_tag
```

or pull the docker image from
[this publicly available repository](https://hub.docker.com/repository/docker/jjerphan/numba_dpex_dev):

```
$ docker pull jjerphan/numba_dpex_dev:latest
```

Run the container in interactive mode with your favorite docker flags, for example:

```
$ docker run --name my_container_name -it -v /my/host/volume/:/mounted/volume --device=/dev/dri my_tag
```

where `my_tag` would be `jjerphan/numba_dpex_dev:latest` if you pulled from the
repository.

⚠ The flag `--device=/dev/dri` is **mandatory** to enable the gpu within the container,
also the user starting the `docker run` command must have access to the gpu, see
[Give permissions to submit GPU workloads](#give-permissions-to-submit-gpu-workloads).

Unless using the flag `--rm` when starting a container, you can restart it after it was
exited, with the command:

```
$ sudo docker start -a -i my_container_name
```

### Step 2: Check the installation of the environment was successfull

Once inside the environment you just installed with one of those two methods, you can
check that the environment works by introspecting the available hardware:

```
$ python -c "import dpctl; print(dpctl.get_devices())"
```

this should print a list of available devices, including `cpu` and `gpu` devices, once
for each available backends (`opencl`, `level_zero`,...).

### Step 3: install the `wip-engines` branch of scikit-learn

TODO: rather than expecting user to manually install this branch, release an end-to-end
conda build.

Once you have loaded into a `numba_dpex` environment, follow those instructions:

```
git clone https://github.com/ogrisel/scikit-learn
cd scikit-learn
git checkout wip-engines
pip install -e .
cd ..
```

### Step 4: install this plugin

TODO: rather than expecting user to manually install this branch, release an end-to-end
conda build.

FIXME: currently, non-editable mode installation does not work.

```
git clone https://github.com/soda-inria/sklearn-numba-dpex
cd sklearn-numba-dpex
pip install -e .
```

## Intended usage

See the `sklearn_numba_dpex/kmeans/tests` folder for example usage.

TODO: write some examples here instead.

### Running the tests

To run the tests run the following:

```
pytest sklearn_numba_dpex/
```

To run the `scikit-learn` tests with the `sklearn_numba_dpex` engine you can run the
following:

```
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

```
pip install -e .[benchmark]
```

(i.e adding the _benchmark_ extra-require), followed by:

```
cd benckmark
python ./kmeans.py
```

to run a benchmark for different k-means implementations and print a short summary of
the performance.

Some parameters in the `__main__` section of the file `./benchmark/kmeans.py` are
exposed for quick edition (`n_clusters`, `max_iter`, `skip_slow`, ...).

### Notes about the preferred floating point precision

In many machine learning applications, operations using single-precision (float32)
floating point data require twice as less memory that double-precision (float64), are
regarded as faster, accurate enough and more suitable for GPU compute. Besides, most
GPUs used in machine learning projects are significantly faster with float32 than with
double-precision (float64) floating point data.

To leverage the full potential of GPU execution, it's strongly advised to use a data
loader that loads float32 data. By default, unless specified otherwise numpy array are
created with type float64, so be especially careful to the type whenever the loader
does not explicitly document the type nor expose a type option.

Although it's less recommended to prevent avoidable data copies, it's also possible to
transform float64 numpy arrays into float32 arrays using the
[numpy.ndarray.astype](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html)
type converter following this example:

```
X = my_data_loader()
X_float32 = X.astype(float32)
my_gpu_compute(X_float32)
```
