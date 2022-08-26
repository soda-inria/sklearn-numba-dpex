# sklearn-numba-dpex

Experimental plugin for scikit-learn to be able to run (some estimators) on
Intel GPUs via [numba-dpex](https://github.com/IntelPython/numba-dpex).

DISCLAIMER: this is work in progres, do not expect this repo to be in a
workable state at the moment.

This requires working with the following branch of scikit-learn which is itself
not yet in a working state at the moment:

- `wip-engines` branch on https://github.com/ogrisel/scikit-learn 

## Getting started:

### Step 1: Installing a `numba_dpex` environment

Getting started requires a working environment for using `numba_dpex`. Currently a [conda install](#using-a-conda-installation) or a [docker image](#using-the-docker-image) are available. The most stable and recommended environment at the moment is using the docker image.

#### Using a conda installation

TODO: update the instructions to install everything from non-dev conda packages
once available.

##### A conda env with Intel Python libraries (1/2)

Let's create a dedicated conda env with the development versions
of the Intel Python libraries (from the `dppy/label/dev` channel).

```
conda create -n dppy-dev dpnp numba cython spirv-tools -c dppy/label/dev -c intel --override-channels
```

Let's activate it and introspect the available hardware:

```
conda activate dppy-dev
python -c "import dpctl; print(dpctl.get_devices())"
```

If you do not not see any CPU device, try again with the following after
setting the `SYCL_ENABLE_HOST_DEVICE=1` environment variable, for instance:

```
SYCL_ENABLE_HOST_DEVICE=1 python -c "import dpctl; print(dpctl.get_devices())"
```

If you have an Intel GPU and it is not detected, check the
following steps:

- Make sure you have installed the [latest GPU drivers](https://dgpu-docs.intel.com/installation-guides/index.html)

- On Linux, check that the i915 driver is properly loaded:

  ```
  $ lspci -nnk | grep i915
        Kernel driver in use: i915
        Kernel modules: i915
  ```

- On Linux, check that the current user is in the `render` group, for instance:

  ```
  $ groups
  myuser adm cdrom sudo dip plugdev render lpadmin lxd sambashare docker
  ```

  If not, add it with `sudo adduser $USER render`, logout and log back in, and check
  again.

- Install recent versions of the [runtime libraries](https://github.com/intel/compute-runtime/releases) (not yet available
  as conda packages)

For more in-depth information, you can refer to the [guides for system configuration for Intel hardware and software](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-dpcpp-system-requirements.html) and to the discussions in this github issue https://github.com/IntelPython/dpnp/issues/1149


The `dpctl.lsplatform()` can also list version informations on your SYCL
runtime environment:

```
python -c "import dpctl; dpctl.lsplatform()"
```

##### Install numba-dpex from source (2/2)

Install and activate the Intel oneAPI DPC++ compiler shipped with the [Intel oneAPI base toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).

For instance, in Ubuntu, once the apt repo has been configured:

```
sudo apt update
sudo apt install intel-basekit
source /opt/intel/oneapi/compiler/latest/env/vars.sh
```

Ensure that the `icx` command can be found in the PATH with the command:

```
which icx
```

Install numba-dpex in the same conda env as previously:

```
git clone https://github.com/IntelPython/numba-dpex/
cd numba-dpex
pip install -e . --no-build-isolation
```

Important: in order to use `numba-dpex`, the `llvm-spirv` compiler is required
to be in the PATH. This can be achieved with:

```
$ export PATH=/opt/intel/oneapi/compiler/latest/linux/bin-llvm:$PATH
```

#### Using the docker image

A docker image is available and provides an up-to-date, one-command install environment. You can either build it from the [Dockerfile](./docker/Dockerfile) :

```
$ cd docker
$ docker build . -t my_tag
```

or pull the docker image from [this publicly available repository](https://hub.docker.com/repository/docker/jjerphan/numba_dpex_dev):

```
$ docker pull jjerphan/numba_dpex_dev:latest
```

Run the container in interactive mode with your favorite docker flags, for example:

```
$ docker run --name my_container_name -it -v /my/host/volume/:/mounted/volume --device=/dev/dri my_tag
```

where `my_tag` would be `jjerphan/numba_dpex_dev:latest` if you pulled from the repository.

⚠ The flag `--device=/dev/dri` is **mandatory** to enable the gpu within the container, also the user starting the `docker run` command must have access to the gpu, e.g. by being a member of the `render` group.

Unless using the flag `--rm` when starting a container, you can restart it after it was exited, with the command:

```
sudo docker start -a -i my_container_name
```

Once inside the container, you can check that the environment works: the command

```
python -c "import dpctl; print(dpctl.get_devices())"
```

will introspect the available hardware, and should display working `opencl` cpu and gpu devices, and `level_zero` gpu devices.

### Step 2: install the `wip-engines` branch of scikit-learn

Once you have loaded into a `numba_dpex` development environment, following one of the two previous guides, follow those instructions:

```
git clone https://github.com/ogrisel/scikit-learn
cd scikit-learn
git checkout wip-engines
pip install -e . --no-build-isolation
cd ..
```

Then install this plugin in that same env:

```
git clone https://github.com/soda-inria/sklearn-numba-dpex
cd sklearn-numba-dpex
pip install -e . --no-build-isolation
```

## Intended usage

See the `sklearn_numba_dpex/tests` folder for example usage.

TODO: write some doc here instead.

### Running the benchmarks

The commands:

```
cd benckmark
python ./kmeans.py
```

will run a benchmark for different k-means implementations and print a short summary of the performances.

Some parameters in the `__main__` section of the file `./benchmark/kmeans.py` are exposed for quick edition (`n_clusters`, `max_iter`, `skip_slow`, ...).

### Notes about the preferred floating point precision

In many machine learning applications, operations using single-precision (float32) floating point data require twice as less memory that double-precision (float64), are regarded as faster, accurate enough and more suitable for GPU compute. Besides, most GPUs used in machine learning projects are significantly faster with float32 than with double-precision (float64) floating point data.

To leverage the full potential of GPU execution, it's strongly advised to use a data loader that loads float32 data. By default, unless specified otherwise numpy array are created with type float64, so be especially careful to the type whenever the loader does not explicitly document the type nor expose a type option.

Although it's less recommended to prevent avoidable data copies, it's also possible to transform float64 numpy arrays into float32 arrays using the [numpy.ndarray.astype](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html) type converter following this example:

```
X = my_data_loader()
X_float32 = X.astype(float32)
my_gpu_compute(X_float32)
```
