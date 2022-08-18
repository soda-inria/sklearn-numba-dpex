# sklearn-numba-dpex

Experimental plugin for scikit-learn to be able to run (some estimators) on
Intel GPUs via [numba-dpex](https://github.com/IntelPython/numba-dpex).

DISCLAIMER: this is work in progres, do not expect this repo to be in a
workable state at the moment.

This requires working with the following branch of scikit-learn which is itself
not yet in a working state at the moment:

- `wip-engines` branch on https://github.com/ogrisel/scikit-learn 

## Getting started:

At this time, it is required to install the development versions of
dependencies.

TODO: update the instructions to install everything from non-dev conda packages
once available.

### Step 1: a conda env with Intel Python libraries

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

- Make sure you have installed the latest GPU drivers:

  https://dgpu-docs.intel.com/installation-guides/index.html

- On Linux, check that the i915 driver is properly loaded:

  ```
  $ lspci -nnk | grep i915
        Kernel driver in use: i915
        Kernel modules: i915
  ```

- On Linux, check that the current user is in the `render` group, for instance:

  ```
  $ groups
  ogrisel adm cdrom sudo dip plugdev render lpadmin lxd sambashare docker
  ```

  If not, add it with `sudo adduser $USER render`, logout and log back in, and check
  again.

- Install recent versions of the following runtime libraries (not yet available
  as conda packages): https://github.com/intel/compute-runtime/releases

More documentation on system configuration for Intel hardware and software available at:

https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-dpcpp-system-requirements.html

and in the discussion of this github issue:

https://github.com/IntelPython/dpnp/issues/1149


The `dpctl.lsplatform()` can also list version informations on your SYCL
runtime environment:

```
python -c "import dpctl; dpctl.lsplatform()"
```


### Step 2: install numba-dpex from source

Install and activate the Intel oneAPI DPC++ compiler shipped with the Intel oneAPI base toolkit:

https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

For instance, in Ubuntu, once the apt repo has been configured:

```
sudo apt update
sudo apt install intel-basekit
source /opt/intel/oneapi/compiler/latest/env/vars.sh
```

You can check that the `icx` command can be found in the PATH with `which icx`.

The install numba-dpex in the same conda env as previously:

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


### Step 3: install the `wip-engines` branch of scikit-learn

In the same conda env:

```
git clone https://github.com/ogrisel/scikit-learn
cd scikit-learn
git checkout wip-engines
pip install -e . --no-build-isolation
cd ..
```

Then install this plugin in that same env:

```
git clone https://github.com/ogrisel/sklearn-numba-dpex
cd sklearn-numba-dpex
pip install -e . --no-build-isolation
```

## Intended usage

See the `sklearn_numba_dpex/tests` folder for example usage.

TODO: write some doc here instead.
