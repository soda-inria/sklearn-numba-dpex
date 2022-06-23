# sklearn-numba-dpex

Experimental plugin for scikit-learn to be able to run (some estimators) on
Intel GPUs via [numba-dpex](https://github.com/IntelPython/numba-dpex).

DISCLAIMER: this is work in progres, do not expect this repo to be in a
workable state at the moment.

This requires working with the following branch of scikit-learn which is itself
not yet in a working state at the moment:

- `wip-engines` branch on https://github.com/ogrisel/scikit-learn 

## Getting started:

```
conda create -n dpnp install -c intel dpnp dpctl numba-dppy cython
conda activate dpnp
python -c "import dpctl; print(dpctl.get_devices())"
```

If you do not see any device (CPU or GPU), please have a look at:
https://github.com/IntelPython/dpnp/issues/1149

Then build the `wip-engines` branch of scikit-learn in that env:

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
