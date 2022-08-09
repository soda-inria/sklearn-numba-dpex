from setuptools import setup


LONG_DESCRIPTION = """\
TODO
"""


setup(
    name="sklearn-numba-dpex",
    maintainer="Olivier Grisel",
    maintainer_email="olivier.grisel@ensta.org",
    description="Computational Engine for scikit-learn based on numba-dpex",
    license="BSD 3-Clause License",
    url="https://github.com/ogrisel/sklearn-numba-dpex",
    version="0.1.0",
    long_description=LONG_DESCRIPTION,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Development Status :: 4 - Beta",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    python_requires=">=3.8",
    # TODO: replace "numba-dppy" by "numba-dpex" once released
    install_requires=["scikit-learn", "numba-dppy", "dpnp"],
    packages=["sklearn_numba_dpex"],
    entry_points={
        "sklearn_engines": ["kmeans=sklearn_numba_dpex.cluster.kmeans:KMeansEngine"]
    },
)
