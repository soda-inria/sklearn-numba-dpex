from setuptools import setup


LONG_DESCRIPTION = """\
TODO
"""


setup(
    name="sklearn-numba-dppy",
    maintainer="Olivier Grisel",
    maintainer_email="olivier.grisel@ensta.org",
    description="numba-dppy Computational Engine for scikit-learn",
    license="BSD 3-Clause License",
    url="https://github.com/ogrisel/sklearn-numba-dppy",
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
    install_requires=["scikit-learn", "numba-dppy", "dpnp"],
    entry_points={
        "sklearn_engines": [
            "kmeans=sklearn_numba_dppy.cluster.kmeans:KMeansEngine"
        ]
    }
)

