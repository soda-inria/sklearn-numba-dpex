# TODO: many auxilliary kernels in this module might be better optimized and we could
# benchmark alternative implementations for each of them, that could include using
# numba + dpnp to directly leverage kernels that are shipped in dpnp to replace numpy
# methods.
# However, in light of our main goal that is bringing a GPU KMeans to scikit-learn, the
# importance of those TODOs is currently seen as secondary, since the execution time of
# those kernels is only a small fraction of the total execution time and the
# improvements that further optimizations can add will only be marginal. There is no
# doubt, though, that a lot could be learnt about kernel programming in the process.
