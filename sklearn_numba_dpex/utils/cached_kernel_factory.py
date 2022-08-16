_KERNEL_CACHE = dict()


def cached_kernel_factory(kernel_factory):

    factory_name = kernel_factory.__name__

    def _cached_kernel_factory(*args, **kwargs):

        cache_key = (factory_name, *args, *sorted(kwargs.items()))
        kernel = _KERNEL_CACHE.setdefault(
            cache_key, kernel_factory(*args, **kwargs)
        )
        return kernel

    return _cached_kernel_factory
