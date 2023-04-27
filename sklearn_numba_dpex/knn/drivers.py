import math

import dpctl.tensor as dpt
import numpy as np

from sklearn_numba_dpex.common.matmul import make_matmul_2d_kernel
from sklearn_numba_dpex.common.topk import _make_get_topk_kernel


def kneighbors(
    query,
    data,
    n_neighbors,
    metric="euclidean",
    return_distance=False,
    maximum_compute_buffer_size=1073741824,  # 1 GiB
):
    n_queries, n_features = query.shape
    n_samples = data.shape[0]
    compute_dtype = query.dtype.type
    compute_dtype_itemsize = np.dtype(compute_dtype).itemsize
    device = query.device.sycl_device

    index_itemsize = np.dtype(np.int64).itemsize

    pairwise_distance_required_bytes_per_query = n_samples * compute_dtype_itemsize

    # TODO: better way to ensure this remains synchronized with future changes to
    # TopK ?
    topk_required_bytes_per_query = (
        (n_neighbors + 1 + 1 + 1) * compute_dtype_itemsize
    ) + ((1 + 1 + 2 + 4) * index_itemsize)
    if return_distance:
        topk_required_bytes_per_query += n_neighbors * index_itemsize

    total_required_bytes_per_query = (
        pairwise_distance_required_bytes_per_query + topk_required_bytes_per_query
    )

    max_slice_size = maximum_compute_buffer_size / total_required_bytes_per_query

    if max_slice_size < 1:
        raise RuntimeError("Buffer size is too small")

    slice_size = min(math.floor(max_slice_size), n_queries)

    n_slices = math.ceil(n_queries / slice_size)
    n_full_slices = n_slices - 1
    last_slice_size = ((n_queries - 1) % slice_size) + 1

    def pairwise_distance_multiply_fn(x, y):
        diff = x - y
        return -(diff * diff)

    squared_pairwise_distance_kernel = make_matmul_2d_kernel(
        slice_size,
        n_samples,
        n_features,
        compute_dtype,
        device,
        multiply_fn=pairwise_distance_multiply_fn,
    )
    last_slice_squared_pairwise_distance_kernel = make_matmul_2d_kernel(
        last_slice_size,
        n_samples,
        n_features,
        compute_dtype,
        device,
        multiply_fn=pairwise_distance_multiply_fn,
    )
    squared_pairwise_distance_buffer = dpt.empty(
        (slice_size, n_samples), dtype=compute_dtype, device=device
    )
    last_slice_squared_pairwise_distance_buffer = dpt.asarray(
        squared_pairwise_distance_buffer[:last_slice_size], copy=False
    )

    _, get_topk_kernel = _make_get_topk_kernel(
        n_neighbors, (slice_size, n_samples), compute_dtype, device, output="idx"
    )

    result_slice_buffer = dpt.empty(
        (slice_size, n_neighbors), dtype=compute_dtype, device=device
    )
    result = dpt.empty((n_queries, n_neighbors), dtype=compute_dtype, device=device)

    slice_sample_idx = 0
    for _ in range(n_full_slices):
        query_slice = query[slice_sample_idx : (slice_sample_idx + slice_size)]
        squared_pairwise_distance_kernel(
            query_slice, data, squared_pairwise_distance_buffer
        )

        # # TODO: This should work instead and avoir having an intermediate buffer
        # # Might be fixed with latest numba_dpex commit and load_numba_dpex patches
        # # removal ?
        # result_slice = dpt.asarray(
        #     result[slice_sample_idx : (slice_sample_idx + slice_size)], copy=False
        # )
        # get_topk_kernel(squared_pairwise_distance_buffer, result_slice)

        get_topk_kernel(squared_pairwise_distance_buffer, result_slice_buffer)
        result[
            slice_sample_idx : (slice_sample_idx + slice_size)
        ] = result_slice_buffer[:]

        slice_sample_idx += slice_size

    query_slice = query[slice_sample_idx:]
    last_slice_squared_pairwise_distance_kernel(
        query_slice, data, last_slice_squared_pairwise_distance_buffer
    )
    result_slice = dpt.asarray(result[slice_sample_idx:], copy=False)

    get_topk_kernel(
        squared_pairwise_distance_buffer,
        result_slice,
    )

    return result
