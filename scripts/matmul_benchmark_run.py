import numpy as np

from time import perf_counter

import dpctl
import dpctl.tensor as dpt
from sklearn_numba_dpex.common.matmul import make_matmul_2d_kernel

import json

n = m = k = 2048
dtype = np.float32

rng = np.random.default_rng(123)

if __name__ == "__main__":
    from argparse import ArgumentParser
    argparser = ArgumentParser()
    argparser.add_argument(
        "--work-group-size",
        default=None,
        type=int
    )
    argparser.add_argument(
        "--sub-group-size",
        default=None,
        type=int
    )
    argparser.add_argument(
        "--arithmetic-intensity-multiplier-X",
        default=None,
        type=int
    )
    argparser.add_argument(
        "--arithmetic-intensity-multiplier-Y",
        default=None,
        type=int
    )
    argparser.add_argument(
        "--private-Y-t-sliding-window-width",
        default=None,
        type=int
    )
    args = argparser.parse_args()
    work_group_size = args.work_group_size
    sub_group_size = args.sub_group_size
    arithmetic_intensity_multiplier_X = args.arithmetic_intensity_multiplier_X
    arithmetic_intensity_multiplier_Y = args.arithmetic_intensity_multiplier_Y
    private_Y_t_sliding_window_width = args.private_Y_t_sliding_window_width
    
    try:
        parameters = dict(
            work_group_size = work_group_size,
            sub_group_size = sub_group_size,
            arithmetic_intensity_multiplier_X = arithmetic_intensity_multiplier_X,
            arithmetic_intensity_multiplier_Y = arithmetic_intensity_multiplier_Y,
            private_Y_t_sliding_window_width = private_Y_t_sliding_window_width
        )
 
        device = dpctl.SyclDevice()
    
        nb_private_memory, matmul_2d_kernel = make_matmul_2d_kernel(
            n, m, k, dtype, device, **parameters
        )
        
        X = rng.random((n, k)).astype(dtype)
        Y = rng.random((k, m)).astype(dtype)
            
        Y = np.random.randn(k, m).astype(dtype)
        result = np.empty(shape=(n, m), dtype=dtype)
        # %timeit np.dot(X, Y, out=result)
        
        X_n_rows = X.shape[0]
        Y_t_n_rows = Y.shape[1]
        n_cols = X.shape[1]
        X = dpt.asarray(X, order="C")
        Y_t = dpt.asarray(Y.T, order="C")
        device = X.device.sycl_device
        result = dpt.zeros((X_n_rows, Y_t_n_rows), dtype, order="C", device=device)

                
        matmul_2d_kernel(X, Y_t, result)
        t0 = perf_counter()
        for i in range(100):
            matmul_2d_kernel(X, Y_t, result)
        t1 = perf_counter()

        dt = t1 - t0

        parameters["nb_private_memory"] = nb_private_memory
        print(f"{parameters} -- {dt}")

        parameters["dt"] = dt

        with open("./benchmark_results.json", "a") as result_file:
            result_file.write(json.dumps(parameters) + "\n")

    except Exception:
        pass

