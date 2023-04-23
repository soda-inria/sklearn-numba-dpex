import numpy as np

from time import perf_counter

import dpctl
import dpctl.tensor as dpt
from sklearn_numba_dpex.common.topk import topk

import json

import os

n = m = k = 2048
dtype = np.float32

rng = np.random.default_rng(123)

if __name__ == "__main__":
    from argparse import ArgumentParser

    argparser = ArgumentParser()
    argparser.add_argument("--work-group-size", default=None, type=int)
    argparser.add_argument("--sub-group-size", default=None, type=int)

    args = argparser.parse_args()
    work_group_size = args.work_group_size
    sub_group_size = args.sub_group_size

    try:
        parameters = dict(
            work_group_size=work_group_size,
            sub_group_size=sub_group_size,
        )
        os.environ["TOPK_WORK_GROUP_SIZE"] = work_group_size
        os.environ["TOPK_SUB_GROUP_SIZE"] = sub_group_size

        n = 2**30
        k = 100
        dtype = np.float32

        X = np.random.randn(n).astype(dtype)
        X = dpt.asarray(X, order="C", dtype=dtype)

        device = dpctl.SyclDevice()
        t0 = perf_counter()
        for i in range(100):
            topk(X, 100)
        t1 = perf_counter()

        dt = t1 - t0

        print(f"{parameters} -- {dt}")

        parameters["dt"] = dt

        with open("./benchmark_results.json", "a") as result_file:
            result_file.write(json.dumps(parameters) + "\n")

    except Exception:
        pass
