import numba_dpex as dpex
import numba
from numba_dpex.compiler import (
    Kernel,
    JitKernel,
    _raise_datatype_mixed_error,
    USMNdArrayType,
    cfd_ctx_mgr_wrng_msg,
    dpctl_version,
    IndeterminateExecutionQueueError,
    _raise_no_device_found_error,
)
import dpctl
import warnings


def kernel_get_task(self, *args, sycl_queues, **kwargs):
    kernelargs = []
    internal_device_arrs = []
    for ty, val, access_type in zip(
        self.argument_types, args, self.ordered_arg_access_types
    ):
        self._unpack_argument(
            ty,
            val,
            self.sycl_queue,
            kernelargs,
            internal_device_arrs,
            access_type,
        )

    sycl_queues.append(self.sycl_queue)

    # the only difference with __call__ is that we return instead of calling
    # and the repack step is skipped (but not needed for dpctl.tensor inputs anyway)
    return self.kernel, kernelargs, self.global_size, self.local_size


Kernel.get_task = kernel_get_task


def jitkernel_get_task(self, *args, **kwargs):
    assert not kwargs, "Keyword Arguments are not supported"
    *args, sycl_queues = args

    argtypes = self._get_argtypes(*args)
    compute_queue = None

    # Get the array type and whether all array are of same type or not
    array_type, uniform = self._datatype_is_same(argtypes)
    if not uniform:
        _raise_datatype_mixed_error(argtypes)

    if type(array_type) == USMNdArrayType:
        if dpctl.is_in_device_context():
            warnings.warn(cfd_ctx_mgr_wrng_msg)

        queues = []
        for i, argtype in enumerate(argtypes):
            if type(argtype) == USMNdArrayType:
                memory = dpctl.memory.as_usm_memory(args[i])
                if dpctl_version < (0, 12):
                    queue = memory._queue
                else:
                    queue = memory.sycl_queue
                queues.append(queue)

        # dpctl.utils.get_exeuction_queue() checks if the queues passed are
        # equivalent and returns a SYCL queue if they are equivalent and
        # None if they are not.
        compute_queue = dpctl.utils.get_execution_queue(queues)
        if compute_queue is None:
            raise IndeterminateExecutionQueueError(
                "Data passed as argument are not equivalent. Please "
                "create dpctl.tensor.usm_ndarray with equivalent SYCL queue."
            )

    if compute_queue is None:
        try:
            compute_queue = dpctl.get_current_queue()
        except:
            _raise_no_device_found_error()

    kernel = self.specialize(argtypes, compute_queue)
    cfg = kernel.configure(kernel.sycl_queue, self.global_size, self.local_size)
    # the only difference with __call__ is that we return instead of calling
    return cfg.get_task(*args, sycl_queues=sycl_queues)


JitKernel.get_task = jitkernel_get_task

device = dpctl.select_default_device()


array_in1 = dpctl.tensor.empty(sh=(4,), device=device)
array_in2 = dpctl.tensor.empty(sh=(8,), device=device)

f32zero = numba.float32(0)
f32one = numba.float32(1)
f32two = numba.float32(2)


@dpex.kernel((numba.float32[:],))
def zeros(array):
    i = dpex.get_global_id(0)
    array[i] = f32zero


@dpex.kernel((numba.float32[:],))
def add_1(array):
    i = dpex.get_global_id(0)
    array[i] += f32one


@dpex.kernel((numba.float32[:],))
def add_2(array):
    i = dpex.get_global_id(0)
    array[i] += f32two


queues = []

zeros_kernel1_task = (zeros[4, 2]).get_task(array_in1, queues)
add_1_kernel1_task = (add_1[4, 2]).get_task(array_in1, queues)
add_2_kernel1_task = (add_2[4, 2]).get_task(array_in1, queues)
zeros_kernel2_task = (zeros[8, 8]).get_task(array_in2, queues)
add_1_kernel2_task = (add_1[8, 8]).get_task(array_in2, queues)
add_2_kernel2_task = (add_2[8, 8]).get_task(array_in2, queues)

queue = dpctl.utils.get_execution_queue(queues)


def async_iter():
    for i in range(100):
        zeros_kernel1_event = queue.submit(*zeros_kernel1_task)
        add_1_kernel1_event = queue.submit(
            *add_1_kernel1_task, dEvents=[zeros_kernel1_event]
        )
        add_2_kernel1_event = queue.submit(
            *add_2_kernel1_task, dEvents=[add_1_kernel1_event]
        )
        zeros_kernel2_event = queue.submit(*zeros_kernel2_task)
        add_2_kernel2_event = queue.submit(
            *add_2_kernel2_task, dEvents=[zeros_kernel2_event]
        )
        queue.wait()


def sync_iter():
    for i in range(100):
        zeros[4, 2](array_in1)
        add_1[4, 2](array_in1)
        add_2[4, 2](array_in1)
        zeros[8, 8](array_in2)
        add_2[8, 8](array_in2)
