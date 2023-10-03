import inspect

from llvmlite import binding as ll
from numba_dpex import config

_native_optimize_final_module = None


def _load_numba_dpex_with_patches(with_patches=True):
    """This function hacks `numba_dpex` init to work around stability issues after
    `numba_dpex>=0.21.0dev2` bumps. It will be reverted when the official fixes are out.
    See the issue tracker at https://github.com/IntelPython/numba-dpex/issues/1152 .
    """
    # TODO: revert patches when https://github.com/IntelPython/numba-dpex/issues/1152
    # is fixed
    global _native_optimize_final_module

    from numba_dpex.core.codegen import SPIRVCodeLibrary

    if _native_optimize_final_module is None:
        _native_optimize_final_module = SPIRVCodeLibrary._optimize_final_module

    if not with_patches:
        SPIRVCodeLibrary._optimize_final_module = _native_optimize_final_module
        return

    _expected_optimize_final_module_code = """    def _optimize_final_module(self):
        # Run some lightweight optimization to simplify the module.
        pmb = ll.PassManagerBuilder()

        # Make optimization level depending on config.OPT variable
        pmb.opt_level = config.OPT

        pmb.disable_unit_at_a_time = False
        pmb.inlining_threshold = 2
        pmb.disable_unroll_loops = True
        pmb.loop_vectorize = False
        pmb.slp_vectorize = False

        pm = ll.ModulePassManager()
        pmb.populate(pm)
        pm.run(self._final_module)
"""

    _actual_optimize_final_module_code = inspect.getsource(
        _native_optimize_final_module
    )

    if _actual_optimize_final_module_code != _expected_optimize_final_module_code:
        raise RuntimeError(
            "Cannot apply patches to `numba_dpex` because the source code is has"
            " changed."
        )

    def _monkey_patch_optimize_final_module(self):
        # Run some lightweight optimization to simplify the module.
        pmb = ll.PassManagerBuilder()

        # Make optimization level depending on config.OPT variable
        pmb.opt_level = config.OPT

        pmb.disable_unit_at_a_time = False
        pmb.disable_unroll_loops = True
        pmb.loop_vectorize = False
        pmb.slp_vectorize = False

        pm = ll.ModulePassManager()
        pmb.populate(pm)
        pm.run(self._final_module)

    SPIRVCodeLibrary._optimize_final_module = _monkey_patch_optimize_final_module
