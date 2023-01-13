import dpctl

dpctl_select_default_device = dpctl.select_default_device
native_dpex_spirv_generator_cmdline = None
native_dpex_compile_func = None
native_dpex_compile_func_template = None


def _load_numba_dpex_with_patches(with_spirv_fix=True, with_compiler_fix=True):
    """This function hacks `numba_dpex` init to work around issues after
    `dpctl>=0.14.1dev1` and `numba_dpex>=0.19.0` bumps. It will be
    reverted when the official fixes are out.
    See https://github.com/IntelPython/numba-dpex/pull/858 ,
    https://github.com/IntelPython/numba-dpex/issues/867 and
    https://github.com/IntelPython/numba-dpex/issues/868
    """
    global native_dpex_spirv_generator_cmdline
    global native_dpex_compile_func
    global native_dpex_compile_func_template

    def _patch_mock_dpctl_select_default_device():
        class _mock_device:
            is_host = False

        return _mock_device()

    try:
        # A better fix for this is already available in the development tree of
        # `numba_dpex` but it's not released yet.
        # See https://github.com/IntelPython/numba-dpex/pull/858
        dpctl.select_default_device = _patch_mock_dpctl_select_default_device
        import numba_dpex.config as dpex_config
        import numba_dpex.decorators as dpex_decorators
        import numba_dpex.spirv_generator as dpex_spirv_generator

        if native_dpex_spirv_generator_cmdline is None:
            native_dpex_spirv_generator_cmdline = dpex_spirv_generator.CmdLine

        if native_dpex_compile_func is None:
            native_dpex_compile_func = dpex_decorators.compile_func

        if native_dpex_compile_func_template is None:
            native_dpex_compile_func_template = dpex_decorators.compile_func_template

        # TODO; revert this once https://github.com/IntelPython/numba-dpex/issues/867
        # is fixed.
        def _dpex_compile_func(pyfunc, *args, **kwargs):
            pyfunc_id = id(pyfunc)
            pyfunc.__name__ += f"_{pyfunc_id}"
            pyfunc.__qualname__ += f"_{pyfunc_id}"
            return native_dpex_compile_func(pyfunc, *args, **kwargs)

        def _dpex_compile_func_template(pyfunc, *args, **kwargs):
            pyfunc_id = id(pyfunc)
            pyfunc.__name__ += f"_{pyfunc_id}"
            pyfunc.__qualname__ += f"_{pyfunc_id}"
            return native_dpex_compile_func_template(pyfunc, *args, **kwargs)

        if with_compiler_fix:
            dpex_decorators.compile_func = _dpex_compile_func
            dpex_decorators.compile_func_template = _dpex_compile_func_template
        else:
            dpex_decorators.compile_func = native_dpex_compile_func
            dpex_decorators.compile_func_template = native_dpex_compile_func_template

        # TODO: revert this once https://github.com/IntelPython/numba-dpex/issues/868
        # is fixed.
        class _CmdLine(dpex_spirv_generator.CmdLine):
            def generate(self, llvm_spirv_args, *args, **kwargs):
                if not dpex_config.NATIVE_FP_ATOMICS:
                    llvm_spirv_args = ["--spirv-max-version", "1.0"] + llvm_spirv_args
                super().generate(llvm_spirv_args, *args, **kwargs)

        if with_spirv_fix:
            dpex_spirv_generator.CmdLine = _CmdLine
        else:
            dpex_spirv_generator.CmdLine = native_dpex_spirv_generator_cmdline

    finally:
        dpctl.select_default_device = dpctl_select_default_device
