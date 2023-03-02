from dataclasses import dataclass
from weakref import WeakKeyDictionary, WeakValueDictionary

_native_dpex_typeof_helper = None
_native_Packer_unpack_usm_array = None


@dataclass(frozen=True)
class _ObjectId:
    id: int


def _load_numba_dpex_with_patches(with_patches=True):
    """This function hacks `numba_dpex` init to work around performance issues after
    `numba_dpex>=0.20.0dev3` bumps. It will be reverted when the official fixes are out.
    See the issue tracker at https://github.com/IntelPython/numba-dpex/issues/945 .
    """
    # TODO: revert patches when https://github.com/IntelPython/numba-dpex/issues/945 is
    # fixed
    global _native_dpex_typeof_helper
    global _native_Packer_unpack_usm_array

    from numba_dpex.core.kernel_interface.arg_pack_unpacker import Packer
    from numba_dpex.core.typing import typeof

    if _native_dpex_typeof_helper is None:
        _native_dpex_typeof_helper = typeof._typeof_helper

    if _native_Packer_unpack_usm_array is None:
        _native_Packer_unpack_usm_array = Packer._unpack_usm_array

    if not with_patches:
        typeof._typeof_helper = _native_dpex_typeof_helper
        Packer._unpack_usm_array = _native_Packer_unpack_usm_array
        return

    _SYCL_OBJECTS_IDS = WeakValueDictionary()

    _SYCL_OBJECTS_UNPACK_CACHE = WeakKeyDictionary()

    def _monkey_patch_unpack_usm_array(self, val):
        # We work around the fact that `val` is not hashable and can't get used as a
        # key of the `WeakKeyDictionary` by pairing it with an instance of an object
        # that is hashable and whose garbage collection is paired to `val`'s.
        id_val = id(val)
        object_id = _SYCL_OBJECTS_IDS.setdefault(id_val, _ObjectId(id=id(val)))
        _SYCL_OBJECTS_IDS[object_id] = val
        return _SYCL_OBJECTS_UNPACK_CACHE.setdefault(
            object_id, _native_Packer_unpack_usm_array(self, val)
        )

    _SYCL_OBJECTS_NUMBA_TYPE_CACHE = dict()

    def _monkey_patch_typeof_helper(val, array_class_type):
        array_class_type_cache = _SYCL_OBJECTS_NUMBA_TYPE_CACHE.setdefault(
            array_class_type, WeakKeyDictionary()
        )
        # NB: the same trick than for _monkey_patch_unpack_usm_array is used here
        id_val = id(val)
        object_id = _SYCL_OBJECTS_IDS.setdefault(id_val, _ObjectId(id=id(val)))
        _SYCL_OBJECTS_IDS[object_id] = val
        return array_class_type_cache.setdefault(
            object_id, _native_dpex_typeof_helper(val, array_class_type)
        )

    Packer._unpack_usm_array = _monkey_patch_unpack_usm_array
    typeof._typeof_helper = _monkey_patch_typeof_helper
