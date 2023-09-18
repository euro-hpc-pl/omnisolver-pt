import os

from numba import typeof


def numba_type_of_instance(t):
    return t if os.getenv("NUMBA_DISABLE_JIT", "0") == "1" else typeof(t).class_type.instance_type


def numba_type_of_cls(t):
    return t if os.getenv("NUMBA_DISABLE_JIT", "0") == "1" else t.class_type.instance_type
