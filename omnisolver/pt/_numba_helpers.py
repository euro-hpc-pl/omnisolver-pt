"""Some helpers easing the pain of interacting with numba."""
import os

from numba import typeof


def numba_type_of_instance(t):
    """Get numba type of a jitted class instance.

    This works both in normal and debug mode.
    """
    return t if os.getenv("NUMBA_DISABLE_JIT", "0") == "1" else typeof(t).class_type.instance_type


def numba_type_of_cls(t):
    """Get numba type of a jitted class.

    This works both in normal and debug mode.
    """
    return t if os.getenv("NUMBA_DISABLE_JIT", "0") == "1" else t.class_type.instance_type
