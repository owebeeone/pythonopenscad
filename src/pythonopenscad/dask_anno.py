from dask.delayed import Delayed, delayed as dask_delayed
from typing import TypeVar, Callable, ParamSpec, Generic

P = ParamSpec('P')  # For parameters
R = TypeVar('R')    # For return type

class TypedDelayed(Delayed, Generic[R]):
    """A type wrapper for Dask Delayed that preserves type annotation.
    This class cannot be instantiated - it's only used for type hints.
    """
    
    def __init__(self) -> None:
        """This class cannot be instantiated"""
        raise NotImplementedError("TypedDelayed is a type hint and cannot be instantiated")
    
    def compute(self, **kwargs) -> R:
        """Type hint for compute() that preserves return type"""
        raise NotImplementedError("Only used for type annotations")


def delayed(func: Callable[P, R], name=None, pure=None, nout=None, traverse=True, **kwargs) \
    -> Callable[P, TypedDelayed[R]]:
    """Decorator that returns a type-aware delayed function, replaces dask.delayed."""
    return dask_delayed(func, name=name, pure=pure, nout=nout, traverse=traverse, **kwargs)

