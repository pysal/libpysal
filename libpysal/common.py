import copy
import sys
import time

# external imports
import numpy as np
import numpy.linalg as la

import scipy as sp
import scipy.stats as stats
from .cg.kdtree import KDTree
from scipy.spatial.distance import pdist, cdist

import pandas

try:
    from patsy import PatsyError
except ImportError:
    PatsyError = Exception

RTOL = 0.00001
ATOL = 1e-7
MISSINGVALUE = None

####################
# Decorators/Utils #
####################

# import numba.jit OR create mimic decorator and set existence flag
try:
    from numba import jit

    HAS_JIT = True

except ImportError:

    def jit(function=None, **kwargs):
        """Mimic numba.jit() with synthetic wrapper."""

        if function is not None:

            def wrapped(*original_args, **original_kw):
                """Case 1 - structure of a standard decorator
                i.e., jit(function)(*args, **kwargs).
                """

                return function(*original_args, **original_kw)

            return wrapped

        else:

            def partial_inner(func):
                """Case 2 - returns Case 1
                i.e., jit()(function)(*args, **kwargs).
                """

                return jit(func)

            return partial_inner

    HAS_JIT = False


def simport(modname):
    """Safely import a module without raising an error.

    Parameters
    ----------
    modname : str
        Module name needed to import.
    
    Returns
    -------
    _simport : tuple
        Either (True, Module) or (False, None) depending
        on whether the import succeeded.

    Notes
    -----
    
    Wrapping this function around an iterative context or a with
    context would allow the module to be used without necessarily
    attaching it permanently in the global namespace:

        for t,mod in simport('pandas'):
            if t:
                mod.DataFrame()
            else:
                #do alternative behavior here
            del mod #or don't del, your call

    instead of:

        t, mod = simport('pandas')
        if t:
            mod.DataFrame()
        else:
            #do alternative behavior here

    The first idiom makes it work kind of a like a with statement.
    
    """

    try:
        exec("import {}".format(modname))
        _simport = True, eval(modname)
    except:
        _simport = False, None

    return _simport



def requires(*args, **kwargs):
    """Decorator to wrap functions with extra dependencies.

    Parameters
    ----------
    args : list
        Modules names as strings to import.
    verbose : bool
        Set as ``True`` to print a warning message on import failure.
    
    Returns
    -------
    inner : func
        The original function if all arg in args are importable.
    passer : func
        A function that passes if ``inner`` fails.
    
    """

    v = kwargs.pop("verbose", True)
    wanted = copy.deepcopy(args)

    def inner(function):
        available = [simport(arg)[0] for arg in args]
        if all(available):
            return function
        else:

            def passer(*args, **kwargs):
                if v:
                    missing = [arg for i, arg in enumerate(wanted) if not available[i]]
                    print(("missing dependencies: {d}".format(d=missing)))
                    print(("not running {}".format(function.__name__)))
                else:
                    pass

            return passer

    return inner
