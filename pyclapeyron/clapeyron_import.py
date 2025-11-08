import numpy as np
import juliacall
jl = juliacall.Main
from .utils import create_wrapper

jl.seval("using Clapeyron")

module_globals = globals()

# functions not to be wrapped
wrapper_exceptions = ['SRK','cPR','gcPCSAFT']

# main exports
cl_exports = [str(n) for n in jl.names(jl.Clapeyron) if not str(n).startswith('@')]

for cl_name in cl_exports:
    cl_obj = getattr(jl.Clapeyron, cl_name)
    if jl.isa(cl_obj, jl.Function) and not cl_obj.__name__ in wrapper_exceptions:
        python_wrapper = create_wrapper(cl_obj)
        module_globals[python_wrapper.__name__] = python_wrapper
    else:
        module_globals[cl_name] = cl_obj

# submodules
class _SubModule:
    """Container for Clapeyron submodule functions"""
    pass

for submod in ['VT','PT','PH','PS','QT','QP','TS']:
    # Create a new submodule instance
    submodule_obj = _SubModule()
    submodule_obj.__name__ = submod
    
    # Get Julia submodule
    jl_submod = getattr(jl.Clapeyron, submod)
    cl_names = jl.names(jl_submod, all=True)
    
    for _name in cl_names:
        name = str(_name)
        if not name.startswith('#'):
            cl_obj = getattr(jl_submod, name)
            python_wrapper = create_wrapper(cl_obj)
            setattr(submodule_obj, python_wrapper.__name__, python_wrapper)
    
    # Add the submodule to module globals
    module_globals[submod] = submodule_obj
