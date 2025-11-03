import juliacall
jl = juliacall.Main
import numpy as np

def create_wrapper(julia_func):

    def wrapper(*args, **kwargs):
        # Check for Python lists in args
        for i, arg in enumerate(args):
            if isinstance(arg, list) and not all(isinstance(x, str) for x in arg):
                raise TypeError(f"Argument {i+1} ({arg}): arrays must be passed as numpy arrays")
        
        # Check for Python lists in kwargs
        for key, value in kwargs.items():
            if isinstance(value, list) and not all(isinstance(x, str) for x in value):
                raise TypeError(f"Keyword argument '{key}' ({arg}): arrays must be passed as numpy arrays")
        
        result_from_julia = julia_func(*args, **kwargs)

        if isinstance(result_from_julia, tuple):
            processed_results = []
            for item in result_from_julia:
                if isinstance(item, juliacall.ArrayValue):
                    processed_results.append(np.asarray(item))
                else:
                    processed_results.append(item)
            return tuple(processed_results)
        elif isinstance(result_from_julia, juliacall.ArrayValue):
            return np.asarray(result_from_julia)
        else:
            return result_from_julia

    # Set name and docs
    wrapper.__name__ = julia_func.__name__.replace('!','_b')
    wrapper.__doc__ = f"Python wrapper for Julia's `{wrapper.__name__}` function."
    
    return wrapper

# Helper functions to "concretize" a dict on the Julia side
#TODO better way? -> see database.jl:718 `... elseif v isa Dict{Tuple{Tuple{String,String},Tuple{String,String}}} ...`
jl.seval("concretize(d::AbstractDict) = Dict(k => v for (k,v) in d)")
jl.seval("concretize(d) = d")
