from juliacall import Main as jl

jl.seval("using Clapeyron")
cl_exports = [str(n) for n in jl.names(jl.Clapeyron) if not str(n).startswith('@')]
cl_exports = [s.replace('!','_b') for s in cl_exports]
for s in cl_exports:
    exec(f'{s} = jl.{s}')

__all__ = [s for s in cl_exports]