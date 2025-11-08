from .clapeyron_import import *
from .clapeyron_import import jl

jl.seval("import Clapeyron")
for mod in ['vt','pt','ph','ps','qt','qp','ts']:
    exec(f'{mod} = jl.seval("Clapeyron.{mod.upper()}")')