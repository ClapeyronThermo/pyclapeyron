# pyclapeyron

Python wrapper for [Clapeyron.jl](https://clapeyronthermo.github.io/Clapeyron.jl/).

## Installation

Install `pyclapeyron` from PyPI:

```
pip install pyclapeyron
```

## Documentation

The [Clapeyron.jl documentation](https://clapeyronthermo.github.io/Clapeyron.jl/) is also valid for `pyclapeyron`.
In comparison to Julia, the module name (e.g. `cl` for the import line `import pyclapyron as cl`) must be put up front.
Functions ending with `!` in Julia (often used for inplace operations) are renamed to end with `_b`, e.g. `fun!(...)` in Julia is `fun_b(...)` in Python.

## Examples

In the examples below, it is assumed that `pyclapeyron` is imported as `cl`, i.e. `import pyclapeyron as cl`.
They are taken from the [Notebook examples section](https://clapeyronthermo.github.io/Clapeyron.jl/stable/notebook_examples/) in the documentation.

### Ideal equations of state

```python
import pyclapeyron as cl
import matplotlib.pyplot as plt
import numpy as np

model1 = cl.SRK(["pentane"])
model2 = cl.SRK(["pentane"], idealmodel=cl.JobackIdeal)
model3 = cl.JobackIdeal(["pentane"])
model4 = cl.GERG2008(["pentane"])
model5 = cl.CKSAFT(["pentane"], idealmodel=cl.WalkerIdeal)
model6 = cl.SAFTgammaMie(["pentane"], idealmodel=cl.ReidIdeal)
models = [model1,model2,model3,model4,model5,model6]

p = 5e6
T = range(450,580,length=200)
Cp = zeros(200,6)

for i=1:6
    Cp[:,i] = isobaric_heat_capacity.(models[i],p,T)
end

T_exp = [450,455,460,465,470,475,480,485,490,495,500,505,510,515,520,525,530,535,540,545,550,555,560,565,570]

Cp_exp = [241.68,247.53,254.38,262.63,272.97,286.52,305.32,332.89,373.34,418.96,429.35,391.95,348.08,315.69,293.51,278.11,267.07,258.94,252.82,248.12,244.46,241.6,239.36,237.59,236.21];

plt.clf()
plt.plot(T,Cp[:,3],label="JobackIdeal",linestyle=":")
plt.plot(T,Cp[:,1],label="SRK",linestyle="--")
plt.plot(T,Cp[:,2],label="SRK{JobackIdeal}",linestyle="-.")
plt.plot(T,Cp[:,4],label="GERG-2008",linestyle=(0, (3, 1, 1, 1)))
plt.plot(T,Cp[:,5],label="CK-SAFT{WalkerIdeal}",linestyle=(0, (5, 1)))
plt.plot(T,Cp[:,6],label="SAFT-γ Mie{ReidIdeal}",linestyle=(0, (3, 1, 1, 1, 1, 1)))
plt.plot(T_exp,Cp_exp,label="experimental","o",color="k")
plt.legend(loc="upper right",frameon=false,fontsize=12) 
plt.xlabel("Temperature / K",fontsize=16)
plt.ylabel("Isobaric Heat Capacity / (J K⁻¹)",fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([450,580])
plt.ylim([50,600])
display(plt.gcf())
```