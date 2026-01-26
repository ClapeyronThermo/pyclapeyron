# pyclapeyron

Python wrapper for [Clapeyron.jl](https://clapeyronthermo.github.io/Clapeyron.jl/).

## Installation

Install `pyclapeyron` from PyPI via pip or uv:

```
pip install pyclapeyron
```
or
```
uv add pyclapeyron
```

## Documentation

See the [Clapeyron.jl documentation](https://clapeyronthermo.github.io/Clapeyron.jl/stable/).

In general, `pyclapeyron` only acts as a wrapper around the Julia functions using the same syntax.
In the following, notable deviations from the Julia syntax are highlighted.

- **Julia functions with `!`:** Functions ending with `!` in Julia (often used for inplace operations) are renamed to end with `_b`, e.g. `fun!(...)` in Julia is `fun_b(...)` in Python.
- **Array type:** All numeric arrays passed to `Clapeyron` functions need to be `numpy` arrays, e.g.

```python
import pyclapeyron as cl
import numpy as np

model = cl.vdW(["nitrogen", "argon"], userlocations={
    "Tc": np.array([127., 150.]),
    "Pc": np.array([34e5., 49e5]),
    "Mw": np.array([28.01, 39.95])
})

cl.molar_density(model, 1e5, 300., np.array([0.5,0.5]))
```
- **Syntax for group-contribution (GC) models:** The components for group contribution models that include the groups need to be passed in the form `("component name", {"group1": n1, ...}, ...)`. The dict replaces Julias' `Vector{Pair}` as there is no direct equivalent in Python, e.g.
```python
import pyclapeyron as cl

model = cl.CompositeModel(
    ["water", "ethanol", ("ibuprofen", {"ACH": 4, "ACCH2": 1, "ACCH": 1, "CH3": 3, "COOH": 1, "CH": 1})],
    liquid=cl.UNIFAC, solid=cl.SolidHfus
)
```
- **Electrolyte models:** Similar to GC models, salts in electrolyte models need to be defined as `salts = (("sodium chloride", [("sodium", 1), ("chloride", 1)]),)`.
- **Functions as function arguments:** If you need to pass a `Clapeyron` function to another function, e.g. as in `partial_property`, you need to pass the original `Clapeyron.some_func` function, e.g. ```cl.partial_property(model, p, T, z, cl.Clapeyron.volume)```.

## Examples

- pure substance VLE

```python
import pyclapeyron as cl
import numpy as np
import matplotlib.pyplot as plt

model = cl.PCSAFT("benzene")
Tc, pc, vc = cl.crit_pure(model)

Tx = np.linspace(0.5*Tc, Tc, 100)
ps, vl, vv = np.zeros(100), np.zeros(100), np.zeros(100)
for (i,Ti) in enumerate(Tx):
    ps[i], vl[i], vv[i] = cl.saturation_pressure(model, Ti)

fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6,3), tight_layout=True)
axs[0].plot(1e-3/np.concat([vl,np.flip(vv)]), np.concat([ps,np.flip(ps)])/1e5, c="blue")
axs[1].plot(Tx, ps/1e5, c="blue")
axs[0].scatter(1e-3/vc, pc/1e5, marker="*", c="blue")
axs[1].scatter(Tc, pc/1e5, marker="*", c="blue")
axs[0].set(yscale="log", xlabel="ϱ / mol l⁻¹", ylabel="p / bar")
axs[1].set(xlabel="T / K")
```
<p align="center">
    <img src="docs/pure_vle.svg" width="900">
</p>

- mixture VLE

```python
import pyclapeyron as cl
import numpy as np
import matplotlib.pyplot as plt

components = ["methanol", "r123"]
components_with_groups = [("methanol", {"CH3OH": 1}), ("r123", {"CF3": 1, "CHCL2": 1})]
model = cl.UNIFAC2(components_with_groups, puremodel=cl.AntoineEqSat(components))

T_iso = 300.
x1 = np.linspace(0.,1.,100)
ps, y1 = np.zeros(100), np.zeros(100)
for (i,x1i) in enumerate(x1):
    ps[i], _, _, yi = cl.dew_pressure(model, T_iso, np.array([x1i,1-x1i]))
    y1[i] = yi[0]

fig, ax = plt.subplots(figsize=(4,3), tight_layout=True)
ax.plot(x1, ps/1e5, c="red")
ax.plot(y1, ps/1e5, c="red")
ax.set(xlim=(0,1), xlabel="x₁ / mol l⁻¹", ylabel="p / bar")
```
<p align="center">
    <img src="docs/mix_vle.svg" width="600">
</p>

## For Developers 

- Use `uv add --dev path/to/local/pyclapeyron` to add `pyclapeyron` to a local env. 
- You can optionally use a local `dev` version of `Clapeyron.jl` by replacing the line
```json
"Clapeyron": {
    "uuid": "7c7805af-46cc-48c9-995b-ed0ed2dc909a",
    "version": "X.X.X"
}
```
by 
```json
"Clapeyron": {
    "uuid": "7c7805af-46cc-48c9-995b-ed0ed2dc909a",
    "dev": true,
    "path": "/path/to/.julia/dev/Clapeyron"
},
```
in the `pyclapeyron/juliapkg.json` file.
