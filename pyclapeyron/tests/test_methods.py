import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np
import os

def test_bulk_properties():
    models = [
        cl.PCSAFT(["ethanol", "toluene"]),
        cl.vdW(["ethanol", "toluene"]),
    ]

    for model in models:
        v = cl.volume(model, 1e5, 300., np.array([.5,.5]))
        p = cl.pressure(model, v, 300., np.array([.5,.5]))
        assert 1e5 == approx(p)

def test_pure_properties():
    models = [
        cl.PCSAFT(["ethanol"]),
        cl.vdW(["ethanol"]),
    ]

    for model in models:
        Tc, pc, vc = cl.crit_pure(model)
        ps, vl, vv = cl.saturation_pressure(model, Tc*0.6)
        Ts, vl, vv = cl.saturation_temperature(model, ps)
        assert Tc*0.6 == approx(Ts)

def test_mix_properties():
    models = [
        cl.PCSAFT(["ethanol", "toluene"]),
        cl.vdW(["ethanol", "toluene"]),
        cl.UNIFAC(["ethanol", "toluene"], puremodel=cl.AntoineEqSat)
    ]

    for model in models:
        ps1, _, _, y = cl.bubble_pressure(model, 300., np.array([.5,.5]))
        ps2, _, _, x = cl.dew_pressure(model, 300., y)
        assert approx(ps1) == approx(ps2)
        assert [.5,.5] == approx(x, rel=1e-3)
