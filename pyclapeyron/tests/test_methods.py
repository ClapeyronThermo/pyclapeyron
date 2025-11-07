import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np
from juliacall import Main as jl
import os

def test_bulk_properties():
    models = [
        cl.PCSAFT(["ethanol", "toluene"]),
        cl.vdW(["ethanol", "toluene"]),
        cl.UNIFAC(["ethanol", "toluene"])
    ]

    for model in models:
        cl.volume(model, 1e5, 300., np.array([.5,.5]))