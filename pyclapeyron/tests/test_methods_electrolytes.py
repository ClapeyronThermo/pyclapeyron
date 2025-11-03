import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np
from juliacall import Main as jl

# @testset "Electrolyte methods" begin
def test_Bulk_properties():
    system = cl.SAFTVREMie(["water"], ["sodium", "chloride"])
    salts = (("sodium chloride", [("sodium", 1), ("chloride", 1)]),)
    p = 1e5
    T = 298.15
    m = 1.
    z = cl.molality_to_composition(system, salts, m)
    
    assert cl.mean_ionic_activity_coefficient(system, salts, p, T, m)[0] == approx(0.6321355339999393, rel=1e-6)
    assert cl.osmotic_coefficient(system, salts, p, T, m)[0] == approx(0.9301038212951828, rel=1e-6)
    
    assert cl.mean_ionic_activity_coefficient_sat(system, salts, T, m)[0] == approx(0.632123986095978, rel=1e-6)
    assert cl.osmotic_coefficient_sat(system, salts, T, m)[0] == approx(0.9300995995153606, rel=1e-5)

def test_Solid_solubility():
    fluid = cl.SAFTVREMie(["water"], ["sodium", "chloride"])
    solid = cl.SolidKs(["water", "sodium.chloride", "sodium.chloride.2water"])
    mapping = {
        (("water", 1),):   ("water", 1),
        (("sodium", 1), ("chloride", 1)): ("sodium.chloride", 1),
        (("sodium", 1), ("chloride", 1), ("water", 2)): ("sodium.chloride.2water", 1)
    }
    
    system = cl.CompositeModel(["water", "sodium", "chloride"], mapping=mapping, fluid=fluid, solid=solid)
    p = 1e5
    
    assert cl.sle_solubility(system, p, 270., np.array([1, 1, 1]), solute=["water"])[0] == approx(0.9683019351679348, rel=1e-6)
    assert cl.sle_solubility(system, p, 300., np.array([1, 1, 1]), solute=["sodium.chloride"], x0=jl.Array[jl.Float64]([-1.2]))[0] == approx(0.7570505178871523, rel=1e-6)

def test_Tp_flash():
    system = cl.ePCSAFT(["water", "acetonitrile"], ["sodium", "chloride"])
    p = 1e5
    T = 298.15
    
    x, n, G = cl.tp_flash(system, p, T, np.array([0.4, 0.4, 0.1, 0.1]), 
                          cl.MichelsenTPFlash(equilibrium=jl.Symbol("lle"), K0=np.array([100., 1e-3, 1000., 1000.])))
    # assert x[0, 0] == approx(0.07303405420993273, rel = 1e-6)   # broken
