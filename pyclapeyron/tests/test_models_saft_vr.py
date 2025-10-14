import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np
import gc

# @testset "SAFT-VR-Mie Models" begin
def test_SAFTVRMie():
    T = 298.15
    V = 1e-4
    z1 = np.array([1.0])
    z = [0.5, 0.5]
    
    system = cl.SAFTVRMie(["methanol", "water"])
    assert cl.a_mono(system, V, T, z) == approx(-0.9729139704318698, rel=1e-6)
    _a_chain = cl.a_chain(system, V, T, z)
    _a_disp = cl.a_disp(system, V, T, z)
    assert _a_chain == approx(-0.028347378889242814, rel=1e-6)
    assert cl.a_dispchain(system, V, T, z) - _a_chain == approx(_a_disp, rel=1e-6)
    assert cl.a_assoc(system, V, T, z) == approx(-4.18080707238976, rel=1e-6)
    cl.test_gibbs_duhem(system, V, T, z)
    cl.test_recombine(system)
    gc.collect()

def test_SAFTVRMieGV():
    T = 298.15
    z = [0.5, 0.5]
    
    system = cl.SAFTVRMieGV(["benzene", "acetone"])
    V_GV = 8e-5
    cl.test_gibbs_duhem(system, V_GV, T, z)
    cl.test_recombine(system)
    
    assert cl.a_mp(system, V_GV, T, z) == approx(-0.7521858819355216, rel=1e-6)
    
    system2 = cl.SAFTVRMieGV(["carbon dioxide", "benzene"])
    assert cl.a_mp(system2, V_GV, T, z) == approx(-0.33476290652200424, rel=1e-6)
    
    system3 = cl.SAFTVRMieGV(["acetone", "diethyl ether", "ethyl acetate"])
    assert cl.a_mp(system3, V_GV, T, [0.3, 0.3, 0.4]) == approx(-0.8088095725122674, rel=1e-6)
    gc.collect()

def test_SAFTVRQMie():
    T = 298.15
    V = 1e-4
    z1 = np.array([1.0])
    
    system = cl.SAFTVRQMie(["helium"])
    assert cl.a_mono(system, V, T, z1) == approx(0.12286776703976324, rel=1e-6)
    cl.test_gibbs_duhem(system, V, T, z1)
    gc.collect()

def test_SAFTVRSMie():
    T = 298.15
    z1 = np.array([1.0])
    
    system = cl.SAFTVRSMie(["carbon dioxide"])
    V_sol = 3e-5
    assert cl.a_mono(system, V_sol, T, z1) == approx(0.43643302846919896, rel=1e-6)
    assert cl.a_chain(system, V_sol, T, z1) == approx(-0.4261294644079463, rel=1e-6)
    cl.test_gibbs_duhem(system, V_sol, T, z1, rtol=1e-12)
    gc.collect()

def test_SAFTVRMie15():
    z1 = np.array([1.0])
    v15 = 1/(1000*1000*1.011/18.015)
    T15 = 290.0
    vr15 = cl.SAFTVRMie15("water")
    # Dufal, table 4, 290K, f_OH(free) = 0.089
    assert cl.X(vr15, v15, T15, z1)[0][0] == approx(0.08922902098124778, rel=1e-6)

def test_SAFTgammaMie():
    T = 298.15
    V = 1e-4
    z = [0.5, 0.5]
    
    system = cl.SAFTgammaMie(["methanol", "butane"])
    V_gammaMie = 10**(-3.5)
    assert cl.a_mono(system, V_gammaMie, T, z) == approx(-1.0400249396482548, rel=1e-6)
    assert cl.a_chain(system, V_gammaMie, T, z) == approx(-0.07550931466871749, rel=1e-6)
    assert cl.a_assoc(system, V_gammaMie, T, z) == approx(-0.8205840455850311, rel=1e-6)
    cl.test_gibbs_duhem(system, V, T, z)
    cl.test_scales(system)
    cl.test_recombine(system)
    cl.test_repr(system, str=["\"methanol\": \"CH3OH\" => 1"])
    gc.collect()

def test_structSAFTgammaMie():
    T = 298.15
    z = [0.5, 0.5]
    
    species = [("ethanol", {"CH3": 1, "CH2OH": 1}, {("CH3", "CH2OH"): 1}),
               ("octane", {"CH3": 2, "CH2": 6}, {("CH3", "CH2"): 2, ("CH2", "CH2"): 5})]
    
    system = cl.structSAFTgammaMie(species)
    V_gammaMie = 10**(-3.5)
    assert cl.a_chain(system, V_gammaMie, T, z) == approx(-0.11160851237651681, rel=1e-6)
    cl.test_gibbs_duhem(system, V_gammaMie, T, z, rtol=1e-12)
    gc.collect()
