import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np
from juliacall import Main as jl

# @testset "PCSAFT Models" begin
def test_PCSAFT():
    T = 298.15
    V = 1e-4
    z = np.array([0.5, 0.5])
    
    system = cl.PCSAFT(["butane", "ethanol"])
    assert cl.Clapeyron.a_hc(system, V, T, z) == approx(3.1148229872928654, rel=1e-6)
    assert cl.Clapeyron.a_disp(system, V, T, z) == approx(-6.090736508783152, rel=1e-6)
    assert cl.Clapeyron.a_assoc(system, V, T, z) == approx(-1.6216064387201956, rel=1e-6)

def test_PCPSAFT():
    T = 298.15
    V = 1e-4
    z3 = np.array([0.333, 0.333, 0.333])
    
    system = cl.PCPSAFT(["acetone", "butane", "DMSO"])
    cl.Clapeyron.set_k_b(system, np.zeros((3, 3)))
    cl.Clapeyron.set_l_b(system, np.zeros((3, 3)))
    
    assert cl.Clapeyron.a_polar(system, V, T, z3) == approx(-0.6541688650413224, rel=1e-6)
    
    system2 = cl.PCPSAFT(["acetone", "water", "DMSO"])
    assert cl.Clapeyron.a_polar(system2, V, T, z3) == approx(-0.6318599798899884, rel=1e-6)

def test_QPCPSAFT():
    T = 298.15
    V = 1e-4
    z3 = np.array([0.333, 0.333, 0.333])
    
    system1 = cl.QPCPSAFT(["carbon dioxide", "acetone", "hydrogen sulfide"])
    system2 = cl.QPCPSAFT(["carbon dioxide", "chlorine", "carbon disulfide"])
    assert cl.Clapeyron.a_mp(system1, V, T, z3) == approx(-0.37364363283985724, rel=1e-6)
    assert cl.Clapeyron.a_mp(system2, V, T, z3) == approx(-0.1392358363758833, rel=1e-6)

def test_gcPCPSAFT():
    T = 298.15
    V = 1e-4
    z3 = np.array([0.333, 0.333, 0.333])
    
    system = cl.gcPCPSAFT(["acetone", "ethane", "ethanol"], mixing=jl.Symbol("homo"))
    system2 = cl.gcPCPSAFT(["acetone", "ethane", "ethanol"], mixing=jl.Symbol("hetero"))    

def test_sPCSAFT():
    T = 298.15
    V = 1e-4
    z = np.array([0.5, 0.5])
    
    system = cl.sPCSAFT(["propane", "methanol"])
    assert cl.Clapeyron.a_hc(system, V, T, z) == approx(2.024250583187793, rel=1e-6)
    assert cl.Clapeyron.a_disp(system, V, T, z) == approx(-4.138653131750594, rel=1e-6)
    assert cl.Clapeyron.a_assoc(system, V, T, z) == approx(-1.1459701721909195, rel=1e-6)
    system2 = cl.sPCSAFT(["water", "ethanol"])
    assert cl.Clapeyron.a_assoc(system2, V, T, z) == approx(-3.2282881877257728, rel=1e-6)

def test_gcsPCSAFT():
    T = 298.15
    V = 1e-4
    z = np.array([0.5, 0.5])
    
    system = cl.gcsPCSAFT(["acetone", "ethane"])

def test_gcPCSAFT():
    T = 298.15
    V = 1e-4
    z = np.array([0.5, 0.5])
    z3 = np.array([0.333, 0.333, 0.333])
    
    species = [("ethanol", {"CH3": 1, "CH2": 1, "OH": 1}, {("CH3", "CH2"): 1, ("OH", "CH2"): 1}),
               ("hexane", {"CH3": 2, "CH2": 4}, {("CH3", "CH2"): 2, ("CH2", "CH2"): 3})]
    
    system = cl.gcPCSAFT(species)
    assert cl.Clapeyron.a_hc(system, V, T, z) == approx(5.485662509904188, rel=1e-6)
    assert cl.Clapeyron.a_disp(system, V, T, z) == approx(-10.594659479487497, rel=1e-6)
    assert cl.Clapeyron.a_assoc(system, V, T, z) == approx(-0.9528180944200482, rel=1e-6)
    
    system2 = cl.gcPCPSAFT(["acetone", "ethane", "ethanol"], mixing=jl.Symbol("hetero"))
    assert cl.Clapeyron.a_assoc(system2, V, T, z3) == approx(-0.46072184884780865, rel=1e-6)
    

# @testset "PCSAFT Models - others" begin
def test_CP_PCSAFT():
    T = 298.15
    V = 1e-4
    z1 = np.array([1.0])
    z = np.array([0.5, 0.5])
    
    system = cl.CPPCSAFT(["butane", "ethanol"])
    assert cl.Clapeyron.a_hc(system, V, T, z) == approx(3.5147121772115604, rel=1e-6)
    assert cl.Clapeyron.a_disp(system, V, T, z) == approx(-6.096718490450866, rel=1e-6)
    assert cl.Clapeyron.a_assoc(system, V, T, z) == approx(-1.86685186469097, rel=1e-6)
    
    system2 = cl.CPPCSAFT("water")
    assert cl.Clapeyron.a_assoc(system2, V, T, np.array([1.0])) == approx(-5.928187577713146, rel=1e-6)
    
    system3 = cl.CPPCSAFT(["butane", "propane"])
    assert cl.Clapeyron.a_hc(system3, V, T, z) == approx(3.856483933013827, rel=1e-6)
    assert cl.Clapeyron.a_disp(system3, V, T, z) == approx(-6.613302753897683, rel=1e-6)
    
    system4 = cl.CPPCSAFT(["water", "ethanol"])
    assert cl.Clapeyron.a_assoc(system4, V, T, z) == approx(-4.228106574319739, rel=1e-6)

def test_GEPCSAFT():
    T = 298.15
    V = 1e-4
    z = np.array([0.5, 0.5])
    
    system = cl.GEPCSAFT(["propane", "methanol"])
    assert cl.Clapeyron.a_hc(system, V, T, z) == approx(1.6473483928460233, rel=1e-6)
    assert cl.Clapeyron.a_disp(system, V, T, z) == approx(-3.271039575934372, rel=1e-6)
    assert cl.Clapeyron.a_assoc(system, V, T, z) == approx(-1.9511233680313027, rel=1e-6)

def test_ADPCSAFT():
    T = 298.15
    V = 1e-4
    z1 = np.array([1.0])
    
    system = cl.ADPCSAFT(["water"])
    assert cl.Clapeyron.a_hs(system, V, T, z1) == approx(0.3130578789492178, rel=1e-6)
    assert cl.Clapeyron.a_disp(system, V, T, z1) == approx(-1.2530666693292463, rel=1e-6)
    assert cl.Clapeyron.a_assoc(system, V, T, z1) == approx(-3.805796041192079, rel=1e-6)

def test_DAPT():
    T = 298.15
    V = 1e-4
    z1 = np.array([1.0])
    
    system = cl.DAPT(["water"])
    assert cl.Clapeyron.a_hs(system, V, T, z1) == approx(0.35240995905438116, rel=1e-6)
    assert cl.Clapeyron.a_disp(system, V, T, z1) == approx(-1.7007754776344663, rel=1e-6)
    assert cl.Clapeyron.a_assoc(system, V, T, z1) == approx(-1.815041612389342, rel=1e-6)
