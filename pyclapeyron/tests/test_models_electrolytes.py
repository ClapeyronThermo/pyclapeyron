import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np

# @testset "Electrolyte models" begin
def test_ConstRSP():
    T = 298.15
    V = 1e-3
    w = [0.99, 0.005, 0.005]
    Z = [0, 1, -1]
    
    system = cl.ConstRSP()
    assert cl.dielectric_constant(system, V, T, w, Z) == approx(78.38484961, rel=1e-6)

def test_ZuoFurst():
    T = 298.15
    V = 1e-3
    w = [0.99, 0.005, 0.005]
    Z = [0, 1, -1]
    
    system = cl.ZuoFurst()
    assert cl.dielectric_constant(system, V, T, w, Z) == approx(78.30270731614397, rel=1e-6)

def test_LinMixRSP():
    T = 298.15
    V = 1e-3
    w = [0.99, 0.005, 0.005]
    Z = [0, 1, -1]
    
    system = cl.LinMixRSP(["water"], ["sodium", "chloride"])
    assert cl.dielectric_constant(system, V, T, w, Z) == approx(77.68100111390001, rel=1e-6)

def test_Schreckenberg():
    T = 298.15
    w = [0.99, 0.005, 0.005]
    Z = [0, 1, -1]
    
    system = cl.Schreckenberg(["water"], ["sodium", "chloride"])
    assert cl.dielectric_constant(system, 1.8e-5, T, w, Z) == approx(77.9800485493879, rel=1e-6)

def test_ePCSAFT():
    T = 298.15
    V = 1e-3
    m = 1
    
    system = cl.ePCSAFT(["water"], ["sodium", "chloride"])
    cl.test_repr(system, str=["Neutral Model:", "Ion Model:", "RSP Model"])
    salts = [("sodium chloride", [("sodium", 1), ("chloride", 1)])]
    z = cl.molality_to_composition(system, salts, m)
    # this result is using normal water parameters. From Clapeyron 0.6.9 onwards, "water" will use T-dependent sigma on pharmaPCSAFT.
    # assert cl.a_res(system, V, T, z) == approx(-1.1094732499161208, rel=1e-6)
    assert cl.a_res(system, V, T, z) == approx(-1.0037641675166717, rel=1e-6)

def test_eSAFTVRMie():
    T = 298.15
    V = 1e-3
    m = 1
    
    system = cl.eSAFTVRMie(["water"], ["sodium", "chloride"])
    salts = [("sodium chloride", [("sodium", 1), ("chloride", 1)])]
    z = cl.molality_to_composition(system, salts, m)
    assert cl.a_res(system, V, T, z) == approx(-6.8066182496746634, rel=1e-6)

def test_SAFTVREMie():
    T = 298.15
    V = 1e-3
    m = 1
    
    system = cl.SAFTVREMie(["water"], ["sodium", "chloride"])
    salts = [("sodium chloride", [("sodium", 1), ("chloride", 1)])]
    z = cl.molality_to_composition(system, salts, m)
    assert cl.a_res(system, V, T, z) == approx(-5.37837866742139, rel=1e-6)

def test_SAFTVREMie_MSA():
    T = 298.15
    V = 1e-3
    m = 1
    
    system = cl.SAFTVREMie(["water"], ["sodium", "chloride"], ionmodel=cl.MSA)
    salts = [("sodium chloride", [("sodium", 1), ("chloride", 1)])]
    z = cl.molality_to_composition(system, salts, m)
    assert cl.a_res(system, V, T, z) == approx(-2.266066359288185, rel=1e-6)

def test_SAFTVREMie_Born():
    T = 298.15
    V = 1e-3
    m = 1
    
    system = cl.SAFTVREMie(["water"], ["sodium", "chloride"], ionmodel=cl.Born)
    salts = [("sodium chloride", [("sodium", 1), ("chloride", 1)])]
    z = cl.molality_to_composition(system, salts, m)
    assert cl.a_res(system, V, T, z) == approx(-4.907123437550198, rel=1e-6)

def test_SAFTVREMie_MSAID():
    ionmodel = cl.MSAID(["water"], ["sodium", "chloride"],
                        userlocations=dict(
                            charge=[0, 1, -1],
                            sigma=[4., 4., 4.],
                            dipole=[2.2203, 0., 0.]))
    
    system = cl.SAFTVREMie(["water"], ["sodium", "chloride"], ionmodel=ionmodel)
    iondata = cl.iondata(system, 1.8e-5, 298.15, [0.9, 0.05, 0.05])
    assert cl.a_res(ionmodel, 1.8e-5, 298.15, [0.9, 0.05, 0.05], iondata) == approx(-224.13923847423726, rel=1e-6)

def test_SAFTgammaEMie():
    T = 298.15
    V = 1e-3
    m = 1
    
    system = cl.SAFTgammaEMie(["water"], ["sodium", "acetate"])
    cl.test_scales(system)
    cl.test_recombine(system)
    salts = [("sodium acetate", [("sodium", 1), ("acetate", 1)])]
    z = cl.molality_to_composition(system, salts, m)
    # assert cl.a_res(system, V, T, z) == approx(-5.090195753415607, rel=1e-6)
    assert cl.a_res(system, V, T, z) == approx(-5.401187603151029, rel=1e-6)
    
    model = cl.SAFTgammaEMie(["water"], ["calcium", "chloride"])  # issue 349
    assert isinstance(model, cl.EoSModel)
