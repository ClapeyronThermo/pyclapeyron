import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np
import gc

# @testset "SAFT models - misc" begin
def test_ogSAFT():
    T = 298.15
    V = 1e-4
    z = [0.5, 0.5]
    
    system = cl.ogSAFT(["water", "ethylene glycol"])
    assert cl.a_seg(system, V, T, z) == approx(-2.0332062924093366, rel=1e-6)
    assert cl.a_chain(system, V, T, z) == approx(-0.006317441684202759, rel=1e-6)
    assert cl.a_assoc(system, V, T, z) == approx(-4.034042081699316, rel=1e-6)
    cl.test_gibbs_duhem(system, V, T, z)
    gc.collect()

def test_CKSAFT():
    T = 298.15
    V = 1e-4
    z = [0.5, 0.5]
    
    system = cl.CKSAFT(["carbon dioxide", "2-propanol"])
    assert cl.a_seg(system, V, T, z) == approx(-1.24586302917188, rel=1e-6)
    assert cl.a_chain(system, V, T, z) == approx(-0.774758615408493, rel=1e-6)
    assert cl.a_assoc(system, V, T, z) == approx(-1.2937079004096872, rel=1e-6)
    cl.test_gibbs_duhem(system, V, T, z)
    gc.collect()

def test_sCKSAFT():
    T = 298.15
    V = 1e-4
    z = [0.5, 0.5]
    
    system = cl.sCKSAFT(["benzene", "acetic acid"])
    assert cl.a_seg(system, V, T, z) == approx(-3.1809330810925256, rel=1e-6)
    assert cl.a_assoc(system, V, T, z) == approx(-3.3017434376105514, rel=1e-6)
    cl.test_gibbs_duhem(system, V, T, z)
    gc.collect()

def test_BACKSAFT():
    T = 298.15
    V = 1e-4
    z1 = np.array([1.0])
    
    system = cl.BACKSAFT(["carbon dioxide"])
    assert cl.a_hcb(system, V, T, z1) == approx(1.0118842111801198, rel=1e-6)
    assert cl.a_chain(system, V, T, z1) == approx(-0.14177009317268635, rel=1e-6)
    assert cl.a_disp(system, V, T, z1) == approx(-2.4492518566426296, rel=1e-6)
    cl.test_gibbs_duhem(system, V, T, z1)
    gc.collect()

def test_LJSAFT():
    T = 298.15
    V = 1e-4
    z = [0.5, 0.5]
    
    system = cl.LJSAFT(["ethane", "1-propanol"])
    assert cl.a_seg(system, V, T, z) == approx(-2.207632433058473, rel=1e-6)
    assert cl.a_chain(system, V, T, z) == approx(-0.04577483379871112, rel=1e-6)
    assert cl.a_assoc(system, V, T, z) == approx(-1.3009761155167205, rel=1e-6)
    cl.test_gibbs_duhem(system, V, T, z)
    gc.collect()

def test_SAFTVRSW():
    T = 298.15
    V = 1e-4
    z = [0.5, 0.5]
    
    system = cl.SAFTVRSW(["water", "ethane"])
    assert cl.a_mono(system, V, T, z) == approx(-1.4367205951569462, rel=1e-6)
    assert cl.a_chain(system, V, T, z) == approx(0.022703335564111336, rel=1e-6)
    assert cl.a_assoc(system, V, T, z) == approx(-0.5091186813915323, rel=1e-6)
    cl.test_gibbs_duhem(system, V, T, z)
    gc.collect()

def test_softSAFT():
    T = 298.15
    V = 1e-4
    z = [0.5, 0.5]
    
    system = cl.softSAFT(["hexane", "1-propanol"])
    assert cl.a_LJ(system, V, T, z) == approx(-3.960728242264164, rel=1e-6)
    assert cl.a_chain(system, V, T, z) == approx(0.3736728407455211, rel=1e-6)
    assert cl.a_assoc(system, V, T, z) == approx(-2.0461376618069034, rel=1e-6)
    cl.test_recombine(system)
    # TODO: check here why the error is so big
    # we disable this test for now, it takes too much time
    # cl.test_gibbs_duhem(system, V, T, z, rtol=1e-12)
    gc.collect()

def test_softSAFT2016():
    T = 298.15
    V = 1e-4
    z = [0.5, 0.5]
    
    system = cl.softSAFT2016(["hexane", "1-propanol"])
    assert cl.a_LJ(system, V, T, z) == approx(-3.986690073534575, rel=1e-6)
    cl.test_gibbs_duhem(system, V, T, z)
    gc.collect()

def test_solidsoftSAFT():
    T = 298.15
    z1 = np.array([1.0])
    
    system = cl.solidsoftSAFT(["octane"])
    V_sol = 1e-4
    assert cl.a_LJ(system, V_sol, T, z1) == approx(7.830498923903852, rel=1e-6)
    assert cl.a_chain(system, V_sol, T, z1) == approx(-2.3460460361188207, rel=1e-6)
    cl.test_gibbs_duhem(system, V_sol, T, z1, rtol=1e-12)
    gc.collect()

# @testset "CPA" begin
def test_CPA():
    T = 298.15
    V = 1e-4
    z = [0.5, 0.5]
    
    system = cl.CPA(["ethanol", "benzene"])
    assert cl.a_assoc(system, V, T, z) == approx(-1.1575210505284332, rel=1e-6)
    cl.test_gibbs_duhem(system, V, T, z)
    cl.test_scales(system)
    cl.test_repr(system, str=["RDF: Carnahan-Starling (original CPA)"])
    gc.collect()

def test_sCPA():
    T = 298.15
    V = 1e-4
    z = [0.5, 0.5]
    
    system = cl.sCPA(["water", "carbon dioxide"])
    cl.test_repr(system, str=["RDF: Kontogeorgis (s-CPA)"])
    assert cl.a_assoc(system, V, T, z) == approx(-1.957518287413705, rel=1e-6)
    gc.collect()

# @testset "COFFEE" begin
def test_COFFEE():
    T = 298.15
    V = 1e-4
    z = np.array([1.0])
    
    system = cl.COFFEE(["a1"], userlocations=dict(
        Mw=[1.],
        segment=[1.],
        sigma=[[3.]],
        epsilon=[[300.]],
        lambda_r=[[12]],
        lambda_a=[[6]],
        shift=[3*0.15],
        dipole=[2.0*1.0575091914494172],))
    assert cl.a_ff(system, V, T, z) == approx(-2.3889723633885107, rel=1e-6)
    assert cl.a_nf(system, V, T, z) == approx(-0.7418363729609996, rel=1e-6)
    
    cl.test_gibbs_duhem(system, V, T, z)
    gc.collect()
