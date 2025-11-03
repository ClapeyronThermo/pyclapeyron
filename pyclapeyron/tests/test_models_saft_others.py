import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np

# @testset "SAFT models - misc" begin
def test_ogSAFT():
    T = 298.15
    V = 1e-4
    z = np.array([0.5, 0.5])
    
    system = cl.ogSAFT(["water", "ethylene glycol"])
    assert cl.Clapeyron.a_seg(system, V, T, z) == approx(-2.0332062924093366, rel=1e-6)
    assert cl.Clapeyron.a_chain(system, V, T, z) == approx(-0.006317441684202759, rel=1e-6)
    assert cl.Clapeyron.a_assoc(system, V, T, z) == approx(-4.034042081699316, rel=1e-6)

def test_CKSAFT():
    T = 298.15
    V = 1e-4
    z = np.array([0.5, 0.5])
    
    system = cl.CKSAFT(["carbon dioxide", "2-propanol"])
    assert cl.Clapeyron.a_seg(system, V, T, z) == approx(-1.24586302917188, rel=1e-6)
    assert cl.Clapeyron.a_chain(system, V, T, z) == approx(-0.774758615408493, rel=1e-6)
    assert cl.Clapeyron.a_assoc(system, V, T, z) == approx(-1.2937079004096872, rel=1e-6)

def test_sCKSAFT():
    T = 298.15
    V = 1e-4
    z = np.array([0.5, 0.5])
    
    system = cl.sCKSAFT(["benzene", "acetic acid"])
    assert cl.Clapeyron.a_seg(system, V, T, z) == approx(-3.1809330810925256, rel=1e-6)
    assert cl.Clapeyron.a_assoc(system, V, T, z) == approx(-3.3017434376105514, rel=1e-6)

def test_BACKSAFT():
    T = 298.15
    V = 1e-4
    z1 = np.array([1.0])
    
    system = cl.BACKSAFT(["carbon dioxide"])
    assert cl.Clapeyron.a_hcb(system, V, T, z1) == approx(1.0118842111801198, rel=1e-6)
    assert cl.Clapeyron.a_chain(system, V, T, z1) == approx(-0.14177009317268635, rel=1e-6)
    assert cl.Clapeyron.a_disp(system, V, T, z1) == approx(-2.4492518566426296, rel=1e-6)

def test_LJSAFT():
    T = 298.15
    V = 1e-4
    z = np.array([0.5, 0.5])
    
    system = cl.LJSAFT(["ethane", "1-propanol"])
    assert cl.Clapeyron.a_seg(system, V, T, z) == approx(-2.207632433058473, rel=1e-6)
    assert cl.Clapeyron.a_chain(system, V, T, z) == approx(-0.04577483379871112, rel=1e-6)
    assert cl.Clapeyron.a_assoc(system, V, T, z) == approx(-1.3009761155167205, rel=1e-6)

def test_SAFTVRSW():
    T = 298.15
    V = 1e-4
    z = np.array([0.5, 0.5])
    
    system = cl.SAFTVRSW(["water", "ethane"])
    assert cl.Clapeyron.a_mono(system, V, T, z) == approx(-1.4367205951569462, rel=1e-6)
    assert cl.Clapeyron.a_chain(system, V, T, z) == approx(0.022703335564111336, rel=1e-6)
    assert cl.Clapeyron.a_assoc(system, V, T, z) == approx(-0.5091186813915323, rel=1e-6)

def test_softSAFT():
    T = 298.15
    V = 1e-4
    z = np.array([0.5, 0.5])
    
    system = cl.softSAFT(["hexane", "1-propanol"])
    assert cl.Clapeyron.a_LJ(system, V, T, z) == approx(-3.960728242264164, rel=1e-6)
    assert cl.Clapeyron.a_chain(system, V, T, z) == approx(0.3736728407455211, rel=1e-6)
    assert cl.Clapeyron.a_assoc(system, V, T, z) == approx(-2.0461376618069034, rel=1e-6)

def test_softSAFT2016():
    T = 298.15
    V = 1e-4
    z = np.array([0.5, 0.5])
    
    system = cl.softSAFT2016(["hexane", "1-propanol"])
    assert cl.Clapeyron.a_LJ(system, V, T, z) == approx(-3.986690073534575, rel=1e-6)

def test_solidsoftSAFT():
    T = 298.15
    z1 = np.array([1.0])
    
    system = cl.solidsoftSAFT(["octane"])
    V_sol = 1e-4
    assert cl.Clapeyron.a_LJ(system, V_sol, T, z1) == approx(7.830498923903852, rel=1e-6)
    assert cl.Clapeyron.a_chain(system, V_sol, T, z1) == approx(-2.3460460361188207, rel=1e-6)

# @testset "CPA" begin
def test_CPA():
    T = 298.15
    V = 1e-4
    z = np.array([0.5, 0.5])
    
    system = cl.CPA(["ethanol", "benzene"])
    assert cl.Clapeyron.a_assoc(system, V, T, z) == approx(-1.1575210505284332, rel=1e-6)

def test_sCPA():
    T = 298.15
    V = 1e-4
    z = np.array([0.5, 0.5])
    
    system = cl.sCPA(["water", "carbon dioxide"])
    assert cl.Clapeyron.a_assoc(system, V, T, z) == approx(-1.957518287413705, rel=1e-6)

# @testset "COFFEE" begin
def test_COFFEE():
    T = 298.15
    V = 1e-4
    z = np.array([1.0])
    
    system = cl.COFFEE(["a1"], userlocations=dict(
        Mw=np.array([1.]),
        segment=np.array([1.]),
        sigma=np.array([[3.]]),
        epsilon=np.array([[300.]]),
        lambda_r=np.array([[12]]),
        lambda_a=np.array([[6]]),
        shift=np.array([3*0.15]),
        dipole=np.array([2.0*1.0575091914494172]),))
    assert cl.Clapeyron.a_ff(system, V, T, z) == approx(-2.3889723633885107, rel=1e-6)
    assert cl.Clapeyron.a_nf(system, V, T, z) == approx(-0.7418363729609996, rel=1e-6)
