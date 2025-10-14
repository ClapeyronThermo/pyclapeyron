import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np

# @testset "Cubic models" begin
def test_Default_vdW():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = [0.5, 0.5]
    
    system = cl.vdW(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-0.7088380780265725, rel=1e-6)
    assert cl.cubic_poly(system, p, T, z)[0][0] == approx(-0.0002475728429728521, rel=1e-6)
    assert cl.cubic_p(system, V, T, z) == approx(cl.pressure(system, V, T, z), rel=1e-6)

def test_Clausius():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.Clausius(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-1.2945136000972637, rel=1e-6)

def test_Berthelot():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.Berthelot(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-0.5282310106210891, rel=1e-6)

# @testset "RK Models" begin
def test_Default_RK():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = [0.5, 0.5]
    
    system = cl.RK(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-0.9825375012134132, rel=1e-6)
    assert cl.cubic_poly(system, p, T, z)[0][0] == approx(-0.00022230043592123767, rel=1e-6)
    assert cl.cubic_p(system, V, T, z) == approx(cl.pressure(system, V, T, z), rel=1e-6)
    cl.test_recombine(system)
    cl.test_scales(system)
    cl.test_kl(system)

def test_SRK():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.SRK(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-1.2572506872856557, rel=1e-6)

def test_PSRK():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.PSRK(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-1.2265133881057408, rel=1e-6)

def test_tcRK():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.tcRK(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-1.2519495469149748, rel=1e-6)

def test_RK_BMAlpha():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.RK(["ethane", "undecane"], alpha=cl.BMAlpha)
    assert cl.a_res(system, V, T, z) == approx(-1.2569334957019538, rel=1e-6)

def test_RK_PenelouxTranslation():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.RK(["ethane", "undecane"], translation=cl.PenelouxTranslation)
    assert cl.a_res(system, V, T, z) == approx(-0.9800472116681871, rel=1e-6)

def test_RK_KayRule():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.RK(["ethane", "undecane"], mixing=cl.KayRule)
    assert cl.a_res(system, V, T, z) == approx(-0.8176850121211936, rel=1e-6)

def test_RK_HVRule():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.RK(["methanol", "benzene"], mixing=cl.HVRule, activity=cl.Wilson)
    assert cl.a_res(system, V, T, z) == approx(-0.5209112693371991, rel=1e-6)

def test_RK_MHV1Rule():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.RK(["methanol", "benzene"], mixing=cl.MHV1Rule, activity=cl.Wilson)
    assert cl.a_res(system, V, T, z) == approx(-0.5091065987876959, rel=1e-6)

def test_RK_MHV2Rule():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.RK(["methanol", "benzene"], mixing=cl.MHV2Rule, activity=cl.Wilson)
    assert cl.a_res(system, V, T, z) == approx(-0.5071048453490448, rel=1e-6)

def test_RK_WSRule():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.RK(["methanol", "benzene"], mixing=cl.WSRule, activity=cl.Wilson)
    assert cl.a_res(system, V, T, z) == approx(-0.5729126903890258, rel=1e-6)

def test_RK_modWSRule():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.RK(["methanol", "benzene"], mixing=cl.modWSRule, activity=cl.Wilson)
    assert cl.a_res(system, V, T, z) == approx(-0.5531088611093051, rel=1e-6)

# @testset "PR Models" begin
def test_Default_PR():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = [0.5, 0.5]
    
    system = cl.PR(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-1.244774062489359, rel=1e-6)
    assert cl.cubic_poly(system, p, T, z)[0][0] == approx(-0.0002328543992909459, rel=1e-6)
    assert cl.cubic_p(system, V, T, z) == approx(cl.pressure(system, V, T, z), rel=1e-6)

def test_PR78():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.PR78(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-1.246271020258271, rel=1e-6)

def test_VTPR():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.VTPR(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-1.234012667532541, rel=1e-6)

def test_UMRPR():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.UMRPR(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-1.1447330557895619, rel=1e-6)
    cl.test_l(system)

def test_QCPR():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.QCPR(["neon", "helium"])
    assert cl.a_res(system, V, 25, z) == approx(-0.04727884027682022, rel=1e-6)
    assert cl.lb_volume(system, 25, z) == approx(1.3601716423130568e-5, rel=1e-6)
    cl.test_scales(system)
    cl.test_kl(system)
    _a, _b, _c = cl.cubic_ab(system, V, 25, z)
    assert _a == approx(0.012772722389495079, rel=1e-6)
    assert _b == approx(1.0728356231510917e-5, rel=1e-6)
    assert _c == approx(-2.87335e-6, rel=1e-6)
    # test for the single component branch
    system1 = cl.QCPR(["helium"])
    a1 = cl.a_res(system, V, 25, [0.0, 1.0])
    assert cl.a_res(system1, V, 25, [1.0]) == approx(a1)

def test_cPR():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.cPR(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-1.2438144556398565, rel=1e-6)

def test_tcPR():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.tcPR(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-1.254190142912733, rel=1e-6)

def test_tcPR_Wilson_Res():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.tcPRW(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-1.2106271631685903, rel=1e-6)

def test_EPPR78():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.EPPR78(["benzene", "isooctane"])
    assert cl.a_res(system, V, T, z) == approx(-1.1415931771485368, rel=1e-6)

def test_PR_BMAlpha():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.PR(["ethane", "undecane"], alpha=cl.BMAlpha)
    assert cl.a_res(system, V, T, z) == approx(-1.2445088818575114, rel=1e-6)

def test_PR_TwuAlpha():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.PR(["ethane", "undecane"], alpha=cl.TwuAlpha)
    assert cl.a_res(system, V, T, z) == approx(-1.2650756692036234, rel=1e-6)

def test_PR_MTAlpha():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.PR(["ethane", "undecane"], alpha=cl.MTAlpha)
    assert cl.a_res(system, V, T, z) == approx(-1.2542346631395425, rel=1e-6)

def test_PR_LeiboviciAlpha():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.PR(["ethane", "undecane"], alpha=cl.LeiboviciAlpha)
    assert cl.a_res(system, V, T, z) == approx(-1.2480909069722526, rel=1e-6)

def test_PR_RackettTranslation():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.PR(["ethane", "undecane"], translation=cl.RackettTranslation)
    assert cl.a_res(system, V, T, z) == approx(-1.2453853855058576, rel=1e-6)

def test_PR_MTTranslation():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.PR(["ethane", "undecane"], translation=cl.MTTranslation)
    assert cl.a_res(system, V, T, z) == approx(-1.243918482158021, rel=1e-6)

def test_PR_HVRule():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.PR(["methanol", "benzene"], mixing=cl.HVRule, activity=cl.Wilson)
    assert cl.a_res(system, V, T, z) == approx(-0.6329827794751909, rel=1e-6)

def test_PR_MHV1Rule():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.PR(["methanol", "benzene"], mixing=cl.MHV1Rule, activity=cl.Wilson)
    assert cl.a_res(system, V, T, z) == approx(-0.6211441939544694, rel=1e-6)

def test_PR_MHV2Rule():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.PR(["methanol", "benzene"], mixing=cl.MHV2Rule, activity=cl.Wilson)
    assert cl.a_res(system, V, T, z) == approx(-0.6210843663212396, rel=1e-6)

def test_PR_LCVMRule():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.PR(["methanol", "benzene"], mixing=cl.LCVMRule, activity=cl.Wilson)
    assert cl.a_res(system, V, T, z) == approx(-0.6286241404575419, rel=1e-6)

def test_PR_WSRule():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.PR(["methanol", "benzene"], mixing=cl.WSRule, activity=cl.Wilson)
    assert cl.a_res(system, V, T, z) == approx(-0.6690864227574802, rel=1e-6)

# @testset "KU Models" begin
def test_KU_Models():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.KU(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-1.2261554720898895, rel=1e-6)
    assert cl.cubic_p(system, V, T, z) == approx(cl.pressure(system, V, T, z), rel=1e-6)

# @testset "RKPR Models" begin
def test_RKPR_Models():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.RKPR(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-1.2877492838069213, rel=1e-6)

# @testset "Patel Teja Models" begin
def test_Patel_Teja():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.PatelTeja(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-1.2284322450064429, rel=1e-6)
    assert cl.cubic_p(system, V, T, z) == approx(cl.pressure(system, V, T, z), rel=1e-6)

def test_PTV():
    T = 333.15
    V = 1e-3
    z = [0.5, 0.5]
    
    system = cl.PTV(["ethane", "undecane"])
    assert cl.a_res(system, V, T, z) == approx(-1.2696422558756286, rel=1e-6)
    assert cl.cubic_p(system, V, T, z) == approx(cl.pressure(system, V, T, z), rel=1e-6)

def test_YFR():
    T = 333.15
    V = 1e-3
    zz = [0.95, 0.05]
    
    system = cl.YFR(["ethane", "undecane"])
    assert cl.a_res(system, V, T, zz) == approx(0.17277878581138775, rel=1e-6)
    assert cl.cubic_p(system, V, T, zz) == approx(cl.pressure(system, V, T, zz), rel=1e-6)
    assert cl.crit_pure(cl.YFR("water"))[2] == approx(6.50936094952025e-5, rel=1e-6)
