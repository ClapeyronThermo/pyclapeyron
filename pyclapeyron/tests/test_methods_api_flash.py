import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np
from juliacall import Main as jl

# @testset "Tp flash algorithms" begin
def test_RR_Algorithm():
    # this is a VLLE equilibria
    system = cl.PCSAFT(["water", "cyclohexane", "propane"])
    T = 298.15
    p = 1e5
    z = np.array([0.333, 0.333, 0.334])
    
    method = cl.RRTPFlash()
    assert cl.tp_flash(system, p, T, z, method)[2] == approx(-6.539976318817461, rel=1e-6)
    
    # test for initialization when K suggests single phase but it could be solved supposing bubble or dew conditions.
    substances = ["water", "methanol", "propyleneglycol", "methyloxirane"]
    pcp_system = cl.PCPSAFT(substances)
    res = cl.Clapeyron.tp_flash2(pcp_system, 25_000.0, 300.15, np.array([1.0, 1.0, 1.0, 1.0]), cl.RRTPFlash())
    assert res.data.g == approx(-8.900576759774916, rel=1e-6)
    
    # https://julialang.zulipchat.com/#narrow/channel/265161-Clapeyron.2Ejl/topic/The.20meaning.20of.20subcooled.20liquid.20flash.20results
    z_zulip1 = np.array([0.25, 0.25, 0.25, 0.25])
    p_zulip1 = 1e5
    model_zulip1 = cl.PR(["IsoButane", "n-Butane", "n-Pentane", "n-Hexane"])
    res1 = cl.Clapeyron.tp_flash2(model_zulip1, p_zulip1, 282.2, z_zulip1, cl.RRTPFlash(equilibrium=jl.Symbol("vle")))
    res2 = cl.Clapeyron.tp_flash2(model_zulip1, p_zulip1, 282.3, z_zulip1, cl.RRTPFlash(equilibrium=jl.Symbol("vle")))
    assert res1.fractions[1] == 0
    assert res2.fractions[1] == approx(0.00089161, rel=1e-6)

# MultiComponentFlash.jl tests would be skipped in Python unless that extension is available

def test_DE_Algorithm():
    system = cl.PCSAFT(["water", "cyclohexane", "propane"])
    T = 298.15
    p = 1e5
    z = np.array([0.333, 0.333, 0.334])
    
    # VLLE eq
    assert cl.tp_flash(system, p, T, z, cl.DETPFlash(numphases=3))[2] == approx(-6.759674475174073, rel=1e-6)
    
    # LLE eq with activities
    act_system = cl.UNIFAC(["water", "cyclohexane", "propane"])
    flash0 = cl.tp_flash(act_system, p, T, np.array([0.5, 0.5, 0.0]), cl.DETPFlash(equilibrium=jl.Symbol("lle")))
    act_x0 = cl.activity_coefficient(act_system, p, T, flash0[0][0, :]) * flash0[0][0, :]
    act_y0 = cl.activity_coefficient(act_system, p, T, flash0[0][1, :]) * flash0[0][1, :]
    assert cl.Clapeyron.dnorm(act_x0, act_y0) < 0.01  # not the most accurate, but it is global

def test_Multiphase_algorithm():
    system = cl.PCSAFT(["water", "cyclohexane", "propane"])
    T = 298.15
    p = 1e5
    z = np.array([0.333, 0.333, 0.334])
    
    assert cl.tp_flash(system, p, T, z, cl.MultiPhaseTPFlash())[2] == approx(-6.759674475175065, rel=1e-6)
    system2 = cl.PR(["IsoButane", "n-Butane", "n-Pentane", "n-Hexane"])
    assert cl.tp_flash(system2, 1e5, 284.4, np.array([1, 1, 1, 1]) * 0.25, cl.MultiPhaseTPFlash())[2] == approx(-6.618441125949686, rel=1e-6)

def test_Michelsen_Algorithm():
    system = cl.PCSAFT(["water", "cyclohexane", "propane"])
    T = 298.15
    p = 1e5
    
    x0 = np.array([0.9997755902156433, 0.0002244097843566859, 0.0])
    y0 = np.array([6.425238373915699e-6, 0.9999935747616262, 0.0])
    method = cl.MichelsenTPFlash(x0=x0, y0=y0, equilibrium=jl.Symbol("lle"))
    assert cl.tp_flash(system, p, T, np.array([0.5, 0.5, 0.0]), method)[2] == approx(-7.577270350886795, rel=1e-6)
    
    method2 = cl.MichelsenTPFlash(x0=x0, y0=y0, equilibrium=jl.Symbol("lle"), ss_iters=4, second_order=False)
    assert cl.tp_flash(system, p, T, np.array([0.5, 0.5, 0.0]), method2)[2] == approx(-7.577270350886795, rel=1e-6)
    
    method3 = cl.MichelsenTPFlash(x0=x0, y0=y0, equilibrium=jl.Symbol("lle"), ss_iters=4, second_order=True)
    assert cl.tp_flash(system, p, T, np.array([0.5, 0.5, 0.0]), method3)[2] == approx(-7.577270350886795, rel=1e-6)

def test_Michelsen_Algorithm_nonvolatiles_noncondensables():
    system = cl.PCSAFT(["hexane", "ethanol", "methane", "decane"])
    T = 320.  # K
    p = 1e5  # Pa
    z = np.array([0.25, 0.25, 0.25, 0.25])
    x0 = np.array([0.3, 0.3, 0., 0.4])
    y0 = np.array([0.2, 0.2, 0.6, 0.])
    
    method_normal = cl.MichelsenTPFlash(x0=x0, y0=y0, second_order=True)
    expected_normal = np.array([[0.291924, 0.306002, 0.00222251, 0.399851],
                                [0.181195, 0.158091, 0.656644, 0.00406898]])
    assert np.allclose(cl.tp_flash(system, p, T, z, method_normal)[0], expected_normal, rtol=1e-5)
    
    method_nonvolatiles = cl.MichelsenTPFlash(x0=x0, y0=y0, ss_iters=1, second_order=True, nonvolatiles=["decane"])
    expected_nonvol = np.array([[0.291667, 0.305432, 0.00223826, 0.400663],
                                [0.180861, 0.15802, 0.661119, 0.0]])
    assert np.allclose(cl.tp_flash(system, p, T, z, method_nonvolatiles)[0], expected_nonvol, rtol=1e-5)
    
    method_noncondensables = cl.MichelsenTPFlash(x0=x0, y0=y0, ss_iters=1, second_order=False, noncondensables=["methane"])
    expected_noncond = np.array([[0.292185, 0.306475, 0.0, 0.40134],
                                 [0.181452, 0.158233, 0.65623, 0.00408481]])
    assert np.allclose(cl.tp_flash(system, p, T, z, method_noncondensables)[0], expected_noncond, rtol=1e-5)
    
    method_both = cl.MichelsenTPFlash(x0=x0, y0=y0, ss_iters=1, second_order=False, noncondensables=["methane"], nonvolatiles=["decane"])
    expected_both = np.array([[0.291928, 0.3059, 0.0, 0.402171],
                              [0.181116, 0.158162, 0.660722, 0.0]])
    assert np.allclose(cl.tp_flash(system, p, T, z, method_both)[0], expected_both, rtol=1e-5)
    
    # water-oxygen system, non-condensables
    model_a_ideal = cl.CompositeModel(["water", "oxygen"], liquid=cl.RackettLiquid, gas=cl.BasicIdeal, saturation=cl.DIPPR101Sat)
    expected_wo = np.array([[1.0, 0.0],
                            [0.23252954843762222, 0.7674704515623778]])
    assert np.allclose(cl.tp_flash(model_a_ideal, 134094.74892634258, 70 + 273.15, np.array([18500.0, 24.08]), noncondensables=["oxygen"])[0], expected_wo, rtol=1e-5)
    
    # 403
    model403 = cl.PCSAFT(["water", "carbon dioxide"])
    res = cl.Clapeyron.tp_flash2(model403, 1e5, 323.15, np.array([0.5, 0.5]), cl.MichelsenTPFlash(nonvolatiles=["water"]))
    assert list(res.compositions[1]) == [0., 1.]
    assert np.allclose(list(res.compositions[0]), [0.999642, 0.000358065], rtol=1e-5)

def test_Michelsen_Algorithm_activities():
    # example from https://github.com/ClapeyronThermo/Clapeyron.jl/issues/144
    system = cl.UNIFAC(["water", "hexane"])
    alg1 = cl.MichelsenTPFlash(
        equilibrium=jl.Symbol("lle"),
        K0=np.array([0.00001/0.99999, 0.99999/0.00001]),
    )
    
    flash1 = cl.tp_flash(system, 101325, 303.15, np.array([0.5, 0.5]), alg1)
    act_x1 = cl.activity_coefficient(system, 101325, 303.15, flash1[0][0, :]) * flash1[0][0, :]
    act_y1 = cl.activity_coefficient(system, 101325, 303.15, flash1[0][1, :]) * flash1[0][1, :]
    assert cl.Clapeyron.dnorm(act_x1, act_y1) < 1e-8
    
    alg2 = cl.RRTPFlash(
    equilibrium=jl.Symbol("lle"),
        x0=np.array([0.99999, 0.00001]),
        y0=np.array([0.00001, 0.00009])
    )
    flash2 = cl.tp_flash(system, 101325, 303.15, np.array([0.5, 0.5]), alg2)
    act_x2 = cl.activity_coefficient(system, 101325, 303.15, flash2[0][0, :]) * flash2[0][0, :]
    act_y2 = cl.activity_coefficient(system, 101325, 303.15, flash2[0][1, :]) * flash2[0][1, :]
    assert cl.Clapeyron.dnorm(act_x2, act_y2) < 1e-8
    
    # test K0_lle_init initialization
    alg3 = cl.RRTPFlash(equilibrium=jl.Symbol("lle"))
    flash3 = cl.tp_flash(system, 101325, 303.15, np.array([0.5, 0.5]), alg3)
    act_x3 = cl.activity_coefficient(system, 101325, 303.15, flash3[0][0, :]) * flash3[0][0, :]
    act_y3 = cl.activity_coefficient(system, 101325, 303.15, flash3[0][1, :]) * flash3[0][1, :]
    assert cl.Clapeyron.dnorm(act_x3, act_y3) < 1e-8
    
    # test combinations of Activity + CompositeModel
    system_fluid = cl.CompositeModel(["water", "hexane"], gas=cl.BasicIdeal, liquid=cl.RackettLiquid, saturation=cl.LeeKeslerSat)
    system_cc = cl.CompositeModel(["water", "hexane"], liquid=cl.UNIFAC, fluid=system_fluid)
    flash3 = cl.tp_flash(system_cc, 101325, 303.15, np.array([0.5, 0.5]), alg2)
    act_x3 = cl.activity_coefficient(system_cc, 101325, 303.15, flash3[0][0, :]) * flash3[0][0, :]
    act_y3 = cl.activity_coefficient(system_cc, 101325, 303.15, flash3[0][1, :]) * flash3[0][1, :]
    assert cl.Clapeyron.dnorm(act_x3, act_y3) < 1e-8
    
    # running the vle part
    model_vle = cl.CompositeModel(["octane", "heptane"], liquid=cl.UNIFAC, fluid=cl.cPR)
    flash4 = cl.tp_flash(model_vle, 2500.0, 300.15, np.array([0.9, 0.1]), cl.MichelsenTPFlash())
    expected_vle = np.array([[0.923964726801428, 0.076035273198572],
                             [0.7934765930306608, 0.20652340696933932]])
    assert np.allclose(flash4[0], expected_vle, rtol=1e-6)

def test_Michelsen_Algorithm_CompositeModel():
    p, T, z = 101325., 85+273., np.array([0.2, 0.8])
    system = cl.CompositeModel(["water", "ethanol"], gas=cl.BasicIdeal, liquid=cl.RackettLiquid, saturation=cl.LeeKeslerSat)
    expected = np.array([[0.3618699659002134, 0.6381300340997866],
                         [0.17888243361092543, 0.8211175663890746]])
    assert np.allclose(cl.tp_flash(system, p, T, z, cl.MichelsenTPFlash())[0], expected, rtol=1e-6)
    
    with pytest.raises(Exception):
        cl.tp_flash(system, p, T, z, cl.MichelsenTPFlash(ss_iters=0))

# @testset "XY flash" begin
def test_XY_flash():
    # 1 phase (#320)
    model = cl.cPR(["ethane", "methane"], idealmodel=cl.ReidIdeal)
    p = 101325.0
    z = np.array([1.2, 1.2])
    T = 350.0
    h = cl.enthalpy(model, p, T, z)
    res0 = cl.ph_flash(model, p, h, z)
    assert cl.Clapeyron.temperature(res0) == approx(T, rel=1e-6)
    assert cl.PH.temperature(model, p, h, z) == approx(T, rel=1e-6)
    assert cl.Clapeyron.temperature(cl.PH.flash(model, p, h, z)) == approx(T, rel=1e-6)
    assert cl.enthalpy(model, res0) == approx(h, rel=1e-6)
    
    # 2 phases
    h = -13831.0
    res1 = cl.ph_flash(model, p, h, z)
    assert cl.enthalpy(model, res1) == approx(h, rel=1e-6)
    
    # test for ps_flash:
    model = cl.cPR(["ethane"], idealmodel=cl.ReidIdeal)
    s = 100
    p = 101325
    z = np.array([1.0])
    h = cl.PS.enthalpy(model, p, s, z)
    s2 = cl.PH.entropy(model, p, h, z)
    assert s == approx(s2, rel=1e-6)
    
    # examples for qt, qp flash (#314)
    model = cl.cPR(["ethane", "propane"], idealmodel=cl.ReidIdeal)
    res2 = cl.qt_flash(model, 0.5, 208.0, np.array([0.5, 0.5]))
    assert cl.pressure(res2) == approx(101634.82435966855, rel=1e-6)
    assert cl.QT.pressure(model, 0.5, 208.0, np.array([0.5, 0.5])) == approx(101634.82435966855, rel=1e-6)
    res3 = cl.qp_flash(model, 0.5, 120000.0, np.array([0.5, 0.5]))
    assert cl.Clapeyron.temperature(res3) == approx(211.4972567716822, rel=1e-6)
    assert cl.QP.temperature(model, 0.5, 120000.0, np.array([0.5, 0.5])) == approx(211.4972567716822, rel=1e-6)
    
    # # 1 phase input should error
    model = cl.PR(["IsoButane", "n-Butane", "n-Pentane", "n-Hexane"])
    z = np.array([0.25, 0.25, 0.25, 0.25])
    # p = 1e5
    # h = 6300.0
    # r = cl.ph_flash(model, p, h, z)
    # with pytest.raises(ValueError):
    #     cl.qt_flash(model, 0.5, 308, z, flash_result=r)
    
    res4 = cl.qp_flash(model, 0.7, 60000.0, z)
    T4 = cl.Clapeyron.temperature(res4)
    assert cl.pressure(model, res4.volumes[0], T4, res4.compositions[0]) == approx(60000.0, rel=1e-6)
    assert cl.pressure(model, res4.volumes[1], T4, res4.compositions[1]) == approx(60000.0, rel=1e-6)
    
    # example in documentation for xy_flash
    spec = cl.FlashSpecifications(p=101325.0, T=200.15)  # p-T flash
    model = cl.cPR(["ethane", "propane"], idealmodel=cl.ReidIdeal)
    z = np.array([0.5, 0.5])  # bulk composition
    x1 = np.array([0.25, 0.75])  # liquid composition
    x2 = np.array([0.75, 0.25])  # gas composition
    compositions = [x1, x2]
    volumes = np.array([6.44e-5, 0.016])
    fractions = np.array([0.5, 0.5])
    p0, T0 = np.nan, np.nan  # in p-T flash, pressure and temperature are already specifications
    data = cl.FlashData(p0, T0)
    result0 = cl.FlashResult(compositions, fractions, volumes, data)
    result = cl.xy_flash(model, spec, z, result0)
    assert cl.Clapeyron.temperature(result) == 200.15
    assert cl.pressure(result) == 101325.0
    
    # px_flash_pure/tx_flash_pure, 1 phase (#320)
    model = cl.cPR(["ethane"], idealmodel=cl.ReidIdeal)
    p = 101325
    h = 100
    z = np.array([1])
    T = cl.PH.temperature(model, p, h, z)
    assert cl.enthalpy(model, p, T, z) == approx(h, rel=1e-6)
    res5 = cl.Clapeyron.tx_flash_pure(model, T, h, z, cl.Clapeyron.enthalpy)
    assert cl.pressure(res5) == approx(p, rel=1e-6)
    
    # px_flash_pure: two phase (#320)
    model = cl.cPR(["ethane"], idealmodel=cl.ReidIdeal)
    p = 101325
    z = np.array([5.0])
    T = cl.saturation_temperature(model, p)[0]
    h_liq = cl.enthalpy(model, p, T-0.1, z)
    h_gas = cl.enthalpy(model, p, T+0.1, z)
    h = (h_liq + h_gas)/2
    T2 = cl.PH.temperature(model, p, h, z)
    assert T2 == approx(T, rel=1e-6)
    
    # QP flash pure (#325)
    model = cl.cPR(["methane"], idealmodel=cl.ReidIdeal)
    p = 101325.0
    q = 1.0
    z = np.array([5.0])
    res_qp1 = cl.qp_flash(model, q, p, z)
    assert cl.Clapeyron.temperature(res_qp1) == cl.saturation_temperature(model, p)[0]
    res_qt1 = cl.qt_flash(model, 0.99, 160.0, z)
    assert cl.pressure(res_qt1) == cl.saturation_pressure(model, 160.0)[0]
    
    # QP: test a range of vapour fractions
    q = np.linspace(0.001, 0.999, 100)
    fluids = ["isobutane", "pentane"]
    model = cl.cPR(fluids, idealmodel=cl.ReidIdeal)
    p = 101325.0
    z = np.array([5.0, 5.0])
    qp_flashes = [cl.qp_flash(model, qi, p, z) for qi in q]
    T = [cl.Clapeyron.temperature(flash) for flash in qp_flashes]
    assert max(np.diff(T)) < 0.25
    
    # bubble/dew temperatures via qp_flash
    Tbubble0 = cl.bubble_temperature(model, p, z)[0]
    Tbubble1 = cl.Clapeyron.temperature(cl.qp_flash(model, 0, p, z))
    assert Tbubble0 == approx(Tbubble1, rel=1e-6)
    
    Tdew0 = cl.dew_temperature(model, p, z)[0]
    Tdew1 = cl.Clapeyron.temperature(cl.qp_flash(model, 1, p, z))
    # Note: this test only fails in ubuntu-latest 1.11.3
    # assert Tdew0 == approx(Tdew1, rel=1e-6)
    
    # qp_flash unmasked an error in the calculation of the initial K-values (#325)
    fluids = ["isobutane", "toluene"]
    model = cl.cPR(fluids, idealmodel=cl.ReidIdeal)
    p = 8*101325.0
    z = np.array([5.0, 5.0])
    res_qp2 = cl.qp_flash(model, 0.4, p, z)
    assert np.allclose(res_qp2.fractions, [6.0, 4.0])
    
    # qp_flash scaling error (#325)
    fluids = ["isopentane", "isobutane"]
    model = cl.cPR(fluids, idealmodel=cl.ReidIdeal)
    p = 2*101325.0
    z = np.array([2.0, 5.0])
    q = 0.062744140625
    res_qp3 = cl.qp_flash(model, q, p, z)
    res_qp4 = cl.qp_flash(model, q, p, z / 10)
    assert cl.Clapeyron.temperature(res_qp3) == approx(cl.Clapeyron.temperature(res_qp4))
    
    # VT flash (#331)
    model_a_pr = cl.PR(["water", "oxygen"])
    V_a = 1  # m3
    T = 273.15 + 60  # K
    p_a = 1.2e5  # Pa, 1.2 bar
    n_H2O_a = 1.85e4  # mol H2O
    n_O2_a = 24.08  # mol O2
    sol_fl = cl.vt_flash(model_a_pr, V_a, T, np.array([n_H2O_a, n_O2_a]))
    assert V_a == approx(cl.volume(sol_fl))
    # water_cpr = cl.cPR(["water"], idealmodel=cl.ReidIdeal)
    # with pytest.raises(ValueError):
    #     cl.VT.speed_of_sound(water_cpr, 1e-4, 373.15)
    # water_cpr_flash = cl.VT.flash(water_cpr, 1e-4, 373.15)
    # with pytest.raises(ValueError):
    #     cl.speed_of_sound(water_cpr, water_cpr_flash)
    
    # PH flash with supercritical pure components (#361)
    fluid_model = cl.SingleFluid("Hydrogen")
    T_in = 70  # K
    p_in = 350e5  # Pa
    h_in = cl.enthalpy(fluid_model, p_in, T_in)
    sol_sc = cl.ph_flash(fluid_model, p_in, h_in)
    assert cl.Clapeyron.temperature(sol_sc) == approx(T_in)
    
    # PH Flash where T is in the edge (#373)
    model = cl.cPR(["butane", "isopentane"], idealmodel=cl.ReidIdeal)
    p = 101325
    z = np.array([1.0, 1.0])
    T = 286.43023797357927
    h = -50380.604181769755
    flash_res_ph = cl.ph_flash(model, p, h, z)
    assert cl.Clapeyron.numphases(flash_res_ph) == 2
    
    # Inconsistency in flash computations near bubble and dew points (#353)
    fluids = ["isopentane", "toluene"]
    model = cl.cPR(fluids, idealmodel=cl.ReidIdeal)
    p = 101325
    z = np.array([1.5, 1.5])
    T1, T2 = 380, 307.72162335900924
    h1, h2 = 30118.26278687942, -89833.18975112544
    hrange = np.linspace(h1, h2, 100)
    Trange = np.zeros_like(hrange)
    for i in range(len(hrange)):
        Ti = cl.PH.temperature(model, p, hrange[i], z)
        Trange[i] = Ti
        if i > 0:
            assert Trange[i] < Trange[i-1]  # check that temperature is decreasing
            assert np.isfinite(Ti)  # test that there are no NaNs
    
    # VT flash: water + a tiny amount of hydrogen (#377)
    n_H2O_c = 0.648e4
    V_c = 0.35
    n_H2_c = 251
    mod_pr = cl.cPR(["water", "hydrogen"], idealmodel=cl.ReidIdeal)
    mult_H2 = np.flip(np.arange(0, 5.1, 0.1))
    p_tank = np.zeros_like(mult_H2)
    T_tank = 70 + 273.15
    for i, mH2 in enumerate(mult_H2):
        res_i = cl.vt_flash(mod_pr, V_c, T_tank, np.array([n_H2O_c, 10 ** (-mH2) * n_H2_c]))
        p_tank[i] = cl.pressure(res_i)
    assert np.sum(np.isnan(p_tank)) == 0
    assert np.all(np.diff(p_tank) >= 0)  # issorted equivalent
    
    # 394
    fluid394 = cl.cPR(["R134a"], idealmodel=cl.ReidIdeal)
    h394 = -25000.0

# @testset "Saturation Methods" begin
def test_Saturation_Methods():
    model = cl.PR(["water"])
    vdw = cl.vdW(["water"])
    p0 = 1e5
    T = 373.15
    p, vl, vv = cl.saturation_pressure(model, T)  # default
    
    px, vlx, vvx = cl.saturation_pressure(vdw, T)  # vdw
    
    p1, vl1, vv1 = cl.Clapeyron.saturation_pressure_impl(model, T, cl.IsoFugacitySaturation())
    assert p1 == approx(p, rel=1e-6)
    p2, vl2, vv2 = cl.Clapeyron.saturation_pressure_impl(model, T, cl.IsoFugacitySaturation(p0=1e5))
    assert p1 == approx(p, rel=1e-6)
    p3, vl3, vv3 = cl.Clapeyron.saturation_pressure_impl(model, T, cl.ChemPotDensitySaturation())
    assert p3 == approx(p, rel=1e-6)
    p4, vl4, vv4 = cl.Clapeyron.saturation_pressure_impl(model, T, cl.ChemPotDensitySaturation(vl=vl, vv=vv))
    p4b, vl4b, vv4b = cl.Clapeyron.psat_chempot(model, T, vl, vv)
    assert p4 == approx(p, rel=1e-6)
    assert (p4 == p4b) and (vl4 == vl4b) and (vv4 == vv4b)
    
    # test IsoFugacity, near criticality
    Tc_near = 0.95*647.096
    psat_Tcnear = 1.496059652088857e7
    assert cl.saturation_pressure(model, Tc_near, cl.IsoFugacitySaturation())[0] == approx(psat_Tcnear, rel=1e-6)
    
    # Test that IsoFugacity fails over critical point
    assert np.isnan(cl.saturation_pressure(model, 1.1*647.096, cl.IsoFugacitySaturation())[0])
    
    # SuperAncSaturation
    p5, vl5, vv5 = cl.Clapeyron.saturation_pressure_impl(model, T, cl.SuperAncSaturation())
    assert p5 == approx(p, rel=1e-6)
    assert cl.Clapeyron.saturation_temperature_impl(model, p5, cl.SuperAncSaturation())[0] == approx(T, rel=1e-6)
    assert cl.Clapeyron.saturation_pressure_impl(vdw, T, cl.SuperAncSaturation())[0] == approx(px)
    
    # AntoineSat
    assert cl.saturation_temperature(model, p0, cl.AntoineSaturation(T0=400.0))[0] == approx(374.2401401001685, rel=1e-6)
    assert cl.saturation_temperature(model, p0, cl.AntoineSaturation(vl=vl5, vv=vv5))[0] == approx(374.2401401001685, rel=1e-6)
    # with pytest.raises(Exception):
    #     cl.saturation_temperature(model, p0, cl.AntoineSaturation(vl=vl5, T0=400))
    
    # ClapeyronSat
    assert cl.saturation_temperature(model, p0, cl.ClapeyronSaturation())[0] == approx(374.2401401001685, rel=1e-6)
    
    # Issue #290
    assert cl.saturation_temperature(cl.cPR("R1233zde"), 101325*20, crit_retry=False)[0] == approx(405.98925205830335, rel=1e-6)
    assert cl.saturation_temperature(cl.cPR("isobutane"), 1.7855513185537157e6, crit_retry=False)[0] == approx(366.52386488214876, rel=1e-6)
    assert cl.saturation_temperature(cl.cPR("propane"), 2.1298218093361156e6, crit_retry=False)[0] == approx(332.6046103831853, rel=1e-6)
    assert cl.saturation_temperature(cl.cPR("r134a"), 2.201981727901889e6, crit_retry=False)[0] == approx(344.6869001549851, rel=1e-6)
    
    # Issue 328
    assert cl.saturation_pressure(cl.cPR("butane"), 406.5487245045052)[0] == approx(2.815259927796967e6, rel=1e-6)
    
    # issue 387
    cpr = cl.cPR("Propane", idealmodel=cl.ReidIdeal)
    crit_cpr = cl.crit_pure(cpr)
    assert cl.saturation_temperature(cpr, crit_cpr[1] - 1e3)[0] == approx(369.88681908031606, rel=1e-6)

# @testset "Tproperty/Property" begin
def test_Tproperty_Property():
    model1 = cl.cPR(["propane", "dodecane"])
    p = 101325.0
    T = 300.0
    z = np.array([0.5, 0.5])
    h_ = cl.enthalpy(model1, p, T, z)
    s_ = cl.entropy(model1, p, T, z)
    assert cl.Tproperty(model1, p, h_, z, cl.Clapeyron.enthalpy) == approx(T)
    assert cl.Tproperty(model1, p, s_, z, cl.Clapeyron.entropy) == approx(T)
    
    model2 = cl.cPR(["propane"])
    z2 = np.array([1.])
    h2_ = cl.enthalpy(model2, p, T, z2)
    s2_ = cl.entropy(model2, p, T, z2)
    assert cl.Tproperty(model2, p, h2_, z2, cl.Clapeyron.enthalpy) == approx(T)
    assert cl.Tproperty(model2, p, s2_, z2, cl.Clapeyron.entropy) == approx(T)
    
    # issue 309
    model3 = cl.cPR(["ethane"], idealmodel=cl.ReidIdeal)
    T3 = 300
    z3 = np.array([5])
    s30 = cl.entropy(model3, p, T3, z3)
    p3 = 2*p
    T3_calc = cl.Tproperty(model3, p3, s30, z3, cl.Clapeyron.entropy)
    s3 = cl.entropy(model3, p3, T3_calc, z3)
    assert s3 == approx(s30)
    
    # issue 309 (https://github.com/ClapeyronThermo/Clapeyron.jl/issues/309#issuecomment-2508038968)
    model4 = cl.cPR("R134A", idealmodel=cl.ReidIdeal)
    T_crit, p_crit, _ = cl.crit_pure(model4)
    T1 = 300.0
    p1 = cl.saturation_pressure(model4, T1)[0] + 101325
    s1 = cl.entropy(model4, p1, T1)
    h1 = cl.enthalpy(model4, p1, T1)
    p2 = p_crit + 2*101325
    T2 = cl.Tproperty(model4, p2, s1, np.array([1.0]), cl.Clapeyron.entropy)
    s2 = cl.entropy(model4, p2, T2)
    h2 = cl.enthalpy(model4, p2, T2)
    assert s2 == approx(s1)
    
    # issue 409
    fluid409 = cl.cPR(["Propane", "R134a"], idealmodel=cl.ReidIdeal)
    z409 = np.array([1.0, 1.0])
    s409 = -104.95768957075641
    p409 = 5.910442025416817e6
    assert cl.Tproperty(fluid409, p409, s409, z409, cl.Clapeyron.entropy) == approx(406.0506318701147, rel=1e-6)
    
    # model5 = cl.cPR(["R134a", "propane"], idealmodel=cl.ReidIdeal) #internals
    # mix = np.array([0.5, 0.5])
    # assert cl.Clapeyron._Pproperty(model5, 450.0, 0.03, mix, cl.Clapeyron.volume)[1] == "vapour"
    # assert cl.Clapeyron._Pproperty(model5, 450.0, 0.03, mix, cl.volume)[1] == "vapour"
    # assert cl.Clapeyron._Pproperty(model5, 450.0, 0.00023, mix, cl.volume)[1] == "eq"
    # assert cl.Clapeyron._Pproperty(model5, 450.0, 0.000222, mix, cl.volume)[1] == "eq"
    # assert cl.Clapeyron._Pproperty(model5, 450.0, 0.000222, mix, cl.volume)[1] == "eq"

# @testset "bubble/dew point algorithms" begin
def test_bubble_dew_point_algorithms():
    system1 = cl.PCSAFT(["methanol", "cyclohexane"])
    p = 1e5
    T = 313.15
    z = np.array([0.5, 0.5])
    p2 = 2e6
    T2 = 443.15
    z2 = np.array([0.27, 0.73])
    
    pres1 = 54532.249600937736
    Tres1 = 435.80890506865
    pres2 = 1.6555486543884084e6
    Tres2 = 453.0056727580934
    
    # bubble pressure
    assert cl.bubble_pressure(system1, T, z, cl.ChemPotBubblePressure())[0] == approx(pres1, rel=1e-6)
    assert cl.bubble_pressure(system1, T, z, cl.ChemPotBubblePressure(y0=np.array([0.6, 0.4])))[0] == approx(pres1, rel=1e-6)
    assert cl.bubble_pressure(system1, T, z, cl.ChemPotBubblePressure(p0=5e4))[0] == approx(pres1, rel=1e-6)
    assert cl.bubble_pressure(system1, T, z, cl.ChemPotBubblePressure(p0=5e4, y0=np.array([0.6, 0.4])))[0] == approx(pres1, rel=1e-6)
    
    assert cl.bubble_pressure(system1, T, z, cl.FugBubblePressure())[0] == approx(pres1, rel=1e-6)
    assert cl.bubble_pressure(system1, T, z, cl.FugBubblePressure(y0=np.array([0.6, 0.4])))[0] == approx(pres1, rel=1e-6)
    assert cl.bubble_pressure(system1, T, z, cl.FugBubblePressure(p0=5e4))[0] == approx(pres1, rel=1e-6)
    assert cl.bubble_pressure(system1, T, z, cl.FugBubblePressure(p0=5e4, y0=np.array([0.6, 0.4])))[0] == approx(pres1, rel=1e-6)
    # test multidimensional fugacity solver
    assert cl.bubble_pressure(system1, T, z, cl.FugBubblePressure(itmax_newton=1))[0] == approx(pres1, rel=1e-6)
    
    assert cl.bubble_pressure(system1, T, z, cl.ActivityBubblePressure())[0] == approx(pres1, rel=1e-6)
    assert cl.bubble_pressure(system1, T, z, cl.ActivityBubblePressure(y0=np.array([0.6, 0.4])))[0] == approx(pres1, rel=1e-6)
    assert cl.bubble_pressure(system1, T, z, cl.ActivityBubblePressure(p0=5e4))[0] == approx(pres1, rel=1e-6)
    assert cl.bubble_pressure(system1, T, z, cl.ActivityBubblePressure(p0=5e4, y0=np.array([0.6, 0.4])))[0] == approx(pres1, rel=1e-6)
    
    # bubble temperature
    assert cl.bubble_temperature(system1, p2, z, cl.ChemPotBubbleTemperature())[0] == approx(Tres1, rel=1e-6)
    assert cl.bubble_temperature(system1, p2, z, cl.ChemPotBubbleTemperature(y0=np.array([0.7, 0.3])))[0] == approx(Tres1, rel=1e-6)
    assert cl.bubble_temperature(system1, p2, z, cl.ChemPotBubbleTemperature(T0=450))[0] == approx(Tres1, rel=1e-6)
    assert cl.bubble_temperature(system1, p2, z, cl.ChemPotBubbleTemperature(T0=450, y0=np.array([0.75, 0.25])))[0] == approx(Tres1, rel=1e-6)
    
    assert cl.bubble_temperature(system1, p2, z, cl.FugBubbleTemperature())[0] == approx(Tres1, rel=1e-6)
    assert cl.bubble_temperature(system1, p2, z, cl.FugBubbleTemperature(y0=np.array([0.75, 0.25])))[0] == approx(Tres1, rel=1e-6)
    assert cl.bubble_temperature(system1, p2, z, cl.FugBubbleTemperature(T0=450))[0] == approx(Tres1, rel=1e-6)
    assert cl.bubble_temperature(system1, p2, z, cl.FugBubbleTemperature(T0=450, y0=np.array([0.75, 0.25])))[0] == approx(Tres1, rel=1e-6)
    assert cl.bubble_temperature(system1, p2, z, cl.FugBubbleTemperature(itmax_newton=1))[0] == approx(Tres1, rel=1e-6)
    
    # dew pressure
    assert cl.dew_pressure(system1, T2, z, cl.ChemPotDewPressure())[0] == approx(pres2, rel=1e-6)
    assert cl.dew_pressure(system1, T2, z, cl.ChemPotDewPressure(x0=np.array([0.1, 0.9])))[0] == approx(pres2, rel=1e-6)
    assert cl.dew_pressure(system1, T2, z, cl.ChemPotDewPressure(p0=1.5e6))[0] == approx(pres2, rel=1e-6)
    assert cl.dew_pressure(system1, T2, z, cl.ChemPotDewPressure(p0=1.5e6, x0=np.array([0.1, 0.9])))[0] == approx(pres2, rel=1e-6)
    
    assert cl.dew_pressure(system1, T2, z, cl.FugDewPressure())[0] == approx(pres2, rel=1e-6)
    assert cl.dew_pressure(system1, T2, z, cl.FugDewPressure(x0=np.array([0.1, 0.9])))[0] == approx(pres2, rel=1e-6)
    assert cl.dew_pressure(system1, T2, z, cl.FugDewPressure(p0=1.5e6))[0] == approx(pres2, rel=1e-6)
    assert cl.dew_pressure(system1, T2, z, cl.FugDewPressure(p0=1.5e6, x0=np.array([0.1, 0.9])))[0] == approx(pres2, rel=1e-6)
    # for some reason, it requires 2 newton iterations.
    assert cl.dew_pressure(system1, T2, z, cl.FugDewPressure(itmax_newton=2))[0] == approx(pres2, rel=1e-6)
    # not exactly the same results, as activity coefficients are ultimately an approximation of the real helmholtz function.
    assert cl.dew_pressure(system1, T2, z, cl.ActivityDewPressure())[0] == approx(pres2, rel=1e-3)
    assert cl.dew_pressure(system1, T2, z, cl.ActivityDewPressure(x0=np.array([0.1, 0.9])))[0] == approx(pres2, rel=1e-3)
    assert cl.dew_pressure(system1, T2, z, cl.ActivityDewPressure(p0=1.5e6))[0] == approx(pres2, rel=1e-3)
    assert cl.dew_pressure(system1, T2, z, cl.ActivityDewPressure(p0=1.5e6, x0=np.array([0.1, 0.9])))[0] == approx(pres2, rel=1e-3)
    
    # dew temperature
    assert cl.dew_temperature(system1, p2, z, cl.ChemPotDewTemperature())[0] == approx(Tres2, rel=1e-6)
    assert cl.dew_temperature(system1, p2, z, cl.ChemPotDewTemperature(x0=np.array([0.1, 0.9])))[0] == approx(Tres2, rel=1e-6)
    assert cl.dew_temperature(system1, p2, z, cl.ChemPotDewTemperature(T0=450))[0] == approx(Tres2, rel=1e-6)
    assert cl.dew_temperature(system1, p2, z, cl.ChemPotDewTemperature(T0=450, x0=np.array([0.1, 0.9])))[0] == approx(Tres2, rel=1e-6)
    
    assert cl.dew_temperature(system1, p2, z, cl.FugDewTemperature())[0] == approx(Tres2, rel=1e-6)
    assert cl.dew_temperature(system1, p2, z, cl.FugDewTemperature(x0=np.array([0.1, 0.9])))[0] == approx(Tres2, rel=1e-6)
    assert cl.dew_temperature(system1, p2, z, cl.FugDewTemperature(T0=450))[0] == approx(Tres2, rel=1e-6)
    assert cl.dew_temperature(system1, p2, z, cl.FugDewTemperature(T0=450, x0=np.array([0.1, 0.9])))[0] == approx(Tres2, rel=1e-6)
    assert cl.dew_temperature(system1, p2, z, cl.FugDewTemperature(itmax_newton=2))[0] == approx(Tres2, rel=1e-6)
    
    # 413
    fluid413 = cl.cPR(["Propane", "Isopentane"], idealmodel=cl.ReidIdeal)
    p413 = 502277.914581377
    y413 = np.array([0.9261006181335611, 0.07389938186643885])
    method413 = cl.ChemPotDewTemperature(vol0=None, T0=None, x0=None, noncondensables=None, 
                                        f_limit=0.0, atol=1.0e-8, rtol=1.0e-12, max_iters=1000, ss=False)
    T413, _, _, _ = cl.Clapeyron.dew_temperature_impl(fluid413, p413, y413, method413)
    assert T413 == approx(292.1479303719277, rel=1e-6)
    
    # nonvolatiles/noncondensables testing. it also test model splitting
    system2 = cl.PCSAFT(["hexane", "ethanol", "methane", "decane"])
    T = 320.  # K
    p = 1e5  # Pa
    z = np.array([0.25, 0.25, 0.25, 0.25])
    x0 = np.array([0.3, 0.3, 0., 0.4])
    y0 = np.array([0.2, 0.2, 0.6, 0.])
    pres1 = 33653.25605767739
    Tres1 = 349.3673410368543
    pres2 = 112209.1535730352
    Tres2 = 317.58287413031866
    
    # bubble pressure - nonvolatiles
    pa, vla, vva, ya = cl.bubble_pressure(system2, T, x0, cl.FugBubblePressure(y0=y0, p0=1e5, nonvolatiles=["decane"]))
    assert pa == approx(pres1, rel=1e-6)
    assert ya[3] == 0.0
    pb, vlb, vvb, yb = cl.bubble_pressure(system2, T, x0, cl.ChemPotBubblePressure(y0=y0, p0=1e5, nonvolatiles=["decane"]))
    assert pa == approx(pres1, rel=1e-6)
    assert ya[3] == 0.0
    
    # bubble temperature - nonvolatiles
    Ta, vla, vva, ya = cl.bubble_temperature(system2, p, x0, cl.FugBubbleTemperature(y0=y0, T0=T, nonvolatiles=["decane"]))
    assert Ta == approx(Tres1, rel=1e-6)
    assert ya[3] == 0.0
    Tb, vlb, vvb, yb = cl.bubble_temperature(system2, p, x0, cl.ChemPotBubbleTemperature(y0=y0, T0=T, nonvolatiles=["decane"]))
    assert Tb == approx(Tres1, rel=1e-6)
    assert yb[3] == 0.0
    # test if the nonvolatile neq system is being built
    Tc, vlc, vvc, yc = cl.bubble_temperature(system2, p, x0, cl.FugBubbleTemperature(itmax_newton=1, y0=y0, T0=T, nonvolatiles=["decane"]))
    assert isinstance(Tc, (int, float))
    
    # dew pressure - noncondensables
    pa, vla, vva, xa = cl.dew_pressure(system2, T, y0, cl.FugDewPressure(noncondensables=["methane"], p0=p, x0=x0))
    assert pa == approx(pres2, rel=1e-6)
    assert xa[2] == 0.0
    pb, vlb, vvb, xb = cl.dew_pressure(system2, T, y0, cl.ChemPotDewPressure(noncondensables=["methane"], p0=p, x0=x0))
    assert pb == approx(pres2, rel=1e-6)
    assert xa[2] == 0.0
    
    # dew temperature - noncondensables
    Ta, vla, vva, xa = cl.dew_temperature(system2, p, y0, cl.FugDewTemperature(noncondensables=["methane"], T0=T, x0=x0))
    assert Ta == approx(Tres2, rel=1e-6)
    assert xa[2] == 0.0
    Tb, vlb, vvb, xb = cl.dew_temperature(system2, p, y0, cl.ChemPotDewTemperature(noncondensables=["methane"], T0=T, x0=x0))
    assert Tb == approx(Tres2, rel=1e-6)
    assert xa[2] == 0.0
