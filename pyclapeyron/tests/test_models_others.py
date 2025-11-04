import pyclapeyron as cl
import numpy as np
from pytest import approx
import gc
import pytest
from juliacall import Main as jl


def test_wilson():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = np.array([0.5, 0.5])
    system = cl.Wilson(["methanol", "benzene"])
    assert cl.activity_coefficient(system, p, T, z)[0] == approx(1.530046633499114, rel=1e-6)

def test_nrtl():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = np.array([0.5, 0.5])
    system = cl.NRTL(["methanol", "benzene"])
    assert cl.activity_coefficient(system, p, T, z)[0] == approx(1.5309354738922405, rel=1e-6)


def test_aspen_nrtl():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = np.array([0.5, 0.5])
    nrtl_vanilla = cl.NRTL(["methanol", "benzene"])
    system = cl.aspenNRTL(["methanol", "benzene"])
    system2 = cl.aspenNRTL(nrtl_vanilla)
    assert cl.activity_coefficient(system, p, T, z)[0] == approx(1.5309354738922405, rel=1e-6)
    assert cl.activity_coefficient(system2, p, T, z)[0] == approx(1.5309354738922405, rel=1e-6)


def test_uniquac():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = np.array([0.5, 0.5])
    system = cl.UNIQUAC(["methanol", "benzene"])
    assert cl.activity_coefficient(system, p, T, z)[0] == approx(1.3630421218486388, rel=1e-6)


def test_unifac():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = np.array([0.5, 0.5])
    system = cl.UNIFAC(["methanol", "benzene"])
    assert cl.activity_coefficient(system, p, T, z)[0] == approx(1.5322232657797463, rel=1e-6)


def test_unifac2():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = np.array([0.5, 0.5])
    system = cl.UNIFAC2([
        ("acetaldehyde", {"CH3": 1, "HCO": 1}),
        ("acetonitrile", {"CH3CN": 1}),
    ], puremodel=cl.BasicIdeal())
    assert np.log(cl.activity_coefficient(system, float('nan'), 323.15, np.array([0.5, 0.5]))[0]) == approx(0.029527741236233, rel=1e-6)


def test_og_unifac():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = np.array([0.5, 0.5])
    system = cl.ogUNIFAC(["methanol", "benzene"])
    assert cl.activity_coefficient(system, p, T, z)[0] == approx(1.5133696314734384, rel=1e-6)


def test_og_unifac2():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = np.array([0.5, 0.5])
    system = cl.ogUNIFAC2([
        ("R22", {"HCCLF2": 1}),
        ("carbon disulfide", {"CS2": 1}),
    ], puremodel=cl.BasicIdeal())
    assert np.log(cl.activity_coefficient(system, float('nan'), 298.15, np.array([0.3693, 0.6307]))[0]) == approx(0.613323250984226, rel=1e-6)


def test_unifac_fv():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = np.array([0.5, 0.5])
    system = cl.UNIFACFV(["PMMA", "PS"])
    assert cl.activity_coefficient(system, p, T, z)[0] == approx(8.63962025235759, rel=1e-6)


def test_unifac_fv_poly():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = np.array([0.5, 0.5])
    system = cl.UNIFACFVPoly(["PMMA", "PS"])
    assert cl.activity_coefficient(system, p, T, z)[0] == approx(4.8275769947121985, rel=1e-6)


def test_fh():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = np.array([0.5, 0.5])
    system = cl.FH(["PMMA", "PS"], np.array([100, 100]))
    assert cl.activity_coefficient(system, p, T, z)[0] == approx(1.8265799707907238, rel=1e-6)


def test_cosmosac02():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = np.array([0.5, 0.5])
    system = cl.COSMOSAC02(["water", "ethanol"])
    assert cl.activity_coefficient(system, p, T, z)[0] == approx(1.3871817962565904, rel=1e-6)
    assert cl.Clapeyron.excess_gibbs_free_energy(system, p, T, z) == approx(610.5706657776052, rel=1e-6)


def test_cosmosac10():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = np.array([0.5, 0.5])
    system = cl.COSMOSAC10(["water", "ethanol"])
    assert cl.activity_coefficient(system, p, T, z)[0] == approx(1.4015660588643404, rel=1e-6)


def test_cosmosac_dsp():
    T = 333.15
    V = 1e-3
    p = 1e5
    z = np.array([0.5, 0.5])
    system = cl.COSMOSACdsp(["water", "ethanol"])
    assert cl.activity_coefficient(system, p, T, z)[0] == approx(1.4398951117248127, rel=1e-6)


def test_joback():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    system = cl.JobackIdeal(["hexane"])
    assert cl.Clapeyron.VT_isobaric_heat_capacity(system, V, 298.15) == approx(143.22076150138616, rel=1e-6)
    assert cl.crit_pure(system)[0] == approx(500.2728274871347, rel=1e-6)
    assert cl.Clapeyron.a_ideal(system, V, T, z) == approx(9.210841420941021, rel=1e-6)
    assert cl.Clapeyron.ideal_consistency(system, V, T, z) == approx(0.0, abs=1e-14)
    assert cl.mass_density(system, p, T, z) == approx(cl.Clapeyron.molecular_weight(system, z) * p / (cl.Rgas(system) * T))
    s0 = cl.JobackIdeal("acetone")
    assert cl.JobackGC.T_c(s0)[0] == approx(500.5590, rel=1e-6)
    assert cl.JobackGC.P_c(s0)[0] == approx(48.025e5, rel=1e-6)
    assert cl.JobackGC.V_c(s0)[0] == approx(209.5e-6, rel=1e-6)
    # assert cl.JobackGC.T_b(s0)[0] == approx(322.1100, rel=1e-6)   #TODO overwritten by PythonCall (`T! -> T_b`)
    assert cl.JobackGC.H_form(s0)[0] == approx(-217830.0, rel=1e-6)
    assert cl.JobackGC.G_form(s0)[0] == approx(-154540.0, rel=1e-6)
    assert cl.JobackGC.C_p(s0, 300)[0] == approx(75.3264, rel=1e-6)
    assert cl.JobackGC.H_fusion(s0)[0] == approx(5.1250e3, rel=1e-6)
    assert cl.JobackGC.H_vap(s0)[0] == approx(29.0180e3, rel=1e-6)
    assert cl.JobackGC.Visc(s0, 300)[0] == approx(0.0002942, rel=9e-4)


def test_reid():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    system = cl.ReidIdeal(["butane"])
    assert cl.Clapeyron.a_ideal(system, V, T, z) == approx(9.210842104089576, rel=1e-6)
    assert cl.Clapeyron.ideal_consistency(system, V, T, z) == approx(0.0, abs=1e-14)
    assert cl.mass_density(system, p, T, z) == approx(cl.Clapeyron.molecular_weight(system, z) * p / (cl.Rgas(system) * T))


def test_shomate():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    system = cl.ShomateIdeal(["water"])
    coeff = system.params.coeffs[1]
    assert cl.Clapeyron.evalcoeff(system, coeff, 500) == approx(35.21836175, rel=1e-6)
    assert cl.mass_density(system, p, T, z) == approx(cl.Clapeyron.molecular_weight(system, z) * p / (cl.Rgas(system) * T))

def test_walker():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    system = cl.WalkerIdeal(["hexane"])
    assert cl.Clapeyron.molecular_weight(system) * 1000 == approx(86.21)
    assert cl.Clapeyron.a_ideal(system, V, T, z) == approx(179.51502015696653, rel=1e-6)
    assert cl.Clapeyron.ideal_consistency(system, V, T, z) == approx(0.0, abs=1e-14)
    assert cl.mass_density(system, p, T, z) == approx(cl.Clapeyron.molecular_weight(system, z) * p / (cl.Rgas(system) * T))

def test_monomer():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    system = cl.MonomerIdeal(["hexane"])
    assert cl.Clapeyron.a_ideal(system, V, T, z) == approx(-10.00711774776317, rel=1e-6)
    assert cl.Clapeyron.ideal_consistency(system, V, T, z) == approx(0.0, abs=1e-14)
    assert cl.mass_density(system, p, T, z) == approx(cl.Clapeyron.molecular_weight(system, z) * p / (cl.Rgas(system) * T))


def test_empiric():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    # Empiric Ideal from JSON
    system = cl.EmpiricIdeal(["water"])
    assert cl.Clapeyron.a_ideal(system, V, T, z) == approx(7.932205569922042, rel=1e-6)
    assert cl.Clapeyron.ideal_consistency(system, V, T, z) == approx(0.0, abs=1e-14)
    assert cl.mass_density(system, p, T, z) == approx(cl.Clapeyron.molecular_weight(system, z) * p / (cl.Rgas(system) * T))
    
    # Empiric Ideal from already existing MultiFluid model
    system = cl.Clapeyron.idealmodel(cl.MultiFluid(["water"]))
    assert cl.Clapeyron.a_ideal(system, V, T, z) == approx(7.932205569922042, rel=1e-6)
    assert cl.Clapeyron.ideal_consistency(system, V, T, z) == approx(0.0, abs=1e-14)
    
    # Empiric Ideal from already existing single fluid model
    system = cl.Clapeyron.idealmodel(system.pures[0])
    assert cl.Clapeyron.a_ideal(system, V, T, z) == approx(7.932205569922042, rel=1e-6)
    assert cl.Clapeyron.ideal_consistency(system, V, T, z) == approx(0.0, abs=1e-14)


def test_aly_lee():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    system = cl.AlyLeeIdeal(["methane"])
    pytest.skip("test_broken: a_ideal test is broken")
    # assert cl.Clapeyron.a_ideal(system, V, T, z) == approx(9.239701647126086, rel=1e-6)
    assert cl.Clapeyron.ideal_consistency(system, V, T, z) == approx(0.0, abs=1e-14)
    assert cl.mass_density(system, p, T, z) == approx(cl.Clapeyron.molecular_weight(system, z) * p / (cl.Rgas(system) * T))
    
    # we use the default GERG 2008 parameters for methane, test if the Cp is equal
    system_gerg = cl.Clapeyron.idealmodel(cl.GERG2008(["methane"]))
    Cp_system = cl.Clapeyron.VT_isobaric_heat_capacity(system, V, T, z)
    Cp_gerg = cl.Clapeyron.VT_isobaric_heat_capacity(system_gerg, V, T, z)
    
    assert Cp_system == approx(Cp_gerg, rel=5e-5)


def test_cp_lng_estimation():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    # Mw to obtain γ₀ = 0.708451
    system = cl.CPLNGEstIdeal(["a1"], userlocations={'Mw': np.array([20.5200706797])})
    # test at 324.33 K, paper says Cp = 44.232, but the calculations in the paper seem off
    assert cl.Clapeyron.VT_isobaric_heat_capacity(system, 0.03, 324.33) == approx(44.231, rel=5e-4)


def test_ppds():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    m1 = cl.PPDSIdeal("krypton")
    assert cl.isobaric_heat_capacity(m1, 1, 303.15) / cl.Rgas(m1) == approx(2.5)
    assert cl.mass_density(m1, p, T, z) == approx(cl.Clapeyron.molecular_weight(m1, z) * p / (cl.Rgas(m1) * T))
    mw2 = 32.042
    m2 = cl.PPDSIdeal("methanol")
    # verification point in ref 1, table A.6
    assert cl.isobaric_heat_capacity(m2, 1, 303.15) / mw2 == approx(1.3840, rel=1e-4)


def test_iapws95():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    system = cl.IAPWS95()
    system_ideal = cl.Clapeyron.idealmodel(system)
    assert cl.Clapeyron.a_ideal(system_ideal, V, T, z) == approx(7.9322055699220435, rel=1e-6)
    assert cl.Clapeyron.a_ideal(system, V, T, z) == approx(7.9322055699220435, rel=1e-6)
    assert cl.Clapeyron.a_res(system, V, T, z) == approx(-2.1152889226862166e14, rel=1e-6)
    # because we are in this regime, numerical accuracy suffers. that is why big(V) is used instead.
    # assert cl.Clapeyron.ideal_consistency(system, big(V), T, z) == approx(0.0, abs=1e-14)


def test_propane_ref():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    system = cl.PropaneRef()
    assert cl.Clapeyron.a_ideal(system, V, T, z) == approx(0.6426994942361217, rel=1e-6)
    assert cl.Clapeyron.a_res(system, V, T, z) == approx(-2.436280448227229, rel=1e-6)
    assert cl.Clapeyron.ideal_consistency(system, V, T, z) == approx(0.0, abs=1e-14)


def test_gerg2008():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    system = cl.GERG2008(["water"])
    assert cl.Clapeyron.a_ideal(system, V, T, z) == approx(4.500151936577565, rel=1e-5)
    assert cl.Clapeyron.a_res(system, V, T, z) == approx(-10.122119572808764, rel=1e-5)
    z4 = np.array([0.25, 0.25, 0.25, 0.25])
    system = cl.GERG2008(["water", "carbon dioxide", "hydrogen sulfide", "argon"])
    assert cl.Clapeyron.a_ideal(system, V, T, z4) == approx(3.1136322215343917, rel=1e-5)
    assert cl.Clapeyron.ideal_consistency(system, V, T, z4) == approx(0.0, rel=1e-14)
    assert cl.Clapeyron.a_res(system, V, T, z4) == approx(-1.1706377677539772, rel=1e-5)
    assert cl.Clapeyron.ideal_consistency(system, V, T, z4) == approx(0.0, abs=1e-14)
    
    system_R = cl.GERG2008(["methane", "ethane"], Rgas=8.2)
    assert cl.Rgas(system_R) == 8.2


def test_eos_lng():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    # LNG paper, table 16
    Tx = 150.0
    Vx = 1 / (18002.169)
    zx = np.array([0.6, 0.4])
    system = cl.EOS_LNG(["methane", "butane"])
    dep = cl.Clapeyron.departure_functions(system)
    assert cl.eos(system, Vx, Tx, zx) == approx(-6020.0044, rel=5e-6)


def test_lkp():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    Tx = 150.0
    Vx = 1 / (18002.169)
    zx = np.array([0.6, 0.4])
    system = cl.LKP(["methane", "butane"])
    assert cl.Clapeyron.a_res(system, Vx, Tx, zx) == approx(-6.469596957611441, rel=5e-6)
    system2 = cl.LKPSJT(["methane", "butane"])
    assert cl.Clapeyron.a_res(system2, Vx, Tx, zx) == approx(-5.636923220762173, rel=5e-6)


def test_lj_ref():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    z1 = np.array([1.0])
    system = cl.LJRef(["methane"])
    Tx = 1.051 * cl.Clapeyron.T_scale(system)
    Vx = cl.Clapeyron._v_scale(system) / 0.673
    assert cl.Clapeyron.a_ideal(system, Vx, Tx) == approx(5.704213386278148, rel=1e-6)
    assert cl.Clapeyron.a_res(system, Vx, Tx) == approx(-2.244730279521925, rel=1e-6)
    assert cl.Clapeyron.ideal_consistency(system, Vx, Tx, z1) == approx(0.0, abs=1e-14)


def test_xiang_deiters():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    z1 = np.array([1.0])
    system = cl.XiangDeiters(["water"])
    # equal to cl.Clapeyron.a_ideal(cl.BasicIdeal(["water"]), V, T, z)
    assert cl.Clapeyron.a_ideal(system, V, T, z1) == approx(-0.33605470137749016, rel=1e-6)
    assert cl.Clapeyron.a_res(system, V, T) == approx(-34.16747927719535, rel=1e-6)


def test_multiparameter_misc():
    T = 300.0
    V = 1 / 200
    z2 = np.array([0.5, 0.5])
    z1 = np.array([1.0])
    
    model = cl.MultiFluid(["carbon dioxide", "hydrogen"], verbose=True)
    
    pures = cl.Clapeyron.split_pure_model(model)
    assert cl.Clapeyron.wilson_k_values(model, 1e6, 300.0) == approx([6.738566125478432, 54.26124873240438], rel=1e-3)
    assert cl.Clapeyron.a_res(model, V, T, z2) == approx(-0.005482930754339683, rel=1e-6)
    
    model2 = cl.SingleFluid("ammonia", verbose=True)
    assert cl.Clapeyron.a_res(model2, V, T, z1) == approx(-0.05006143389915488, rel=1e-6)
    
    model5 = cl.SingleFluid("water", Rgas=10.0)
    assert cl.Rgas(model5) == 10.0


def test_spung_srk():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    system = cl.SPUNG(["ethane"])
    assert cl.shape_factors(system, V, T, z)[0] == approx(0.8246924617474896, rel=1e-6)


def test_spung_pcsaft():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    system = cl.SPUNG(["ethane"], cl.PropaneRef(), cl.PCSAFT(["ethane"]), cl.PCSAFT(["propane"]))
    assert cl.shape_factors(system, V, T, z)[0] == approx(0.8090183134644525, rel=1e-6)


def test_sanchez_lacombe_single():
    T = 298.15
    V = 1e-4
    p = 1e5
    z = np.array([1.0])
    system = cl.SanchezLacombe(["carbon dioxide"])
    assert cl.Clapeyron.a_res(system, V, T, z) == approx(-0.9511044462267396, rel=1e-6)


def test_sanchez_lacombe_kij_rule():
    T = 298.15
    V = 1e-4
    p = 1e5
    z2 = np.array([0.5, 0.5])
    system = cl.SanchezLacombe(["carbon dioxide", "benzoic acid"], mixing=cl.SLKRule)
    assert cl.Clapeyron.a_res(system, V, T, z2) == approx(-6.494291842858994, rel=1e-6)


def test_sanchez_lacombe_k0k1l():
    T = 298.15
    V = 1e-4
    p = 1e5
    z2 = np.array([0.5, 0.5])
    system = cl.SanchezLacombe(["carbon dioxide", "benzoic acid"], mixing=cl.SLk0k1lMixingRule)
    assert cl.Clapeyron.a_res(system, V, T, z2) == approx(-5.579621796375229, rel=1e-6)


def test_antoine_sat():
    system = cl.AntoineEqSat(["water"])
    p0 = cl.saturation_pressure(system, 400.01)[0]
    assert p0 == approx(244561.488609, rel=1e-6)
    assert cl.saturation_temperature(system, p0)[0] == approx(400.01, rel=1e-6)


def test_dippr101_sat():
    system = cl.DIPPR101Sat(["water"])
    p0 = cl.saturation_pressure(system, 400.01)[0]
    assert p0 == approx(245338.15099198322, rel=1e-6)
    assert cl.saturation_temperature(system, p0)[0] == approx(400.01, rel=1e-6)


def test_lee_kesler_sat():
    system = cl.LeeKeslerSat(["water"])
    p0 = cl.saturation_pressure(system, 400.01)[0]
    assert p0 == approx(231731.79240876858, rel=1e-6)
    assert cl.saturation_temperature(system, p0)[0] == approx(400.01, rel=1e-6)


def test_costald():
    system = cl.COSTALD(["water"])
    assert cl.volume(system, 1e5, 300.15) == approx(1.8553472145724288e-5, rel=1e-6)
    system2 = cl.COSTALD(["water", "methanol"])
    assert cl.volume(system2, 1e5, 300.15, np.array([0.5, 0.5])) == approx(2.834714146558056e-5, rel=1e-6)
    assert cl.volume(system2, 1e5, 300.15, np.array([1., 0.])) == approx(1.8553472145724288e-5, rel=1e-6)


def test_dippr105_liquid():
    system = cl.DIPPR105Liquid(["water"])
    assert cl.volume(system, 1e5, 300.15) == approx(1.8057164858551208e-5, rel=1e-6)
    system2 = cl.DIPPR105Liquid(["water", "methanol"])
    assert cl.volume(system2, 1e5, 300.15, np.array([0.5, 0.5])) == approx(2.9367802477146567e-5, rel=1e-6)
    assert cl.volume(system2, 1e5, 300.15, np.array([1., 0.])) == approx(1.8057164858551208e-5, rel=1e-6)


def test_rackett_liquid():
    system = cl.RackettLiquid(["water"])
    assert cl.volume(system, 1e5, 300.15) == approx(1.6837207241594103e-5, rel=1e-6)
    system2 = cl.RackettLiquid(["water", "methanol"])
    assert cl.volume(system2, 1e5, 300.15, np.array([0.5, 0.5])) == approx(3.2516352601748416e-5, rel=1e-6)
    assert cl.volume(system2, 1e5, 300.15, np.array([1., 0.])) == approx(1.6837207241594103e-5, rel=1e-6)


def test_yamada_gunn_liquid():
    system = cl.YamadaGunnLiquid(["water"])
    assert cl.volume(system, 1e5, 300.15) == approx(2.0757546189420953e-5, rel=1e-6)
    system2 = cl.YamadaGunnLiquid(["water", "methanol"])
    assert cl.volume(system2, 1e5, 300.15, np.array([0.5, 0.5])) == approx(3.825994563177142e-5, rel=1e-6)
    assert cl.volume(system2, 1e5, 300.15, np.array([1., 0.])) == approx(2.0757546189420953e-5, rel=1e-6)


def test_grenke_elliott_water():
    system = cl.GrenkeElliottWater()
    system_ref = cl.IAPWS95()
    p1, T1 = 611.657, 273.16
    p2, T2 = 101325.0, 273.152519
    for Ti in np.linspace(250.0, 280.0, 30):
        for log10Pi in np.linspace(5, 8, 20):
            Pi = 10**log10Pi
            v = cl.volume(system, Pi, Ti)
            v_ref = cl.volume(system_ref, Pi, Ti, phase='l')
            assert v == approx(v_ref, rel=1e-3)
            assert cl.pressure(system, v, Ti) == approx(cl.pressure(system_ref, v_ref, Ti), rel=1e-3)
        Cpi = cl.mass_isobaric_heat_capacity(system, 101325.0, Ti)
        Cpi_test = cl.Clapeyron.water_cp(system, Ti)
        assert Cpi == approx(Cpi_test, rel=1e-6)


def test_holten_water():
    model = cl.HoltenWater()
    Tc = 228.2
    Rm = 461.523087
    rho0 = 1081.6482
    
    # table 8
    verification_data = [
        [273.15, 0.101325, 999.84229, -0.683042, 5.088499, 4218.3002, 1402.3886, 0.09665472, 0.62120474],
        [235.15, 0.101325, 968.09999, -29.633816, 11.580785, 5997.5632, 1134.5855, 0.25510286, 0.091763676],
        [250, 200, 1090.45677, 3.267768, 3.361311, 3708.3902, 1668.2020, 0.03042927, 0.72377081],
        [200, 400, 1185.02800, 6.716009, 2.567237, 3338.525, 1899.3294, 0.00717008, 1.1553965],
        [250, 400, 1151.71517, 4.929927, 2.277029, 3757.2144, 2015.8782, 0.00535884, 1.4345145],
    ]
    
    for i in range(5):
        T = verification_data[i][0]
        p = verification_data[i][1] * 1e6
        assert cl.mass_density(model, p, T) == approx(verification_data[i][2], rel=1e-6)
        assert cl.isobaric_expansivity(model, p, T) * 1e4 == approx(verification_data[i][3], rel=1e-6)
        assert cl.isothermal_compressibility(model, p, T) * 1e10 == approx(verification_data[i][4], rel=1e-6)
        assert cl.mass_isobaric_heat_capacity(model, p, T) == approx(verification_data[i][5], rel=1e-6)
        assert cl.speed_of_sound(model, p, T) == approx(verification_data[i][6], rel=1e-6)
        
        p_scaled = p / (Rm * Tc * rho0)
        L = cl.Clapeyron.water_L(model, p_scaled, T)
        omega = 2 + 0.5212269 * p_scaled
        xe = cl.Clapeyron.water_x_frac(model, L, omega)
        assert xe == approx(verification_data[i][7], rel=1e-6)
        assert L == approx(verification_data[i][8], rel=1e-6)


def test_abbott_virial():
    system = cl.AbbottVirial(["methane", "ethane"])
    assert cl.volume(system, 1e5, 300, np.array([0.5, 0.5])) == approx(0.024820060368027988, rel=1e-6)
    assert cl.Clapeyron.a_res(system, 0.05, 300, np.array([0.5, 0.5])) == approx(-0.0024543543773067693, rel=1e-6)


def test_tsonopoulos_virial():
    system = cl.TsonopoulosVirial(["methane", "ethane"])
    assert cl.volume(system, 1e5, 300, np.array([0.5, 0.5])) == approx(0.02485310667780686, rel=1e-6)
    assert cl.Clapeyron.a_res(system, 0.05, 300, np.array([0.5, 0.5])) == approx(-0.0017990881811592349, rel=1e-6)


def test_eos_virial2():
    cub = cl.PR(["methane", "ethane"])
    system = cl.EoSVirial2(cub)
    # exact equality here, as cubics have an exact second virial coefficient
    assert cl.volume(system, 1e5, 300, np.array([0.5, 0.5])) == cl.Clapeyron.volume_virial(cub, 1e5, 300, np.array([0.5, 0.5]))
    assert cl.Clapeyron.a_res(system, 0.05, 300, np.array([0.5, 0.5])) == approx(-0.0023728381262076137, rel=1e-6)


def test_solid_hfus():
    model = cl.SolidHfus(["water"])
    assert cl.chemical_potential(model, 1e5, 298.15, np.array([1.]))[0] == approx(549.1488193300384, rel=1e-6)


def test_solid_ks():
    model = cl.SolidKs(["water"])
    assert cl.chemical_potential(model, 1e5, 298.15, np.array([1.]))[0] == approx(549.1488193300384, rel=1e-6)


def test_iapws06():
    model = cl.IAPWS06()
    # table 6 of Ice-Rev2009 document
    p1, T1 = 611.657, 273.16
    p2, T2 = 101325.0, 273.152519
    p3, T3 = 100e6, 100.0
    Mw = cl.Clapeyron.molecular_weight(model)
    assert cl.mass_gibbs_free_energy(model, p1, T1) == approx(0.611784135, rel=1e-6)
    assert cl.mass_gibbs_free_energy(model, p2, T2) == approx(0.10134274069e3, rel=1e-6)
    assert cl.mass_gibbs_free_energy(model, p3, T3) == approx(-0.222296513088e6, rel=1e-6)
    assert cl.volume(model, p1, T1) / Mw == approx(0.109085812737e-2, rel=1e-6)
    assert cl.volume(model, p2, T2) / Mw == approx(0.109084388214e-2, rel=1e-6)
    assert cl.volume(model, p3, T3) / Mw == approx(0.106193389260e-2, rel=1e-6)
    assert cl.mass_isobaric_heat_capacity(model, p1, T1) == approx(0.209678431622e4, rel=1e-6)
    assert cl.mass_isobaric_heat_capacity(model, p2, T2) == approx(0.209671391024e4, rel=1e-6)
    assert cl.mass_isobaric_heat_capacity(model, p3, T3) == approx(0.866333195517e3, rel=1e-6)
    assert cl.isentropic_compressibility(model, p1, T1) == approx(0.114161597779e-9, rel=1e-6)
    assert cl.isentropic_compressibility(model, p2, T2) == approx(0.114154442556e-9, rel=1e-6)
    assert cl.isentropic_compressibility(model, p3, T3) == approx(0.886060982687e-10, rel=1e-6)
    assert cl.isothermal_compressibility(model, p1, T1) == approx(0.117793449348e-9, rel=1e-6)
    assert cl.isothermal_compressibility(model, p2, T2) == approx(0.117785291765e-9, rel=1e-6)
    assert cl.isothermal_compressibility(model, p3, T3) == approx(0.886880048115e-10, rel=1e-6)


def test_jager_span_solid_co2():
    model = cl.JagerSpanSolidCO2()
    # table 5 of 10.1021/je2011677
    p1, T1 = 0.51795e6, 216.592
    p2, T2 = 100e6, 100.0
    assert cl.gibbs_free_energy(model, p1, T1) == approx(-1.447007522e3, rel=1e-6)
    assert cl.gibbs_free_energy(model, p2, T2) == approx(-2.961795962e3, rel=1e-6)
    assert cl.volume(model, p1, T1) == approx(2.848595255e-5, rel=1e-6)
    assert cl.volume(model, p2, T2) == approx(2.614596591e-5, rel=1e-6)
    assert cl.entropy(model, p1, T1) == approx(-1.803247012e1, rel=1e-6)
    assert cl.entropy(model, p2, T2) == approx(-5.623154438e1, rel=1e-6)
    assert cl.isobaric_heat_capacity(model, p1, T1) == approx(5.913420271e1, rel=1e-6)
    assert cl.isobaric_heat_capacity(model, p2, T2) == approx(3.911045710e1, rel=1e-6)
    assert cl.isobaric_expansivity(model, p1, T1) == approx(8.127788321e-4, rel=1e-6)
    assert cl.isobaric_expansivity(model, p2, T2) == approx(3.843376525e-4, rel=1e-6)
    assert cl.isothermal_compressibility(model, p1, T1) == approx(2.813585169e-10, rel=1e-6)
    assert cl.isothermal_compressibility(model, p2, T2) == approx(1.149061787e-10, rel=1e-6)
