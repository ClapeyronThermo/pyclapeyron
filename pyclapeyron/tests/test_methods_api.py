import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np

# Note: Unitful tests are specific to Julia's unit system and would need
# a Python equivalent like pint. Skipping for now unless specific Python unit library is used.

# @testset "association" begin
def test_association():
    model_no_comb = cl.PCSAFT(["methanol", "ethanol"], 
                              assoc_options=cl.AssocOptions(combining="nocombining"))
    model_cr1 = cl.PCSAFT(["methanol", "ethanol"], 
                         assoc_options=cl.AssocOptions(combining="cr1"))
    model_esd = cl.PCSAFT(["methanol", "ethanol"], 
                         assoc_options=cl.AssocOptions(combining="esd"))
    model_esd_r = cl.PCSAFT(["methanol", "ethanol"], 
                           assoc_options=cl.AssocOptions(combining="elliott_runtime"))
    model_dufal = cl.PCSAFT(["methanol", "ethanol"], 
                           assoc_options=cl.AssocOptions(combining="dufal"))
    
    repr(cl.AssocOptions(combining="dufal"))
    
    V = 5e-5
    T = 298.15
    z = np.array([0.5, 0.5])
    
    assert cl.nonzero_extrema(range(4)) == (1, 3)
    assert cl.a_assoc(model_no_comb, V, T, z) == approx(-4.667036481159167, rel=1e-6)
    assert cl.a_assoc(model_cr1, V, T, z) == approx(-5.323469194263458, rel=1e-6)
    assert cl.a_assoc(model_esd, V, T, z) == approx(-5.323420343872591, rel=1e-6)
    assert cl.a_assoc(model_esd_r, V, T, z) == approx(-5.323430326406561, rel=1e-6)
    assert cl.a_assoc(model_dufal, V, T, z) == approx(-5.323605338112626, rel=1e-6)
    
    # system with strong association:
    fluid = cl.PCSAFT(["water", "methanol"], assoc_options=cl.AssocOptions(combining="elliott"))
    fluid.params.epsilon["water", "methanol"] *= (1 + 0.18)
    v = cl.volume(fluid, 1e5, 160.0, np.array([0.5, 0.5]), phase="l")
    X_result = cl.X(fluid, v, 160.0, np.array([0.5, 0.5])).v
    expected_X = np.array([0.0011693187791158642, 0.0011693187791158818, 
                           0.0002916842981727242, 0.0002916842981727286])
    assert np.allclose(X_result, expected_X, rtol=1e-8)
    
    K = np.array([[0.0, 244.24071691762867, 0.0, 3.432863727098509],
                  [244.24071691762867, 0.0, 3.432863727098509, 0.0],
                  [0.0, 2.2885758180656732, 0.0, 0.0],
                  [2.2885758180656732, 0.0, 0.0, 0.0]])
    expected_solve = np.array([0.0562461981664357, 0.0562461981664357, 
                               0.8859564211875895, 0.8859564211875895])
    assert np.allclose(cl.assoc_matrix_solve(K), expected_solve)

# @testset "tpd" begin
def test_tpd():
    system = cl.PCSAFT(["water", "cyclohexane"])
    T = 298.15
    p = 1e5
    phases, tpds, symz, symw = cl.tpd(system, p, T, np.array([0.5, 0.5]))
    assert tpds[0] == approx(-0.6081399681963373, rel=1e-6)
    
    act_system = cl.UNIFAC(["water", "cyclohexane"])
    phases2, tpds2, symz2, symw2 = cl.tpd(act_system, p, T, np.array([0.5, 0.5]), lle=True)
    assert tpds2[0] == approx(-0.9412151812640561, rel=1e-6)
    
    model = cl.PR(["IsoButane", "n-Butane", "n-Pentane", "n-Hexane"])
    z = np.array([0.25, 0.25, 0.25, 0.25])
    p = 1e5
    v1 = cl.volume(model, p, 297.23, z)
    assert not cl.isstable(model, p, 297.23, z)
    assert not cl.VT_isstable(model, v1, 297.23, z)
    
    model = cl.cPR(["ethane"], idealmodel=cl.ReidIdeal)
    p = 101325
    z = np.array([5.0])
    T, vl, vv = cl.saturation_temperature(model, p)
    v_unstable = np.exp(0.5 * (np.log(vl) + np.log(vv)))
    V = cl.volume(model, p, T, z)  # lies in range of Vv
    assert cl.VT_diffusive_stability(model, V, T, z)
    assert not cl.VT_diffusive_stability(model, v_unstable, T, z)

# @testset "reference states" begin
def test_reference_states():
    assert not cl.has_reference_state(cl.PCSAFT("water"))
    assert cl.has_reference_state(cl.PCSAFT("water", idealmodel=cl.ReidIdeal))
    
    ref1 = cl.ReferenceState("nbp")
    ref2 = cl.ReferenceState("ashrae")
    ref3 = cl.ReferenceState("iir")
    ref4 = cl.ReferenceState("volume", T0=298.15, P0=1.0e5, phase="liquid")
    ref5 = cl.ReferenceState("volume", T0=298.15, P0=1.0e5, phase="gas", 
                            z0=np.array([0.4, 0.6]), H0=123, S0=456)

def test_reference_states_nbp():
    ref1 = cl.ReferenceState("nbp")
    model1 = cl.PCSAFT(["water", "pentane"], idealmodel=cl.ReidIdeal, reference_state=ref1)
    pure1 = cl.split_model(model1)
    T11, v11, _ = cl.saturation_temperature(pure1[0], 101325.0)
    assert cl.VT_enthalpy(pure1[0], v11, T11) == approx(0.0, abs=1e-6)
    assert cl.VT_entropy(pure1[0], v11, T11) == approx(0.0, abs=1e-6)
    T12, v12, _ = cl.saturation_temperature(pure1[1], 101325.0)
    assert cl.VT_enthalpy(pure1[1], v12, T12) == approx(0.0, abs=1e-6)
    assert cl.VT_entropy(pure1[1], v12, T12) == approx(0.0, abs=1e-6)
    repr(cl.reference_state(model1))
    
    # test that multifluids work
    model1b = cl.GERG2008("water", reference_state="nbp")
    T1b, v1b, _ = cl.saturation_temperature(model1b, 101325.0)
    assert cl.VT_enthalpy(model1b, v1b, T1b) == approx(0.0, abs=1e-6)
    assert cl.VT_entropy(model1b, v1b, T1b) == approx(0.0, abs=1e-6)

def test_reference_states_ashrae():
    ref2 = cl.ReferenceState("ashrae")
    model2 = cl.PCSAFT(["water", "pentane"], idealmodel=cl.ReidIdeal, reference_state=ref2)
    pure2 = cl.split_model(model2)
    T_ashrae = 233.15
    _, v21, _ = cl.saturation_pressure(pure2[0], T_ashrae)
    assert cl.VT_enthalpy(pure2[0], v21, T_ashrae) == approx(0.0, abs=1e-6)
    assert cl.VT_entropy(pure2[0], v21, T_ashrae) == approx(0.0, abs=1e-6)
    _, v22, _ = cl.saturation_pressure(pure2[1], T_ashrae)
    assert cl.VT_enthalpy(pure2[1], v22, T_ashrae) == approx(0.0, abs=1e-6)
    assert cl.VT_entropy(pure2[1], v22, T_ashrae) == approx(0.0, abs=1e-6)

def test_reference_states_iir():
    ref3 = cl.ReferenceState("iir")
    model3 = cl.PCSAFT(["water", "pentane"], idealmodel=cl.ReidIdeal, reference_state=ref3)
    pure3 = cl.split_model(model3)
    Tiir = 273.15
    H31, H32 = 200 * model3.params.Mw[0], 200 * model3.params.Mw[1]
    S31, S32 = 1.0 * model3.params.Mw[0], 1.0 * model3.params.Mw[1]
    _, v31, _ = cl.saturation_pressure(pure3[0], Tiir)
    assert cl.VT_enthalpy(pure3[0], v31, Tiir) == approx(H31, abs=1e-6)
    assert cl.VT_entropy(pure3[0], v31, Tiir) == approx(S31, abs=1e-6)
    _, v32, _ = cl.saturation_pressure(pure3[1], Tiir)
    assert cl.VT_enthalpy(pure3[1], v32, Tiir) == approx(H32, abs=1e-6)
    assert cl.VT_entropy(pure3[1], v32, Tiir) == approx(S32, abs=1e-6)

def test_reference_states_custom_volume():
    ref4 = cl.ReferenceState("volume", T0=298.15, P0=1.0e5, phase="liquid")
    model4 = cl.PCSAFT(["water", "pentane"], idealmodel=cl.ReidIdeal, reference_state=ref4)
    pure4 = cl.split_model(model4)
    T4, P4 = 298.15, 1.0e5
    v41 = cl.volume(pure4[0], P4, T4, phase="liquid")
    assert cl.VT_enthalpy(pure4[0], v41, T4) == approx(0.0, abs=1e-6)
    assert cl.VT_entropy(pure4[0], v41, T4) == approx(0.0, abs=1e-6)
    v42 = cl.volume(pure4[1], P4, T4, phase="liquid")
    assert cl.VT_enthalpy(pure4[1], v42, T4) == approx(0.0, abs=1e-6)
    assert cl.VT_entropy(pure4[1], v42, T4) == approx(0.0, abs=1e-6)

def test_reference_states_custom_composition():
    ref5 = cl.ReferenceState("volume", T0=298.15, P0=1.0e5, phase="gas", 
                            z0=np.array([0.4, 0.6]), H0=123, S0=456)
    model5 = cl.PCSAFT(["water", "pentane"], idealmodel=cl.ReidIdeal, reference_state=ref5)
    T5, P5 = 298.15, 1.0e5
    z5 = np.array([0.4, 0.6])
    v5 = cl.volume(model5, P5, T5, z5, phase="gas")
    assert cl.VT_enthalpy(model5, v5, T5, z5) == approx(123, abs=1e-6)
    assert cl.VT_entropy(model5, v5, T5, z5) == approx(456, abs=1e-6)

def test_reference_states_vectorparam():
    # reference state from EoSVectorParam
    mod_pr = cl.cPR(["water", "ethanol"], idealmodel=cl.ReidIdeal, reference_state="ntp")
    mod_vec = cl.EoSVectorParam(mod_pr)
    cl.recombine(mod_vec)
    assert cl.reference_state(mod_vec).std_type == "ntp"
    assert len(cl.reference_state(mod_vec).a0) == 2
    
    # reference state from Activity models
    puremodel = cl.cPR(["water", "ethanol"], idealmodel=cl.ReidIdeal)
    act = cl.NRTL(["water", "ethanol"], puremodel=puremodel, reference_state="ntp")
    assert cl.reference_state(act).std_type == "ntp"
    assert len(cl.reference_state(act).a0) == 2

# @testset "Solid Phase Equilibria" begin
def test_Pure_Solid_Liquid_Equilibria():
    model = cl.CompositeModel(["methane"], fluid=cl.SAFTVRMie, solid=cl.SAFTVRSMie)
    
    trp = cl.triple_point(model)
    assert trp[0] == approx(106.01194395351305, rel=1e-6)
    
    sub = cl.sublimation_pressure(model, 100.)
    assert sub[0] == approx(30776.588071307022, rel=1e-6)
    
    mel = cl.melting_pressure(model, 110.)
    assert mel[0] == approx(1.126517131058346e7, rel=1e-6)
    
    sub = cl.sublimation_temperature(model, 1e3)
    assert sub[0] == approx(78.29626523297529, rel=1e-6)
    
    mel = cl.melting_temperature(model, 1e5)
    assert mel[0] == approx(106.02571487518759, rel=1e-6)
    
    model2 = cl.CompositeModel("water", solid=cl.SolidHfus, fluid=cl.IAPWS95())
    assert cl.melting_temperature(model2, 1e5)[0] == approx(273.15, rel=1e-6)
    assert cl.melting_pressure(model2, 273.15)[0] == approx(1e5, rel=1e-6)
    
    # solid gibbs + fluid helmholtz
    model3 = cl.CompositeModel("water", solid=cl.IAPWS06(), fluid=cl.IAPWS95())
    assert cl.melting_temperature(model3, 101325.0)[0] == approx(273.1525192653753, rel=1e-6)
    assert cl.melting_pressure(model3, 273.1525192653753)[0] == approx(101325.0, rel=1e-6)
    
    # solid gibbs + fluid gibbs
    model4 = cl.CompositeModel("water", solid=cl.IAPWS06(), fluid=cl.GrenkeElliottWater())
    assert cl.melting_temperature(model4, 101325.0)[0] == approx(273.15, rel=1e-6)
    assert cl.melting_pressure(model4, 273.15)[0] == approx(101325.0, rel=1e-6)
    
    # solid gibbs + any other fluid helmholtz
    model5 = cl.CompositeModel("water", solid=cl.IAPWS06(), fluid=cl.cPR("water"))
    assert cl.melting_temperature(model5, 101325.0)[0] == approx(273.15, rel=1e-6)
    assert cl.melting_pressure(model5, 273.15)[0] == approx(101325.0, rel=1e-6)
    
    # solid gibbs + helmholtz fluid, without any initial points
    model6 = cl.CompositeModel(["CO2"], solid=cl.JagerSpanSolidCO2(), 
                              fluid=cl.SingleFluid("carbon dioxide"))
    tp6 = cl.triple_point(model6)
    Ttp6 = tp6[0]
    ptp6 = tp6[1]
    assert Ttp6 == approx(model6.fluid.properties.Ttp, rel=1e-5)
    assert cl.sublimation_pressure(model6, Ttp6)[0] == approx(ptp6, rel=1e-6)
    assert cl.sublimation_temperature(model6, ptp6)[0] == approx(Ttp6, rel=1e-6)
    assert cl.melting_pressure(model6, Ttp6)[0] == approx(ptp6, rel=1e-6)
    assert cl.melting_temperature(model6, ptp6)[0] == approx(Ttp6, rel=1e-6)

def test_Mixture_Solid_Liquid_Equilibria():
    model = cl.CompositeModel(
        [("1-decanol", [("CH3", 1), ("CH2", 9), ("OH (P)", 1)]),
         ("thymol", [("ACCH3", 1), ("ACH", 3), ("ACOH", 1), ("ACCH", 1), ("CH3", 2)])],
        liquid=cl.UNIFAC, solid=cl.SolidHfus
    )
    T = 275.
    p = 1e5
    s1 = cl.sle_solubility(model, p, T, np.array([1., 1.]), solute=["1-decanol"])
    s2 = cl.sle_solubility(model, p, T, np.array([1., 1.]), solute=["thymol"])
    assert s1[1] == approx(0.21000625991669147, rel=1e-6)
    assert s2[1] == approx(0.3370264930822045, rel=1e-6)
    
    TE, xE = cl.eutectic_point(model)
    assert TE == approx(271.97967645045804, rel=1e-6)

def test_Solid_Liquid_Liquid_Equilibria():
    model = cl.CompositeModel(
        ["water", "ethanol", 
         ("ibuprofen", [("ACH", 4), ("ACCH2", 1), ("ACCH", 1), ("CH3", 3), ("COOH", 1), ("CH", 1)])],
        liquid=cl.UNIFAC, solid=cl.SolidHfus
    )
    p = 1e5
    T = 323.15
    s1, s2 = cl.slle_solubility(model, p, T)
    assert s1[2] == approx(0.0015804179997257882, rel=1e-6)

# @testset "challenging equilibria" begin
def test_challenging_equilibria_VTPR():
    # carbon monoxide is supercritical
    system = cl.VTPR(["carbon monoxide", "carbon dioxide"])
    # This test is marked as broken in Julia
    # pytest.skip("Known broken test from Julia version")

def test_saturation_points_without_critical_point():
    model1 = cl.PCSAFT("water")
    Tc1, _, _ = cl.crit_pure(model1)
    T1 = 0.999 * Tc1
    assert cl.saturation_pressure(model1, T1, crit_retry=False)[0] == approx(3.6377840330375336e7, rel=1e-6)
    
    model2 = cl.PCSAFT("eicosane")
    Tc2, _, _ = cl.crit_pure(model2)
    T2 = 0.999 * Tc2
    assert cl.saturation_pressure(model2, T2, crit_retry=False)[0] == approx(1.451917823392476e6, rel=1e-6)
    
    # Skip platform-specific tests
    model4 = cl.SAFTVRMie(["methanol"])
    T4 = 164.7095044742657
    assert cl.saturation_pressure(model4, T4, crit_retry=False)[0] == approx(0.02610821545005174, rel=1e-6)

# @testset "partial properties" begin
def test_partial_properties():
    model_pem = cl.PR(["hydrogen", "oxygen", "water"])
    z = np.array([0.1, 0.1, 0.8])
    p, T = 0.95e5, 380.15
    
    for prop in [cl.volume, cl.gibbs_free_energy, cl.helmholtz_free_energy, 
                 cl.entropy, cl.enthalpy, cl.internal_energy]:
        partial_vals = cl.partial_property(model_pem, p, T, z, prop)
        assert np.sum(partial_vals * z) == approx(prop(model_pem, p, T, z))

# @testset "spinodals" begin
def test_spinodals():
    # Example from Ref. https://doi.org/10.1016/j.fluid.2017.04.009
    model = cl.PCSAFT(["methane", "ethane"])
    T_spin = 223.
    x_spin = np.array([0.2, 0.8])
    pl_spin, vl_spin = cl.spinodal_pressure(model, T_spin, x_spin, phase="liquid")
    pv_spin, vv_spin = cl.spinodal_pressure(model, T_spin, x_spin, phase="vapor")
    assert vl_spin == approx(7.218532167482202e-5, rel=1e-6)
    assert vv_spin == approx(0.0004261109817247137, rel=1e-6)
    
    Tl_spin_impl, xl_spin_impl = cl.spinodal_temperature(model, pl_spin, x_spin, T0=220., v0=vl_spin)
    Tv_spin_impl, xv_spin_impl = cl.spinodal_temperature(model, pv_spin, x_spin, T0=225., v0=vv_spin)
    assert Tl_spin_impl == approx(T_spin, rel=1e-6)
    assert Tv_spin_impl == approx(T_spin, rel=1e-6)
    
    # test for #382: pure spinodal at low pressures
    model2 = cl.PCSAFT("carbon dioxide")
    Tc, Pc, Vc = (310.27679925044134, 8.06391600653306e6, 9.976420206333288e-5)
    T = np.linspace(Tc - 70, Tc - 0.1, 50)
    psl = np.array([cl.spinodal_pressure(model2, t, phase="l")[0] for t in T])
    psv = np.array([cl.spinodal_pressure(model2, t, phase="v")[0] for t in T])
    psat = np.array([cl.saturation_pressure(model2, t)[0] for t in T])
    assert np.all(psl < psat)
    assert np.all(psat < psv)
    assert np.all(np.diff(psl) > 0)  # issorted
    assert np.all(np.diff(psv) > 0)  # issorted

# @testset "supercritical lines" begin
def test_supercritical_lines():
    model = cl.PR("methane")
    T_initial = 200.0
    p_widom, v1 = cl.widom_pressure(model, T_initial)
    T_widom, v2 = cl.widom_temperature(model, p_widom)
    assert T_initial == approx(T_widom, rel=1e-6)
    assert v1 == approx(v2, rel=1e-6)
    
    with pytest.raises(Exception):  # ArgumentError
        cl.widom_pressure(model, T_initial, v0=v1, p0=p_widom)
    
    assert T_initial == approx(cl.widom_temperature(model, p_widom, T0=1.01 * T_initial)[0], rel=1e-6)
    assert T_initial == approx(cl.widom_temperature(model, p_widom, v0=1.01 * v1)[0], rel=1e-6)
    assert p_widom == approx(cl.widom_pressure(model, T_initial, p0=1.01 * p_widom)[0], rel=1e-6)
    assert p_widom == approx(cl.widom_pressure(model, T_initial, v0=1.01 * v1)[0], rel=1e-6)
    
    p_ciic, v3 = cl.ciic_pressure(model, T_initial)
    T_ciic, v4 = cl.ciic_temperature(model, p_ciic)
    assert T_initial == approx(T_ciic, rel=1e-6)
    assert v3 == approx(v4, rel=1e-6)
    
    with pytest.raises(Exception):  # ArgumentError
        cl.ciic_pressure(model, T_initial, v0=v3, p0=p_ciic)
    
    assert T_initial == approx(cl.ciic_temperature(model, p_ciic, T0=1.01 * T_initial)[0], rel=1e-6)
    assert T_initial == approx(cl.ciic_temperature(model, p_ciic, v0=1.01 * v3)[0], rel=1e-6)
    assert p_ciic == approx(cl.ciic_pressure(model, T_initial, p0=1.01 * p_ciic)[0], rel=1e-6)
    assert p_ciic == approx(cl.ciic_pressure(model, T_initial, v0=1.01 * v3)[0], rel=1e-6)
