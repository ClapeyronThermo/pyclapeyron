import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np
from juliacall import Main as jl

# @testset "pharmaPCSAFT, single components" begin
def test_pharmaPCSAFT_singlecomp():
    system = cl.pharmaPCSAFT(["water"])
    v1 = cl.saturation_pressure(system, 280.15)[1]
    v2 = cl.saturation_pressure(system, 278.15)[1]
    v3 = cl.saturation_pressure(system, 275.15)[1]

    assert v1 == approx(1.8022929328333385e-5, rel=1e-6)
    assert v2 == approx(1.8022662044726256e-5, rel=1e-6)
    assert v3 == approx(1.802442451376152e-5,  rel=1e-6)
    #density maxima of water
    assert v2 < v1
    assert v2 < v3

    #issue 377
    mod_phsft = cl.pharmaPCSAFT(["water", "hydrogen"])
    p_c = 3e6
    T = 343.15
    y = np.array([1.955278169111263e-5, 0.9999804472183089])
    assert cl.volume(mod_phsft,p_c,T,y,phase="v") == approx(0.0009700986016167609, rel=1e-6)

# @testset "LJSAFT methods, single components" begin
def test_LJSAFT_singlecomp():
    system = cl.LJSAFT(["ethanol"])
    p = 1e5
    T = 298.15
    # @testset "Bulk properties" begin
    assert cl.volume(system, p, T) == approx(5.8990680856791996e-5, rel=1e-6)
    
    # @testset "VLE properties" begin
    assert cl.saturation_pressure(system, T)[0] == approx(7933.046853495474, rel=1e-6)
    assert cl.crit_pure(system)[0] == approx(533.4350720160273, rel=1e-6)

# @testset "softSAFT methods, single components" begin
def test_softSAFT_singlecomp():
    system = cl.softSAFT(["ethanol"])
    solid_system = cl.solidsoftSAFT("octane")
    p = 1e5
    T = 273.15 + 78.24
    # @testset "Bulk properties" begin
    assert cl.volume(system, p, T, phase="v") == approx(0.027368884099868623, rel=1e-6)
    #volume(SAFTgammaMie(["ethanol"]),p,T,phase =:l)  =6.120507339375205e-5
    assert cl.volume(system, p, T, phase="l") == approx(6.245903786961202e-5, rel=1e-6)
    assert cl.volume(solid_system, 2.3e9, 298.15, phase="s") == approx(9.961905037894007e-5, rel=1e-6)
    
    # @testset "VLE properties" begin
    assert cl.saturation_pressure(system, T)[0] == approx(101341.9709136089, rel=1e-6)
    assert cl.crit_pure(system)[0] == approx(540.1347889779657, rel=1e-6)

# @testset "BACKSAFT methods, single components" begin
def test_BACKSAFT_singlecomp():
    system = cl.BACKSAFT(["decane"])
    p = 1e5
    T = 298.15
    # @testset "Bulk properties" begin
    #0.0001950647173402879 with SAFTgammaMie
    assert cl.volume(system, p, T) == approx(0.00019299766073894634, rel=1e-6)
    
    # @testset "VLE properties" begin
    assert cl.saturation_pressure(system, T)[0] == approx(167.8313793818096, rel=1e-6)
    assert cl.crit_pure(system)[0] == approx(618.8455740197799, rel=1e-6)

# @testset "CPA methods, single components" begin
def test_CPA_singlecomp():
    system = cl.CPA(["ethanol"])
    p = 1e5
    T = 298.15
    # @testset "Bulk properties" begin
    assert cl.volume(system, p, T) == approx(5.913050998953597e-5, rel=1e-6)
    # @test volume(CPA("water"), 1e5u"Pa", 303.15u"K") ≈ 1.7915123921401366e-5u"m^3" rtol = 1e-6
    #TODO skipping the unit test - needs proper unit handling in Python
    
    # @testset "VLE properties" begin
    assert cl.saturation_pressure(system, T)[0] == approx(7923.883649594267, rel=1e-6)
    assert cl.crit_pure(system)[0] == approx(539.218257256262, rel=1e-6)

# @testset "SAFT-γ Mie methods, single components" begin
def test_SAFTgammaMie_singlecomp():
    system = cl.SAFTgammaMie(["ethanol"])
    p = 1e5
    T = 298.15
    # @testset "Bulk properties" begin
    assert cl.volume(system, p, T) == approx(5.753982584153832e-5, rel=1e-6)
    assert cl.Clapeyron.molecular_weight(system)*1000 == approx(46.065)
    
    # @testset "VLE properties" begin
    assert cl.saturation_pressure(system, T)[0] == approx(7714.8637084302, rel=1e-5)
    assert cl.crit_pure(system)[0] == approx(521.963002384691, rel=1e-5)

# @testset "SAFT-VR Mie methods, single components" begin
def test_SAFTVRMie_singlecomp():
    system = cl.SAFTVRMie(["methanol"])
    p = 1e5
    T = 298.15
    # @testset "Bulk properties" begin
    assert cl.volume(system, p, T) == approx(4.064466003321247e-5, rel=1e-6)
    
    # @testset "VLE properties" begin
    assert cl.saturation_pressure(system, T)[0] == approx(16957.59261579083, rel=1e-6)
    assert cl.crit_pure(system)[0] == approx(519.5443602179253, rel=1e-5)

# @testset "SAFT-VRQ Mie methods, single component" begin
def test_SAFTVRQMie_singlecomp():
    system = cl.SAFTVRQMie(["helium"])
    # @testset "VLE properties" begin
    assert cl.saturation_pressure(system, 4)[0] == approx(56761.2986265459, rel=1e-6)

# @testset "sCKSAFT methods, single component" begin
def test_sCKSAFT_singlecomp():
    system = cl.sCKSAFT(["ethane"])
    tc_test, pc_test, vc_test = (321.00584034360014, 6.206975436514129e6, 0.0001515067748592245)
    tc, pc, vc = cl.crit_pure(system)
    assert tc == approx(tc_test, rel=1e-3)
    assert pc == approx(pc_test, rel=1e-3)
    assert vc == approx(vc_test, rel=1e-3)
    system2 = cl.CKSAFT("methanol")
    assert cl.crit_pure(system2)[0] == approx(544.4777700204786)

# @testset "ideal model parsing to JSON" begin
def test_ideal_model_parsing():
    comp = ["hexane"]
    idealmodels = []
    idealmodels.append(cl.BasicIdeal(comp))
    idealmodels.append(cl.MonomerIdeal(comp))
    idealmodels.append(cl.JobackIdeal(comp))
    idealmodels.append(cl.WalkerIdeal(comp))
    idealmodels.append(cl.AlyLeeIdeal(comp))

    T0 = 400.15
    V0 = 0.03
    for mi in idealmodels:
        mxd = cl.XiangDeiters(comp, idealmodel=mi)
        id_mxd = cl.Clapeyron.idealmodel(mxd)
       
        #parsed and reconstituted idealmodel
        cp1 = cl.Clapeyron.VT_isobaric_heat_capacity(id_mxd, V0, T0)
        #original
        a1 = cl.Clapeyron.a_ideal(id_mxd, V0, T0, np.array([1.0]))
        a2 = cl.Clapeyron.a_ideal(mi, V0, T0, np.array([1.0]))
        cp2 = cl.Clapeyron.VT_isobaric_heat_capacity(mi, V0, T0)
        if isinstance(mi, type(cl.AlyLeeIdeal(comp))):
            # @test_broken a1 ≈ a2 rtol = 1e-6
            # @test_broken cp1 ≈ cp2 rtol = 1e-6
            pass  # Skip broken tests for AlyLeeIdeal
        else:
            assert a1 == approx(a2, rel=1e-6)
            assert cp1 == approx(cp2, rel=1e-6)

# @testset "SAFT-VRQ Mie methods, multicomponent" begin
def test_SAFTVRQMie_multicomp():
    system = cl.SAFTVRQMie(["hydrogen", "neon"])
    T = -125 + 273.15
    #brewer 1969 data for H2-Ne
    #Texp = [50,25,0,-25,-50,-75,-100,-125] .+ 273.15
    #B12exp = [14.81,14.23,13.69,13.03,12.29,11.23,9.98,8.20]
    #there is a difference between brewer 1969 data and the exact value, but for some reason, their plots use a very thick linewidth...
    assert cl.equivol_cross_second_virial(system, T)*1e6 == approx(8.09, rel=1e-1)

# @testset "RK, single component" begin
def test_RK_singlecomp():
    system = cl.RK(["ethane"])
    p = 1e7
    p2 = 1e5
    T = 250.15
    # @testset "Bulk properties" begin
    assert cl.volume(system, p, T) == approx(6.819297582048736e-5, rel=1e-6)
    assert cl.volume(system, p2, T) == approx(0.020539807199804024, rel=1e-6)
    assert cl.speed_of_sound(system, p, T) == approx(800.288303407983, rel=1e-6)
    
    # @testset "VLE properties" begin
    psat, _, _ = cl.saturation_pressure(system, T)
    assert psat == approx(1.409820798879772e6, rel=1e-6)
    assert cl.saturation_pressure(system, T, cl.SuperAncSaturation())[0] == approx(psat, rel=1e-6)
    assert cl.crit_pure(system)[0] == approx(305.31999999999994, rel=1e-6)
    assert cl.Clapeyron.wilson_k_values(system, p, T) == approx(np.array([0.13840091523637849]), rel=1e-6)

# GC.gc()
# @testset "Patel-Teja, single component" begin
def test_PatelTeja_singlecomp():
    system = cl.PatelTeja(["water"])
    p = 1e5
    T = 550.15
    # @testset "Bulk properties" begin
    assert cl.volume(system, p, T) == approx(0.04557254632681239, rel=1e-6)
    
    # @testset "VLE properties" begin
    assert cl.saturation_pressure(system, T)[0] == approx(4.51634223156497e6, rel=1e-6)
    Tc, Pc, Vc = cl.crit_pure(system)
    assert Tc == system.params.Tc.values[0]
    assert Pc == system.params.Pc.values[0]
    assert cl.pressure(system, Vc, Tc) == approx(Pc)

# @testset "Patel-Teja-Valderrama, single component" begin
def test_PTV_singlecomp():
    system = cl.PTV(["water"])
    p = 1e5
    T = 298.15
    # @testset "Bulk properties" begin
    assert cl.volume(system, p, T) == approx(1.9221342043684064e-5, rel=1e-6)
    
    # @testset "VLE properties" begin
    assert cl.saturation_pressure(system, T)[0] == approx(2397.1315826665273, rel=1e-6)
    Tc, Pc, Vc = cl.crit_pure(system)
    assert Tc == system.params.Tc.values[0]
    assert cl.pressure(system, Vc, Tc) == approx(Pc)

# @testset "KU, single component" begin
def test_KU_singlecomp():
    system = cl.KU(["water"])
    p = 1e5
    T = 298.15
    # @testset "Bulk properties" begin
    assert cl.volume(system, p, T) == approx(2.0614093101238483e-5, rel=1e-6)
    
    # @testset "VLE properties" begin
    assert cl.saturation_pressure(system, T)[0] == approx(2574.7348636211996, rel=1e-6)

# @testset "RKPR, single component" begin
def test_RKPR_singlecomp():
    system = cl.RKPR(["methane"])
    vc_vol = cl.volume(system, system.params.Pc[1], system.params.Tc[1])  #vc calculated via cubic_poly
    Tc, Pc, Vc = cl.crit_pure(system)  #vc calculated via cubic_pure_zc
    assert vc_vol == approx(Vc, rel=1e-4)
    assert vc_vol/system.params.Vc[1] == approx(1.168, rel=1e-4)  #if Zc_exp < 0.29, this should hold, by definition
    assert Vc/system.params.Vc[1] == approx(1.168, rel=1e-4)  #if Zc_exp < 0.29, this should hold, by definition

# @testset "EPPR78, single component" begin
def test_EPPR78_singlecomp():
    system = cl.EPPR78(["carbon dioxide"])
    T = 400  # u"K"
    #TODO Skipping unit tests - needs proper unit handling in Python
    # @test Clapeyron.volume(system, 3311.0u"bar", T) ≈ 3.363141761376883e-5u"m^3"
    # @test Clapeyron.molar_density(system, 3363.1u"bar", T) ≈ 29810.09484964839u"mol*m^-3"
    pass  # Placeholder for unit-based tests

# @testset "Cubic methods, multi-components" begin
def test_Cubic_multicomp():
    system = cl.RK(["ethane", "undecane"])
    system2 = cl.tcPR(["benzene", "toluene", "nitrogen"])
    system3 = cl.cPR(["butane", "toluene"], idealmodel=cl.ReidIdeal)
    p = 1e7
    T = 298.15
    z = np.array([0.5, 0.5])
    p2 = 1.5*101325
    T2 = 350
    z2 = np.array([0.001, 0.001, 0.001])

    # @testset "Bulk properties" begin
    assert cl.volume(system, p, T, z) == approx(0.00017378014541520907, rel=1e-6)
    assert cl.speed_of_sound(system, p, T, z) == approx(892.4941848133369, rel=1e-6)
    assert cl.volume(system2, p2, T2, z2, phase="l") == approx(2.851643999862116e-7, rel=1e-6)
    assert cl.volume(system2, p2, T2, z2 / np.sum(z2), phase="l") == approx(2.851643999862116e-7/np.sum(z2), rel=1e-6)
    
    # @testset "VLE properties" begin
    assert cl.bubble_pressure(system, T, z)[0] == approx(1.5760730143760687e6, rel=1e-6)
    assert cl.crit_mix(system, z)[0] == approx(575.622237585033, rel=1e-6)
    # assert cl.mechanical_critical_point(system, z)[0] == approx(483.08783138464617, rel=1e-6) #TODO from current Master branch
    # assert cl.spinodal_maximum(system, z)[0] == approx(578.7715447554762, rel=1e-6) #TODO from current Master branch
    srksystem = cl.SRK(["ethane", "undecane"])
    assert cl.Clapeyron.wilson_k_values(srksystem, p, T) == approx(np.array([0.4208525854463047, 1.6171551943938252e-5]), rel=1e-6)
    #test scaling of crit_mix
    cm1 = cl.crit_mix(system3, np.array([0.5, 0.5]))
    cm2 = cl.crit_mix(system3, np.array([1.0, 1.0]))
    assert cm1[0] == approx(cm2[0])
    assert 2*cm1[2] == approx(cm2[2])

# GC.gc()
# @testset "Activity methods, pure components" begin
def test_Activity_purecomp():
    # Python doesn't have hasfield, we'll use the simpler approach
    # if hasfield(Wilson,:puremodel)
    #     system = Wilson(["methanol"])
    # else:
    system = cl.CompositeModel(["methanol"], liquid=cl.Wilson, fluid=cl.PR)
    
    p = 1e5
    T = 298.15
    # @testset "Bulk properties" begin
    assert cl.volume(system, p, T) == approx(4.7367867309516085e-5, rel=1e-6)
    assert cl.speed_of_sound(system, p, T) == approx(2136.222735675237, rel=1e-6)
    
    # @testset "VLE properties" begin
    assert cl.crit_pure(system)[0] == approx(512.6400000000001, rel=1e-6)
    assert cl.saturation_pressure(system, T)[0] == approx(15525.93630485447, rel=1e-6)

# @testset "Activity methods, multi-components" begin
def test_Activity_multicomp():
    com = cl.CompositeModel(["water", "methanol"], liquid=cl.DIPPR105Liquid, saturation=cl.DIPPR101Sat, gas=cl.PR)
    
    system = cl.Wilson(["methanol", "benzene"])
    comp_system = cl.CompositeModel(["methanol", "benzene"], fluid=cl.PR, liquid=cl.Wilson, reference_state=jl.Symbol("ashrae")) #TODO

    # Python doesn't have hasfield, we'll use the simpler approach
    system2 = cl.CompositeModel(["water", "methanol"], liquid=cl.Wilson, fluid=com)
    system3 = cl.CompositeModel(["octane", "heptane"], liquid=cl.UNIFAC, fluid=cl.LeeKeslerSat)

    com1 = cl.split_model(com)[0]
    p = 1e5
    T = 298.15
    T2 = 320.15
    z = np.array([0.5, 0.5])
    z_bulk = np.array([0.2, 0.8])
    T3 = 300.15
    z3 = np.array([0.9, 0.1])
    
    # @testset "Bulk properties" begin
    assert cl.crit_pure(com1)[0] == approx(647.13)
    assert cl.volume(system, p, T, z_bulk) == approx(7.967897222918716e-5, rel=1e-6)
    assert cl.volume(comp_system, p, T, z_bulk) == approx(7.967897222918716e-5, rel=1e-6)
    assert cl.speed_of_sound(system, p, T, z_bulk) == approx(1551.9683977722198, rel=1e-6)
    assert cl.speed_of_sound(comp_system, p, T, z_bulk) == approx(1551.9683977722198, rel=1e-6)
    assert cl.mixing(system, p, T, z_bulk, cl.Clapeyron.gibbs_free_energy) == approx(-356.86007792929263, rel=1e-6)
    assert cl.mixing(system, p, T, z_bulk, cl.Clapeyron.enthalpy) == approx(519.0920708672975, rel=1e-6)
    #test that we are actually considering the reference state, even in the vapour phase.
    assert (cl.enthalpy(comp_system, p, T, z_bulk, phase="v") - cl.enthalpy(system, p, T, z_bulk, phase="v") 
            == approx(np.sum(cl.reference_state(comp_system).a0 * z_bulk), rel=1e-6))
    
    # @testset "VLE properties" begin
    assert cl.gibbs_solvation(system, T) == approx(-24707.145697543132, rel=1e-6)
    assert cl.bubble_pressure(system, T, z)[0] == approx(23758.58099358788, rel=1e-6)
    assert cl.bubble_pressure(system, T, z, cl.ActivityBubblePressure(gas_fug=True, poynting=True))[0] == approx(23839.554959977086)
    assert cl.bubble_pressure(system, T, z, cl.ActivityBubblePressure(gas_fug=True, poynting=False))[0] == approx(23833.324475723246)
    assert cl.bubble_pressure(system3, T3, z3)[0] == approx(2460.897944633704, rel=1e-6)
    assert cl.bubble_temperature(system, 23758.58099358788, z)[0] == approx(T, rel=1e-6)

    assert cl.dew_pressure(system2, T2, z)[0] == approx(19386.939256733036, rel=1e-6)
    assert cl.dew_pressure(system2, T2, z, cl.ActivityDewPressure(gas_fug=True, poynting=True))[0] == approx(19393.924550078184, rel=1e-6)
    assert cl.dew_pressure(system2, T2, z, cl.ActivityDewPressure(gas_fug=True, poynting=False))[0] == approx(19393.76058757084, rel=1e-6)
    assert cl.dew_temperature(system2, 19386.939256733036, z)[0] == approx(T2, rel=1e-6)

    # @testset "LLE" begin
    model3 = cl.NRTL(["methanol", "hexane"])
    x1, x2 = cl.LLE(model3, 290.0)
    assert x1[0] == approx(0.15878439462531743, rel=1e-6)

# GC.gc()
# @testset "GERG2008 methods, single components" begin
def test_GERG2008_singlecomp():
    system = cl.GERG2008(["water"])
    met = cl.GERG2008(["methane"])
    p = 1e5
    T = 298.15
    # @testset "Bulk properties" begin
    assert cl.volume(system, p, T) == approx(1.8067969591040684e-5, rel=1e-6)
    assert cl.speed_of_sound(system, p, T) == approx(1484.0034692716843, rel=1e-6)
    #EOS-LNG, table 15
    V1, T1 = 1/27406.6102, 100.0
    assert cl.pressure(met, V1, T1) == approx(1.0e6, rel=2e-6)
    assert cl.VT.speed_of_sound(met, V1, T1) == approx(1464.5158, rel=1e-6)
    assert cl.pressure(met, 1/28000, 140) == approx(86.944725e6, rel=2e-6)
    
    # @testset "VLE properties" begin
    assert cl.saturation_pressure(system, T)[0] == approx(3184.83242429761, rel=1e-5)
    assert cl.saturation_pressure(system, T, cl.IsoFugacitySaturation())[0] == approx(3184.83242429761, rel=1e-5)
    assert cl.crit_pure(system)[0] == approx(647.0960000000457, rel=1e-6)

# @testset "GERG2008 methods, multi-components" begin
def test_GERG2008_multicomp():
    # @testset "Bulk properties" begin
    model = cl.GERG2008(["nitrogen", "methane", "ethane", "propane", "butane", "isobutane", "pentane"])
    lng_composition = np.array([0.93, 92.1, 4.64, 1.7, 0.42, 0.32, 0.09])
    lng_composition_molar_fractions = lng_composition / np.sum(lng_composition)
    assert cl.molar_density(model, (380.5+101.3)*1000.0, -153.0+273.15, lng_composition_molar_fractions)/1000 == approx(24.98, rel=1e-2)
    assert cl.mass_density(model, (380.5+101.3)*1000.0, -153.0+273.15, lng_composition_molar_fractions) == approx(440.73, rel=1e-2)
    #TODO Skipping unit tests - needs proper unit handling in Python
    
    #test found in #371
    model2 = cl.GERG2008(["carbon dioxide", "nitrogen", "water"])
    assert cl.mass_density(model2, 64.0e5, 30+273.15, np.array([0.4975080785711593, 0.0049838428576813995, 0.4975080785711593]), phase="l") == approx(835.3971524715569, rel=1e-6)
    
    #test found in #395:
    p395 = np.logspace(np.log10(3.12e6), np.log10(1e8), 1000)
    model395 = cl.GERG2008(["carbon dioxide", "nitrogen"])
    z395 = np.array([0.95, 0.05])
    v395 = [cl.volume(model395, p, 250.0, z395, phase="v") for p in p395]
    assert np.sum(np.isnan(v395)) == 999

    # @testset "VLE properties" begin
    system = cl.GERG2008(["carbon dioxide", "water"])
    T = 298.15
    z = np.array([0.8, 0.2])
    assert cl.bubble_pressure(system, T, z)[0] == approx(5.853909891112583e6, rel=1e-5)

# @testset "EOS-LNG methods, multi-components" begin
def test_EOS_LNG_multicomp():
    # @testset "Bulk properties" begin
    #EOS-LNG paper, table 16
    system = cl.EOS_LNG(["methane", "isobutane"])
    z = np.array([0.6, 0.4])
    T1, V1 = 160.0, 1/17241.868
    T2, V2 = 350.0, 1/100

    assert cl.VT.speed_of_sound(system, V1, T1, z) == approx(1331.9880, rel=1e-6)
    assert cl.VT.speed_of_sound(system, V2, T2, z) == approx(314.72845, rel=1e-6)
    assert cl.volume(system, 5e6, T1, z) == approx(V1)
    assert cl.pressure(system, V2, T2, z) == approx(0.28707693e6, rel=1e6)

# @testset "IAPWS95 methods" begin
def test_IAPWS95():
    system = cl.IAPWS95()
    p = 1e5
    T = 298.15
    T_v = 380.15
    T_c = 750.
    p_c = 250e5
    mw = cl.Clapeyron.molecular_weight(system)
    # @testset "Bulk properties" begin
    #IAPWS-2018, table 7
    assert cl.mass_density(system, 0.992418352e5, 300.0) == approx(996.556, rel=1e-6)
    assert cl.mass_density(system, 0.200022515e8, 300.0) == approx(1005.308, rel=1e-6)
    assert cl.mass_density(system, 0.700004704e9, 300.0) == approx(1188.202, rel=1e-6)
    # test_volume(system,0.992418352e5,300.0)  # Helper function not available in Python
    # test_volume(system,0.200022515e8,300.0)
    # test_volume(system,0.700004704e9,300.0)
    assert cl.entropy(system, 0.992418352e5, 300.0) == approx(mw*393.062643, rel=1e-6)
    assert cl.entropy(system, 0.200022515e8, 300.0) == approx(mw*387.405401, rel=1e-6)
    assert cl.entropy(system, 0.700004704e9, 300.0) == approx(mw*132.609616, rel=1e-6)
    assert cl.speed_of_sound(system, 0.992418352e5, 300.0) == approx(1501.51914, rel=1e-6)
    assert cl.speed_of_sound(system, 0.200022515e8, 300.0) == approx(1534.92501, rel=1e-6)
    assert cl.speed_of_sound(system, 0.700004704e9, 300.0) == approx(2443.57992, rel=1e-6)
    #below triple point/ liquid
    # test_volume(system,1e6,265.0,phase="l")
    # test_volume(system,1e8,265.0,phase="l")

    assert cl.Clapeyron.molecular_weight(cl.Clapeyron.idealmodel(system)) == mw
    
    # @testset "VLE properties" begin
    assert cl.saturation_pressure(system, T)[0] == approx(3169.9293388718283, rel=1e-6)
    assert cl.saturation_pressure(system, T, cl.IsoFugacitySaturation())[0] == approx(3169.9293388718283, rel=1e-6)
    #saturation temperature tests are noisy
    assert cl.saturation_temperature(system, 3169.9293390134403)[0] == approx(298.1499999999789, rel=1e-6)
    tc, pc, vc = cl.crit_pure(system)
    assert tc == approx(647.096, rel=1e-5)
    v2 = cl.volume(system, pc, tc)
    assert cl.pressure(system, v2, tc) == approx(pc, rel=1e-6)

# @testset "PropaneRef methods" begin
def test_PropaneRef():
    system = cl.PropaneRef()
    p = 1e5
    T = 230.15
    # @testset "Bulk properties" begin
    assert cl.volume(system, p, T) == approx(7.577761282115866e-5, rel=1e-6)
    #PropsSI("D","P",1e4,"T",230.15,"propane") == 0.23126803007122876
    assert cl.mass_density(system, 1e4, T) == approx(0.23126803007122876, rel=1e-6)
    assert cl.speed_of_sound(system, p, T) == approx(1166.6704395959607, rel=1e-6)
    
    # @testset "VLE properties" begin
    ps = 97424.11102296013  #PropsSI("P","T",T,"Q",1,"propane")
    assert cl.saturation_pressure(system, T)[0] == approx(ps, rel=1e-6)
    assert cl.saturation_pressure(system, T, cl.IsoFugacitySaturation())[0] == approx(ps, rel=1e-6)
    assert cl.saturation_temperature(system, ps)[0] == approx(T, rel=1e-6)
    assert cl.crit_pure(system)[0] == approx(369.8900089509652, rel=1e-6)

# @testset "Ammonia2023 methods" begin
def test_Ammonia2023():
    system = cl.Ammonia2023()
    #test for pressures here. we exclude the 0 density (obtained from paper's SI)
    ps = np.array([
       27.63275,
       11.25287,
       0.4270652,
       5.96888,
       58.52754])
    Tρ = [(320.0, 35.0), (405.0, 16.5), (275.0, 0.2), (520.0, 1.5), (620.0, 14.0)]
    for i in range(len(ps)):
        T, rho = Tρ[i]
        assert cl.pressure(system, 0.001/rho, T)*1e-6 == approx(ps[i], rel=1e-6)

# GC.gc()
# @testset "Helmholtz + Activity" begin #TODO CoolProp.jl required
# def test_HelmholtzActivity():
#     model = cl.HelmAct(["water", "ethanol"])
#     p = 12666.0
#     x1 = cl.FractionVector(0.00350)
#     # test_scales(model) #  # Helper function not available in Python
#     assert cl.bubble_temperature(model, p, x1)[3][0] == approx(0.00198, rel=1e-2)

# @testset "SingleFluid - CoolProp" begin #TODO CoolProp.jl required
# def test_SingleFluid_CoolProp():
#     #methanol, uses double exponential term
#     #before, it used the association term, but now no model uses it
#     #TODO PropsSI calls would need CoolProp Python library
#     # @test saturation_pressure(SingleFluid("methanol"),300.15)[0] ≈ PropsSI("P","T",300.15,"Q",1.,"methanol") rtol = 1e-6
    
#     r134 = cl.SingleFluid("r134a")
#     r1342 = cl.MultiFluid("r134a")
#     #TODO The following tests require CoolProp PropsSI function
#     # @test Clapeyron.eos(r134,0.03,373.15,Clapeyron.SA[1.0]) ≈ PropsSI("HELMHOLTZMOLAR","Dmolar",1/0.03,"T",373.15,"R134a")
#     # @test Clapeyron.eos(r1342,0.03,373.15,Clapeyron.SA[1.0]) ≈ PropsSI("HELMHOLTZMOLAR","Dmolar",1/0.03,"T",373.15,"R134a")
#     # @test Clapeyron.a_res(r134,0.03,373.15,Clapeyron.SA[1.0]) ≈ PropsSI("ALPHAR","Dmolar",1/0.03,"T",373.15,"R134a")
#     # @test Clapeyron.a_res(r1342,0.03,373.15,Clapeyron.SA[1.0]) ≈ PropsSI("ALPHAR","Dmolar",1/0.03,"T",373.15,"R134a")

#     #tests send via email
#     #TODO The following tests use test_volume helper function not available in Python
#     fluid1 = cl.SingleFluid("n-Undecane")
#     # test_volume(fluid1,1e-2*fluid1.properties.Pc,0.38*fluid1.properties.Tc)
#     # test_volume(fluid1,3e2*fluid1.properties.Pc,0.38*fluid1.properties.Tc)
#     # test_volume(fluid1,3e2*fluid1.properties.Pc,1.1*fluid1.properties.Tc)

#     fluid2 = cl.SingleFluid("n-Butane")
#     # test_volume(fluid2,1e-2*fluid2.properties.Pc,0.3*fluid2.properties.Tc)
#     # test_volume(fluid2,30*fluid2.properties.Pc,0.3*fluid2.properties.Tc)

#     fluid3 = cl.SingleFluid("water")
#     # test_volume(fluid3,1e-2*fluid3.properties.Pc,0.4*fluid3.properties.Tc)
#     # test_volume(fluid3,40*fluid3.properties.Pc,3.2*fluid3.properties.Tc)

#     fluid4 = cl.SingleFluid("MethylOleate")
#     # test_volume(fluid4,1e-2*fluid4.properties.Pc,0.3*fluid4.properties.Tc)
#     # test_volume(fluid4,4e1*fluid4.properties.Pc,0.3*fluid4.properties.Tc)
#     # test_volume(fluid4,4e1*fluid4.properties.Pc,1.3*fluid4.properties.Tc)

#     fluid5 = cl.SingleFluid("MD3M")
#     # test_volume(fluid5,1e-2*fluid5.properties.Pc,0.3*fluid5.properties.Tc)
#     # test_volume(fluid5,2e2*fluid5.properties.Pc,0.3*fluid5.properties.Tc)
#     # test_volume(fluid5,2e2*fluid5.properties.Pc,1.1*fluid5.properties.Tc)

#     fluid6 = cl.SingleFluid("Toluene")
#     # test_volume(fluid6,1e-2*fluid6.properties.Pc,0.25*fluid6.properties.Tc)
#     # test_volume(fluid6,2e2*fluid6.properties.Pc,0.25*fluid6.properties.Tc)
#     # test_volume(fluid6,2e2*fluid6.properties.Pc,1.2*fluid6.properties.Tc)

#     #CoolProp fluid predicting negative fundamental derivative of gas dynamics
#     #10.1021/acs.iecr.9b00608, figure 17
#     model = cl.SingleFluid("MD4M")
#     TΓmin = 647.72
#     _, _, vv = cl.saturation_pressure(model, TΓmin)
#     Γmin = cl.VT_fundamental_derivative_of_gas_dynamics(model, vv, TΓmin)
#     assert Γmin == approx(-0.2825376983518102, rel=1e-6)

#     #376
#     T_376 = (310.95, 477.95, 644.15)
#     px = [cl.saturation_pressure(fluid3, T)[0] for T in T_376]
#     p1 = np.arange(px[0], 420e5, 1e4)
#     p2 = np.arange(px[1], 420e5, 1e4)
#     p3 = np.arange(px[2], 420e5, 1e4)
#     v_T37 = [cl.volume(fluid3, p, T_376[0], phase="l") for p in p1]
#     v_T202 = [cl.volume(fluid3, p, T_376[1], phase="l") for p in p2]
#     v_T371 = [cl.volume(fluid3, p, T_376[2], phase="l") for p in p3]
#     assert np.sum(np.isnan(v_T37)) == 0
#     assert np.sum(np.isnan(v_T202)) == 0
#     assert np.sum(np.isnan(v_T371)) == 0

# @testset "LKP methods" begin
def test_LKP():
    system = cl.LKP("propane", idealmodel=cl.AlyLeeIdeal)
    system_mod = cl.LKPmod("squalane", userlocations={"Tc": 810, "Pc": 0.728e6, "acentricfactor": 1.075, "Mw": 1.0})

    p = 1e5
    T = 230.15
    
    # Bulk properties
    assert cl.volume(system, p, T, phase="l") == approx(7.865195401331961e-5, rel=1e-6)
    assert cl.volume(system, p, T, phase="v") == approx(0.018388861273788176, rel=1e-6)
    assert cl.speed_of_sound(system, p, T, phase="l") == approx(1167.2461897307874, rel=1e-6)
    assert cl.molar_density(system_mod, 0.0, 298.15, phase="l") == approx(1721.2987626107251, rel=1e-6)  # 0.1007/s10765-024-03360-0, Figure 4
    
    # VLE properties
    assert cl.saturation_pressure(system, T)[0] == approx(105419.26772976149, rel=1e-6)
    # saturation temperature tests are noisy
    assert cl.saturation_temperature(system, 105419.26772976149)[0] == approx(T, rel=1e-6)
    assert cl.crit_pure(system)[0] == approx(369.83, rel=1e-6)

def test_LJRef():
    system = cl.LJRef(["methane"])
    T = 1.051 * cl.Clapeyron.T_scale(system)
    p = 0.035 * cl.Clapeyron.p_scale(system)
    v = cl.Clapeyron._v_scale(system) / 0.673
    
    # Bulk properties
    assert cl.volume(system, p, T) == approx(v, rel=1e-5)
    
    # VLE properties
    assert cl.saturation_pressure(system, T)[0] == approx(p, rel=1e-1)
    assert cl.crit_pure(system)[0] / cl.Clapeyron.T_scale(system) == approx(1.32, rel=1e-4)

def test_SPUNG():
    system = cl.SPUNG(["ethane"])
    p = 1e5
    T = 313.15
    
    # Bulk properties
    # GERG2008(["ethane]): 0.025868956878898026
    vv = 0.025868956878898026
    assert cl.volume(system, p, T) == approx(vv, rel=1e-6)
    assert cl.volume(system, p, T, phase="vapour") == approx(vv, rel=1e-6)

    # GERG2008(["ethane]) : 318 m/s
    assert cl.speed_of_sound(system, p, T) == approx(308.38846317827625, rel=1e-6)

    # VLE properties
    T_sat = 250.15
    assert cl.saturation_pressure(system, T_sat)[0] == approx(1.3085074415334722e6, rel=1e-6)
    # Critical point of ethane: 305.322
    assert cl.crit_pure(system)[0] == approx(305.37187249327553, rel=1e-6)

def test_lattice():
    p = 1e5
    T = 298.15
    T1 = 301.15
    system = cl.SanchezLacombe(["carbon dioxide"])
    
    # Bulk properties
    assert cl.volume(system, p, T1) == approx(0.02492944175392707, rel=1e-6)
    assert cl.speed_of_sound(system, p, T1) == approx(307.7871016597499, rel=1e-6)
    
    # VLE properties
    assert cl.saturation_pressure(system, T)[0] == approx(6.468653945184592e6, rel=1e-6)
    assert cl.crit_pure(system)[0] == approx(304.21081254005446, rel=1e-6)

def test_PeTS():
    system = cl.PeTS(["methane"])
    system.params.sigma.values[0,0] = 1e-10
    system.params.epsilon.values[0,0] = 1
    system.params.epsilon.values[0,0] = 1

    # Values from FeOs notebook example:
    # We can reproduce FeOs values here
    crit = cl.crit_pure(system)
    Tc, Pc, Vc = crit
    assert Tc == approx(1.08905, rel=1e-5)
    assert Vc == approx(1/513383.86, rel=1e-5)

    # first value from saturation pressure(T = 0.64):
    psat, vl, vv = cl.saturation_pressure(system, 0.64)
    assert psat == approx(3.027452e+04, rel=1e-6)
    assert vl == approx(1/1.359958e+06, rel=1e-6)
    assert vv == approx(1/5892.917088, rel=1e-6)

    # uses the default x0_saturation_temperature initial guess
    assert cl.saturation_temperature(system, psat)[0] == approx(0.64, rel=1e-6)

    # uses the default x0_psat initial guess
    assert cl.saturation_pressure(system, 0.64, cl.IsoFugacitySaturation(crit=crit))[0] == approx(3.027452e+04, rel=1e-6)

    T_nearc = 1.084513  # The last value of their critical point is actually above ours.
    psat_nearc = 1.374330e+06
    assert cl.saturation_pressure(system, T_nearc)[0] == approx(psat_nearc, rel=1e-6)
    assert cl.saturation_pressure(system, T_nearc, cl.IsoFugacitySaturation(crit=crit))[0] == approx(psat_nearc, rel=1e-6)