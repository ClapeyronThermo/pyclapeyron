import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np
from juliacall import Main as jl

# @testset "Reported errors" begin

# @testset "#104" begin
def test_issue_104():
    model = cl.VTPR(["carbon dioxide"])
    p = 1e5
    T = 273.15
    assert cl.fugacity_coefficient(model, p, T)[0] == approx(0.9928244080356565, rel=1e-6)
    assert cl.activity_coefficient(model, p, T)[0] == approx(1.0)

# @testset "#112" begin
def test_issue_112():
    model = cl.CPA(["methanol"])
    assert cl.crit_pure(model)[0] == approx(538.2329369300235, rel=1e-6)

# @testset "DM - SAFTgammaMie 1" begin
def test_DM_SAFTgammaMie_1():
    # this constructor was failing on Clapeyron, 3.10-dev
    model = cl.SAFTgammaMie(["water", "ethyl acetate"])
    assocparam = model.vrmodel.params.bondvol
    assert list(assocparam.sites[0]) == ["H2O/H", "H2O/e1"]
    assert list(assocparam.sites[1]) == ["COO/e1"]
    
    # a symmetric matrix was generated from non-symmetric GC values, Clapeyron 0.6.4
    model_mix = cl.SAFTgammaMie(
        [("MEA", {"NH2": 1}), ("Carbon Dioxide", {"CO2": 1})],
        userlocations=jl.Dict({
            "Mw": np.array([16.02285, 44.01]),
            "epsilon": np.array([[284.78, 134.58],
                                 [134.58, 207.89]]),
            "sigma": np.array([3.2477, 3.05]),
            "lambda_a": np.array([6, 5.055]),
            "lambda_r": np.array([[10.354, 50.06],
                                  [50.06, 26.408]]),
            "vst": np.array([1, 2]),
            "S": np.array([0.79675, 0.84680]),
            "n_H": np.array([2, 0]),
            "n_e": np.array([1, 0]),
            "n_a1": np.array([0, 1]),
            "n_a2": np.array([0, 1]),
            "epsilon_assoc": jl.concretize({
                (("NH2", "H"), ("NH2", "e")): 1070.80,
                (("CO2", "a1"), ("NH2", "e")): 3313,
                (("CO2", "a2"), ("NH2", "e")): 4943.6
            }),
            "bondvol": jl.concretize({
                (("NH2", "H"), ("NH2", "e")): 95.225e-30,
                (("CO2", "a1"), ("NH2", "e")): 3280.3e-30,
                (("CO2", "a2"), ("NH2", "e")): 142.64e-30
            })
        })
    )
    
    bondvol_mixed = model_mix.vrmodel.params.bondvol
    co2 = "Carbon Dioxide"
    mea = "MEA"
    # normal
    assert bondvol_mixed[(co2, "CO2/a1"), (mea, "NH2/e")] == 3280.3e-30
    assert bondvol_mixed[(co2, "CO2/a2"), (mea, "NH2/e")] == 142.64e-30
    # reverse
    assert bondvol_mixed[(mea, "NH2/e"), (co2, "CO2/a1")] == 3280.3e-30
    assert bondvol_mixed[(mea, "NH2/e"), (co2, "CO2/a2")] == 142.64e-30
    
    # switch sites: this should result in zeros:
    assert bondvol_mixed[(co2, "CO2/e"), (mea, "NH2/a1")] == 0
    assert bondvol_mixed[(mea, "CO2/e"), (co2, "NH2/a2")] == 0
    assert bondvol_mixed[(mea, "NH2/a1"), (co2, "CO2/e")] == 0
    assert bondvol_mixed[(co2, "NH2/a2"), (mea, "CO2/e")] == 0

# @testset "#140" begin
def test_issue_140():
    # FractionVector with length(x) == 0.
    model = cl.PCSAFT(["water", "carbon dioxide"])
    res = cl.bubble_pressure(model, 280, cl.Clapeyron.FractionVector(0.01), 
                             cl.ChemPotBubblePressure(nonvolatiles=["water"]))
    assert res[0] == approx(4.0772545187410433e6, rel=1e-6)

# @testset "#145" begin
def test_issue_145():
    # incorrect indexing of gc_to_comp_sites when there is more than one assoc site.
    like_data = """Clapeyron Database File
SAFTgammaMie Like Parameters [csvtype = like,grouptype = SAFTgammaMie]
species,vst,S,lambda_r,lambda_a,sigma,epsilon,n_H,n_e1,n_e2,Mw
CH2_PEO,1,1,12,6,4,300,0,0,0,100
cO_1sit,1,1,12,6,4,300,0,1,0,100
"""
    
    mw_data = """Clapeyron Database File
SAFTgammaMie Like Parameters
species,Mw
CH2_PEO,100
cO_1sit,100
"""
    
    assoc_data = """Clapeyron Database File,,,,,,
SAFTgammaMie Assoc Parameters [csvtype = assoc,grouptype = SAFTgammaMie]
species1,site1,species2,site2,epsilon_assoc,bondvol,source
H2O,H,cO_1sit,e1,2193.2,5e-29,
CH2OH,H,cO_1sit,e1,1572.5,4.331e-28,
"""
    
    group_data = """Clapeyron Database File,
SAFTgammaMie Groups [csvtype = groups,grouptype = SAFTgammaMie]
species,groups
PEG_1sit,"[""CH2_PEO"" => 2000,""cO_1sit"" => 1000,""CH2OH"" => 2]"
"""
    
    model = cl.SAFTgammaMie(["water", "PEG_1sit"], 
                           userlocations=jl.Array[jl.String]([like_data, assoc_data, mw_data]),
                           group_userlocations=[group_data])
    # in this case, because the groups are 1 to 1 correspondence to each molecule, 
    # the amount of groups should be the same
    assert len(model.vrmodel.params.epsilon_assoc.values.values) == len(model.params.epsilon_assoc.values.values)
    # test if we got the number of sites right
    assert model.vrmodel.sites.n_sites[1][0] == 1000  # 1000 sites cO_1sit/e1 in PEG.

# @testset "#162" begin
def test_issue_162():
    # a longstanding problem, init_model didn't worked with functions.
    model1 = cl.Wilson(["water", "ethanol"], puremodel=cl.SRK)
    assert jl.isa(model1, cl.EoSModel)
    
    # this case is just for compatibility with the notebooks that were originally released.
    model2 = cl.VTPR(["carbon monoxide", "carbon dioxide"], alpha=cl.BMAlpha)
    assert jl.isa(model2, cl.EoSModel)

# @testset "#171, #366" begin
def test_issue_171_366():
    # #171
    # This is a problem that occurs in an intersection between split_model and cross-association sites
    model = cl.SAFTgammaMie(["water", "acetone"])
    model_split = cl.split_model(model)[1]
    model_pure = cl.SAFTgammaMie(["acetone"])
    res_pure = cl.eos(model_pure, 1.013e6, 298.15)  # works
    res_split = cl.eos(model_split, 1.013e6, 298.15)  # should work
    assert res_pure == approx(res_split)
    
    # # #366 # Internals
    # # incorrect conversion of MixedGCSegmentParam
    # mix_segment_f64 = model.params.mixed_segment
    # mix_segment_bigfloat = cl.convert(cl.MixedGCSegmentParam, mix_segment_f64, dtype=float)
    # assert np.allclose(mix_segment_f64.values.v, mix_segment_bigfloat.values.v)

# @testset "#188" begin
def test_issue_188():
    data = {
        "species": ["A", "B"],
        "Tc": np.array([18.0, 3.5]),
        "Pc": np.array([2.3, 0.1]),
        "Mw": np.array([1.0, 1.0]),
        "acentricfactor": np.array([0.1, 0.3])
    }
    
    file = cl.ParamTable(jl.Symbol("single"), jl.concretize(data), name="db1")
    
    system = cl.PR(["A", "B"], userlocations=jl.Array[jl.String]([file]))
    assert system.params.Tc[2] == 3.5

# @testset "#201" begin
def test_issue_201():
    # Ternary LLE
    if hasattr(cl.aspenNRTL, "puremodel"):
        assert jl.isa(cl.aspenNRTL(["water", "acetone", "dichloromethane"], 
                                       puremodel=cl.PR), cl.EoSModel)
    else:
        assert jl.isa(cl.aspenNRTL(["water", "acetone", "dichloromethane"]), 
                         cl.EoSModel)
    
    assert jl.isa(cl.UNIFAC(["water", "acetone", "dichloromethane"]), 
                     cl.EoSModel)

# @testset "#212" begin
def test_issue_212():
    # Polar PCSAFT - it uses `a` and `b` as site names
    model = cl.PCPSAFT("water")
    assert jl.isa(model, cl.EoSModel)
    assert "a" in model.sites.flattenedsites
    assert "b" in model.sites.flattenedsites

# @testset "#75 - regression" begin
def test_issue_75_regression():
    # pharmaPCSAFT, incorrect mixing rules
    userlocations = {
        "Mw": np.array([352.77, 18.01]),
        "segment": np.array([14.174, 1.2046817736]),
        "sigma": np.array([3.372, 2.7927]),
        "epsilon": np.array([221.261, 353.9449]),
        "n_H": np.array([2, 1]),
        "n_e": np.array([2, 1]),
        "k": np.array([[0, -0.155], [-0.155, 0]]),
        "kT": np.array([[0, 2.641e-4], [2.641e-4, 0]]),
        "epsilon_assoc": jl.concretize({
            (("griseofulvin", "e"), ("griseofulvin", "H")): 1985.49,
            (("water08", "e"), ("water08", "H")): 2425.6714
        }),
        "bondvol": jl.concretize({
            (("griseofulvin", "e"), ("griseofulvin", "H")): 0.02,
            (("water08", "e"), ("water08", "H")): 0.04509
        })
    }
    model = cl.pharmaPCSAFT(["griseofulvin", "water08"], userlocations=userlocations)
    # found while testing, test that we don't mutate the input
    assert np.allclose(userlocations["sigma"], [3.372, 2.7927])
    
    T = 310.15
    p = 1.01325e5
    z = np.array([7.54e-7, 1 - 7.54e-7])
    
    γ1 = cl.activity_coefficient(model, p, T, z)
    assert γ1[0] == approx(51930.06908022231, rel=1e-4)

# @testset "#262" begin [test of internals]
# def test_issue_262(): 
#     # SAFT-gamma Mie
#     # Ensure ij,ab == ji,ba, and that the group-comp is correct for assoc
#     model = cl.SAFTgammaMie(["water", "ethanol"])
#     assert model.params.epsilon_assoc.values[1, 2][1, 2] == model.params.epsilon_assoc.values[2, 1][2, 1]
#     assert model.params.epsilon_assoc.values[1, 2][1, 2] == model.vrmodel.params.epsilon_assoc.values[1, 2][1, 2]

# @testset "SorptionModels.jl - init kij with user" begin
def test_SorptionModels_init_kij():
    def v_star(P_star, T_star):
        return 8.31446261815324 * T_star / P_star / 1000000
    
    def epsilon_star(T_star):
        return 8.31446261815324 * T_star
    
    def r(P_star, T_star, rho_star, mw):
        return mw * (P_star * 1000000) / (8.31446261815324 * T_star * (rho_star / 1e-6))
    
    P_star = [534., 630.]
    T_star = [755., 300.]
    rho_star = [1.275, 1.515]
    mw = [100000, 44.01]
    kij = np.array([[0, -0.0005], [-0.0005, 0]])
    
    model1 = cl.SanchezLacombe(
        ["PC", "CO2"],
        userlocations={
            "vol": np.array([v_star(P_star[i], T_star[i]) for i in range(2)]),
            "segment": np.array([r(P_star[i], T_star[i], rho_star[i], mw[i]) for i in range(2)]),
            "epsilon": np.array([epsilon_star(T_star[i]) for i in range(2)]),
            "Mw": np.array(mw)
        },
        mixing_userlocations={"k0": kij, "k1": np.zeros((2, 2)), "l": np.zeros((2, 2))}
    )
    
    model2 = cl.SanchezLacombe(
        ["PC", "CO2"],
        userlocations={
            "vol": np.array([v_star(P_star[i], T_star[i]) for i in range(2)]),
            "segment": np.array([r(P_star[i], T_star[i], rho_star[i], mw[i]) for i in range(2)]),
            "epsilon": np.array([epsilon_star(T_star[i]) for i in range(2)]),
            "Mw": np.array(mw),
            "k": np.array(kij)
        }
    )
    
    assert np.allclose(cl.get_k(model1)[0], cl.get_k(model2)[0])

# @testset "https://github.com/ClapeyronThermo/Clapeyron.jl/discussions/239" begin
def test_discussion_239():
    # test for easier initialization of CPA/SAFT without association
    m1 = cl.CPA(["Methanol"])
    m2 = cl.CPA(["Methanol"], userlocations={
        "a": np.array([m1.params.a.values[0,0]]),
        "b": np.array([m1.params.b.values[0,0]]),
        "c1": m1.params.c1.values,
        "Mw": m1.params.Mw.values,
        "Tc": m1.params.Tc.values,
        "Pc": m1.cubicmodel.params.Pc.values,
        "n_H": np.array([1]),
        "n_e": np.array([1]),
        "epsilon_assoc": jl.concretize({(("Methanol", "H"), ("Methanol", "e")): m1.params.epsilon_assoc.values.values[0]}),
        "bondvol": jl.concretize({(("Methanol", "H"), ("Methanol", "e")): m1.params.bondvol.values.values[0]})
    })
    assert cl.volume(m1, 1e5, 333.0) == approx(cl.volume(m2, 1e5, 333.0))
    
    m3 = cl.PCSAFT("water", userlocations={"segment": 1, "Mw": 1, "epsilon": 1, "sigma": 1.0})
    assert len(m3.params.bondvol.values.values) == 0

# @testset "infinite dilution derivatives - association" begin
def test_infinite_dilution_derivatives_association():
    # reported by viviviena in slack
    cp_params = {
        "a": np.array([36.54206320678348, 39.19437678197436, 25.7415, 34.91747774761048]),
        "b": np.array([-0.03480434051958945, -0.05808483585041852, 0.2355, -0.014935581577635826]),
        "c": np.array([0.000116818199785053, 0.0003501220208504329, 0.0001578, 0.000756101594841365]),
        "d": np.array([-1.3003819534791665e-7, -3.6941157412454843e-7, -4.0939e-7, -1.0894144551347726e-6]),
        "e": np.array([5.2547403746728466e-11, 1.276270011886522e-10, 2.1166e-10, 4.896983427747592e-10])
    }
    idealmodel = cl.ReidIdeal(["water", "methanol", "propyleneglycol", "methyloxirane"], 
                             userlocations=cp_params)
    pcpsaft = cl.PCPSAFT(["water", "methanol", "propyleneglycol", "methyloxirane"], 
                        idealmodel=idealmodel)
    water = cl.split_model(pcpsaft)[0]
    z = np.array([1.0, 0.0, 0.0, 0.0])
    v = cl.volume(pcpsaft, 101325.0, 298.15, z, phase="liquid")
    cp_pure = cl.VT.isobaric_heat_capacity(water, v, 298.15, 1.0)
    cp_mix = cl.VT.isobaric_heat_capacity(pcpsaft, v, 298.15, z)
    assert cp_mix == approx(cp_pure)
    assert cp_mix == approx(69.21259493306137)

# @testset "CPA init" begin
def test_CPA_init():
    model1 = cl.CPA(["C3", "C4"], userlocations={
        "Mw": np.array([44.097, 58.124]),
        "Tc": np.array([369.83, 425.18]),
        "a":  np.array([9.11875, 13.14274]),
        "b":  np.array([0.057834, 0.072081]),
        "c1": np.array([0.6307, 0.70771]),
        "epsilon_assoc": None,
        "bondvol": None
    })
    assert jl.isa(model1, cl.CPA)

# @testset "#357 - electrolyte equilibria" begin
def test_issue_357_electrolyte_equilibria():
    model1 = cl.ePCSAFT(["water"], ["calcium", "chloride"])
    salts1 = (("calcium chloride", [("calcium", 1), ("chloride", 2)]),)
    x1 = cl.molality_to_composition(model1, salts1, 1.0)
    bub_test = 374.7581484748338
    bubP1_test = 2971.744917038001
    
    bubT1 = cl.bubble_temperature(model1, 101325, x1, 
                                  cl.FugBubbleTemperature(nonvolatiles=["calcium", "chloride"]))[0]
    assert bubT1 == approx(bub_test, rel=1e-6)
    
    bubT1_chempot = cl.bubble_temperature(model1, 101325, x1, 
                                         cl.ChemPotBubbleTemperature(T0=373.0, 
                                                                    nonvolatiles=["calcium", "chloride"]))[0]
    # @test_broken - skipped in Python
    
    bubT1_chempot2 = cl.bubble_temperature(model1, 101325, x1, 
                                          cl.ChemPotBubbleTemperature(nonvolatiles=["calcium", "chloride"]))[0]
    # @test_broken - skipped in Python
    
    bubT1_2 = cl.bubble_temperature(model1, 101325, x1, 
                                   cl.FugBubbleTemperature(nonvolatiles=["calcium", "chloride"], 
                                                          y0=[1., 0., 0.], 
                                                          vol0=(1.8e-5, 1.), 
                                                          T0=373.15))[0]
    assert bubT1_2 == approx(bub_test, rel=1e-6)
    
    bubP1 = cl.bubble_pressure(model1, 298.15, x1, 
                              cl.FugBubblePressure(nonvolatiles=["calcium", "chloride"], 
                                                  y0=[1., 0., 0.], 
                                                  vol0=(1e-5, 1.)))[0]
    assert bubP1 == approx(bubP1_test, rel=1e-6)
    
    bubP1_2 = cl.bubble_pressure(model1, 298.15, x1, 
                                cl.FugBubblePressure(nonvolatiles=["calcium", "chloride"]))[0]
    assert bubP1_2 == approx(bubP1_test, rel=1e-6)

# @testset "416" begin
def test_issue_416():
    model = cl.PR(["nitrogen"], userlocations={
        "Tc": jl.Array[jl.Float64]([0.3]),
        "Pc": jl.Array[jl.Float64]([1.]),
        "Mw": jl.Array[jl.Float64]([1.])
    })
    assert not model.alpha.params.acentricfactor.ismissingvalues[0]
