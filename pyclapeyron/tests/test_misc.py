import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np

# @testset "misc" begin

model2 = None
model4 = None
gc3 = None
gc2 = None
ideal1 = None
noparam1 = None
simple1 = None
model_structgc = None

def setup_module():
    global model2, model4, gc3, gc2, ideal1, noparam1, simple1, model_structgc
    model2 = cl.PCSAFT(["water", "ethanol"])
    model4 = cl.SAFTgammaMie(["methane", "butane", "isobutane", "pentane"])
    gc3 = cl.UNIFAC(["propane", "butane", "isobutane"])
    gc2 = cl.SAFTgammaMie([
        "ethanol",
        ("ibuprofen", [("CH3", 3), ("COOH", 1), ("aCCH", 1), ("aCCH2", 1), ("aCH", 4)])
    ])
    ideal1 = cl.WalkerIdeal(["hexane"])
    noparam1 = gc3.puremodel[0].translation
    simple1 = gc3.puremodel[0].alpha
    model_structgc = cl.structSAFTgammaMie(["ethanol", "octane"])

# @testset "split_model" begin
def test_split_model():
    models2 = cl.split_model(model2)
    
    with pytest.raises(Exception):  # ArgumentError
        cl.split_model(noparam1)
    
    with pytest.raises(Exception):  # MethodError
        cl.split_model(model2, None)
    
    assert models2[0].components[0] == model2.components[0]
    assert models2[1].components[0] == model2.components[1]
    
    model2_unsplit = cl.split_model(model2, [[1, 2]])[0]
    assert model2_unsplit.components == model2.components
    
    model4_split = cl.split_model(model4)
    assert model4_split[3].groups.n_groups[0][2] == 3
    assert model4_split[0].components[0] == "methane"
    assert all(len(model4_split[i]) == 1 for i in range(4))
    
    gc3_split = cl.split_model(gc3)
    assert all(len(gc3_split[i]) == 1 for i in range(3))
    assert all(len(gc3_split[i].puremodel) == 1 for i in range(3))
    
    structgc_split = cl.split_model(model_structgc)
    assert np.array_equal(structgc_split[0].groups.n_intergroups[0], np.array([[0, 1], [1, 0]]))
    assert np.array_equal(structgc_split[1].groups.n_intergroups[0], np.array([[0, 2], [2, 5]]))
    
    noparam1_split = cl.split_model(noparam1, list(range(5)))
    assert len(noparam1_split) == 5
    assert noparam1_split[0] == noparam1
    
    # splitting parameters
    assert cl.split_model(model2.params.segment)[0][0] == model2.params.segment[0]
    assert cl.split_model(model2.params.sigma)[0][0, 0] == model2.params.sigma[0, 0]
    
    # from notebooks, #173
    nb_test = cl.SAFTgammaMie(["methane", "nitrogen", "carbon dioxide", "ethane", "propane", 
                              "butane", "isobutane", "pentane", "isopentane", "hexane", 
                              "heptane", "octane"])
    assert len(cl.split_model(nb_test)) == 12
    
    # weird error found on splitting groups
    model0 = cl.SAFTgammaMie(["ethane"])
    model0_split = cl.split_model(cl.SAFTgammaMie(["methane", "ethane"]))[-1]
    assert model0.params.epsilon.values[0, 0] == model0_split.params.epsilon.values[0, 0]
    
    # error on spliting assoc models without sites
    model_nosites = cl.PCSAFT(["a"], userlocations={"Mw": 1.0, "segment": 1.0, "sigma": 1.0, "epsilon": 1.0})
    assert isinstance(cl.split_model(model_nosites)[0], cl.PCSAFT)
    
    # error on splitting models with ReferenceState
    model_reference_state = cl.JobackIdeal(["propane", "hexane"])
    assert isinstance(cl.split_model(model_reference_state)[0], cl.JobackIdeal)
    
    # index reduction testing
    model_idx = cl.PCSAFT(["ethane", "propane", "methane"])
    assert len(cl.index_reduction(model_idx, np.array([1., 0., 0.]))[0]) == 1
    assert len(cl.index_reduction(model_idx, np.array([1., 0., 1.]))[0]) == 2
    assert len(cl.index_reduction(model_idx, np.array([1., 1., 1.]))[0]) == 3
    assert len(cl.index_reduction(model_idx, np.array([1., -5e-16, 1.]))[0]) == 2
    
    with pytest.raises(Exception):  # ErrorException
        cl.index_reduction(model_idx, np.array([0., -5e-16, 0.]))
    with pytest.raises(Exception):  # ErrorException
        cl.index_reduction(model_idx, np.array([0., 0.0, 0.]))
    with pytest.raises(Exception):  # BoundsError
        cl.index_reduction(model_idx, np.array([0., -5e-16, 0., 0.0]))
    with pytest.raises(Exception):  # BoundsError
        cl.index_reduction(model_idx, np.array([0., -5e-16]))
    with pytest.raises(Exception):  # BoundsError
        cl.index_reduction(model_idx, np.array([True, True]))
    with pytest.raises(Exception):  # BoundsError
        cl.index_reduction(model_idx, np.array([True, True, True, True]))
    with pytest.raises(Exception):  # ErrorException
        cl.index_reduction(model_idx, np.array([False, False, False]))
    with pytest.raises(Exception):  # ErrorException
        cl.index_reduction(model_idx, np.array([0., 0.0, 0.]))

# @testset "export_model" begin
def test_export_model_SAFT():
    model_og = cl.PCSAFT(["water", "ethanol"])
    cl.export_model(model_og)
    model_ex = cl.PCSAFT(["water", "ethanol"], 
                        userlocations=["singledata_PCSAFT.csv", "pairdata_PCSAFT.csv", 
                                      "assocdata_PCSAFT.csv"])
    
    assert np.array_equal(model_og.params.segment.values, model_ex.params.segment.values)
    assert np.array_equal(model_og.params.epsilon.values, model_ex.params.epsilon.values)
    assert np.array_equal(model_og.params.epsilon_assoc.values.values, 
                         model_ex.params.epsilon_assoc.values.values)

def test_export_model_Cubic():
    model_og = cl.PR(["water", "ethanol"])
    cl.export_model(model_og)
    model_ex = cl.PR(["water", "ethanol"], 
                    userlocations=["singledata_PR.csv", "pairdata_PR.csv"],
                    alpha_userlocations=["singledata_PRAlpha.csv"])
    
    assert np.array_equal(model_og.params.a.values, model_ex.params.a.values)
    assert np.array_equal(model_og.alpha.params.acentricfactor.values, 
                         model_ex.alpha.params.acentricfactor.values)

def test_export_model_Activity_GC():
    model_og = cl.UNIFAC(["water", "ethanol"])
    cl.export_model(model_og)
    model_ex = cl.UNIFAC(["water", "ethanol"], 
                        userlocations=["singledata_UNIFAC.csv", "pairdata_UNIFAC.csv"])
    
    assert np.array_equal(model_og.params.Q.values, model_ex.params.Q.values)
    assert np.array_equal(model_og.params.A.values, model_ex.params.A.values)

# @testset "single component error" begin
def test_single_component_error():
    model = cl.PCSAFT(["water", "methane"])
    
    with pytest.raises(Exception):  # DimensionMismatch
        cl.saturation_pressure(model, 300.15)
    with pytest.raises(Exception):  # DimensionMismatch
        cl.crit_pure(model)
    with pytest.raises(Exception):  # DimensionMismatch
        cl.saturation_temperature(model, 1e5)
    with pytest.raises(Exception):  # DimensionMismatch
        cl.acentric_factor(model)
    with pytest.raises(Exception):  # DimensionMismatch
        cl.enthalpy_vap(model, 300.15)
    with pytest.raises(Exception):  # DimensionMismatch
        cl.x0_sat_pure(model, 300.15)
    with pytest.raises(Exception):  # DimensionMismatch
        cl.saturation_liquid_density(model, 300.15)

# @testset "macros" begin
def test_macros():
    def comps(model):
        return cl.comps(model)
    
    def groups_func(model, i=None):
        if i is None:
            return cl.groups(model)
        return cl.groups(model, i)
    
    def sites_func(model, i):
        return cl.sites(model, i)
    
    def f_eos(model, V, T, z, c):
        return 2 + c
    
    assert list(groups_func(gc2)) == list(range(6))
    assert list(comps(gc2)) == list(range(2))
    assert groups_func(gc2, 2) == [2, 3, 4, 5, 6]  # Second component groups (1-indexed in Julia)
    assert groups_func(gc3, 1) == groups_func(gc3, 2)  # propane and butane has the same amount of groups
    assert sites_func(gc2.vrmodel, 2) == [3, 4, 5]
    assert sites_func(model2, 1) == [1, 2]
    assert f_eos(None, None, None, None, np.pi) == 2 + np.pi
    assert cl.nan_default(np.log(-1), 3) == 3
    
    with pytest.raises(Exception):  # MethodError
        cl.nan_default(None, 3)

# @testset "has_sites-has_groups" begin
def test_has_sites_has_groups():
    assert cl.has_sites(type(gc3)) == False
    assert cl.has_sites(type(model2)) == cl.has_sites(type(model4)) == cl.has_sites(type(gc2)) == True
    assert cl.has_groups(type(gc2)) == cl.has_groups(type(gc3)) == cl.has_groups(type(ideal1)) == True
    assert cl.has_groups(type(simple1)) == cl.has_groups(type(model2)) == False

# @testset "eosshow" begin
def test_eosshow():
    # @newmodelgc
    assert repr(ideal1) == "WalkerIdeal{BasicIdeal}(\"hexane\")"
    
    # @newmodel
    assert repr(model2) == "PCSAFT{BasicIdeal, Float64}(\"water\", \"ethanol\")"
    
    # @newmodelsimple
    assert repr(noparam1) == "NoTranslation()"
    assert repr(simple1) == "PRAlpha(\"propane\")"

# @testset "Clapeyron Param show" begin
def test_Clapeyron_Param_show():
    assert repr(model2.params) == "Clapeyron.PCSAFTParam{Float64}"

# @testset "phase symbols" begin
def test_phase_symbols():
    assert cl.canonical_phase("l") == "liquid"
    assert cl.canonical_phase("v") == "vapour"
    assert cl.canonical_phase("m") == "m"

# @testset "citing" begin
def test_citing():
    umr = cl.UMRPR(["water"], idealmodel=cl.WalkerIdeal)
    
    citation_full = set(cl.cite(umr))
    citation_top = set(cl.doi(umr))
    citation_ideal = set(cl.cite(umr.idealmodel))
    citation_mixing = set(cl.cite(umr.mixing))
    citation_translation = set(cl.cite(umr.translation))
    
    assert citation_ideal.issubset(citation_full)
    assert citation_top.issubset(citation_full)
    assert citation_mixing.issubset(citation_full)
    assert citation_translation.issubset(citation_full)
    
    citation_show = cl.show_references(umr)
    assert citation_show.startswith("\nReferences:")
    assert cl.doi2bib("10.1021/I160057A011").startswith("@article{Peng_1976")

# @testset "alternative input" begin
def test_alternative_input():
    assert isinstance(cl.PCSAFT({"water": [("H2O", 1)]}, idealmodel=cl.WalkerIdeal), cl.EoSModel)
    assert isinstance(cl.PCSAFT([{"water": [("H2O", 1)]}], idealmodel=cl.WalkerIdeal), cl.EoSModel)
    assert isinstance(cl.PCSAFT("water"), cl.EoSModel)
    assert isinstance(cl.JobackIdeal("hexane"), cl.EoSModel)
    assert isinstance(cl.PCSAFT([{"water": [("H2O", 1)]}]), cl.EoSModel)

# @testset "core utils" begin
def test_core_utils():
    assert cl.parameterless_type(type(np.random.rand(5))) == np.ndarray
    assert cl.vec_parser("1 2 3") == [1, 2, 3]
    assert cl.vec_parser("1 2 3.5") == [1, 2, 3.5]
    
    with pytest.raises(Exception):  # ErrorException
        cl.vec_parser("not numbers")
    
    assert cl.split_2("a b") == ("a", "b")
    assert cl.split_2("a|b", '|') == ("a", "b")
