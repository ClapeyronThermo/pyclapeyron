import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np
from juliacall import Main as jl

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
        ("ibuprofen", {"CH3": 3, "COOH": 1, "aCCH": 1, "aCCH2": 1, "aCH": 4})
    ])
    ideal1 = cl.WalkerIdeal(["hexane"])
    noparam1 = gc3.puremodel[1].translation
    simple1 = gc3.puremodel[1].alpha
    model_structgc = cl.structSAFTgammaMie(["ethanol", "octane"])

# @testset "split_model" begin
def test_split_model():
    models2 = cl.split_model(model2)
    
    assert models2[0].components[0] == model2.components[0]
    assert models2[1].components[0] == model2.components[1]
    
    model2_unsplit = cl.split_model(model2, ([1, 2],))[0]
    assert model2_unsplit.components == model2.components
    
    model4_split = cl.split_model(model4)
    assert model4_split[3].groups.n_groups[0][1] == 3
    assert model4_split[0].components[0] == "methane"
    assert all(len(model4_split[i]) == 1 for i in range(4))
    
    gc3_split = cl.split_model(gc3)
    assert all(len(gc3_split[i]) == 1 for i in range(3))
    assert all(len(gc3_split[i].puremodel) == 1 for i in range(3))
    
    structgc_split = cl.split_model(model_structgc)
    assert np.array_equal(structgc_split[0].groups.n_intergroups[0], np.array([[0, 1], [1, 0]]))
    assert np.array_equal(structgc_split[1].groups.n_intergroups[0], np.array([[0, 2], [2, 5]]))
    
    noparam1_split = cl.split_model(noparam1, tuple(range(5)))
    assert len(noparam1_split) == 5
    assert noparam1_split[0] == noparam1
    
    # splitting parameters
    assert cl.split_model(model2.params.segment)[0][1] == model2.params.segment[1]
    assert cl.split_model(model2.params.sigma)[0][1, 1] == model2.params.sigma[1, 1]
    
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
    assert jl.isa(cl.split_model(model_nosites)[0], cl.PCSAFT)
    
    # error on splitting models with ReferenceState
    model_reference_state = cl.JobackIdeal(["propane", "hexane"])
    assert jl.isa(cl.split_model(model_reference_state)[0], cl.JobackIdeal)
    
    # index reduction testing
    model_idx = cl.PCSAFT(["ethane", "propane", "methane"])
    assert len(cl.index_reduction(model_idx, np.array([1., 0., 0.]))[0]) == 1
    assert len(cl.index_reduction(model_idx, np.array([1., 0., 1.]))[0]) == 2
    assert len(cl.index_reduction(model_idx, np.array([1., 1., 1.]))[0]) == 3
    assert len(cl.index_reduction(model_idx, np.array([1., -5e-16, 1.]))[0]) == 2

# @testset "export_model" begin
def test_export_model_SAFT():
    model_og = cl.PCSAFT(["water", "ethanol"])
    cl.export_model(model_og)
    model_ex = cl.PCSAFT(["water", "ethanol"], 
                        userlocations=jl.Array[jl.String](["singledata_PCSAFT.csv", "pairdata_PCSAFT.csv", 
                                      "assocdata_PCSAFT.csv"]))
    
    assert np.array_equal(model_og.params.segment.values, model_ex.params.segment.values)
    assert np.array_equal(model_og.params.epsilon.values, model_ex.params.epsilon.values)
    assert np.array_equal(model_og.params.epsilon_assoc.values.values, 
                         model_ex.params.epsilon_assoc.values.values)

def test_export_model_Cubic():
    model_og = cl.PR(["water", "ethanol"])
    cl.export_model(model_og)
    model_ex = cl.PR(["water", "ethanol"], 
                    userlocations=jl.Array[jl.String](["singledata_PR.csv", "pairdata_PR.csv"]),
                    alpha_userlocations=jl.Array[jl.String](["singledata_PRAlpha.csv"]))
    
    assert np.array_equal(model_og.params.a.values, model_ex.params.a.values)
    assert np.array_equal(model_og.alpha.params.acentricfactor.values, 
                         model_ex.alpha.params.acentricfactor.values)

def test_export_model_Activity_GC():
    model_og = cl.UNIFAC(["water", "ethanol"])
    cl.export_model(model_og)
    model_ex = cl.UNIFAC(["water", "ethanol"], 
                        userlocations=jl.Array[jl.String](["singledata_UNIFAC.csv", "pairdata_UNIFAC.csv"]))
    
    assert np.array_equal(model_og.params.Q.values, model_ex.params.Q.values)
    assert np.array_equal(model_og.params.A.values, model_ex.params.A.values)

# @testset "has_sites-has_groups" begin
def test_has_sites_has_groups():
    assert cl.has_sites(jl.typeof(gc3)) == False
    assert cl.has_sites(jl.typeof(model2)) == cl.has_sites(jl.typeof(model4)) == cl.has_sites(jl.typeof(gc2)) == True
    assert cl.has_groups(jl.typeof(gc2)) == cl.has_groups(jl.typeof(gc3)) == cl.has_groups(jl.typeof(ideal1)) == True
    assert cl.has_groups(jl.typeof(simple1)) == cl.has_groups(jl.typeof(model2)) == False


