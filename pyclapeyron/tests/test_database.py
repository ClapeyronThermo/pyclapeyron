import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np
import juliacall
from juliacall import Main as jl
import os

# The rest of the test will be conducted with a custom dataset in the test_csvs directory.
testspecies = ["sp1", "sp2", "sp3", "sp4", "sp5"]

csv_path = os.path.dirname(os.path.abspath(__file__))
filepath_normal = [csv_path + "/test_csvs/normal_pair_test.csv",
                   csv_path + "/test_csvs/normal_single_test.csv",
                   csv_path + "/test_csvs/normal_assoc_test.csv",
                   csv_path + "/test_csvs/normal_single2_test.csv",
                   csv_path + "/test_csvs/normal_assoc2_test.csv"]

filepath_clashingheaders = [csv_path + "/test_csvs/headercollision_single_test.csv",
                            csv_path + "/test_csvs/headercollision_assoc_test.csv"]

filepath_asymmetry = [csv_path + "/test_csvs/asymmetry_pair_test.csv",
                      csv_path + "/test_csvs/asymmetry_assoc_test.csv"]

filepath_multiple_identifiers = [csv_path + "/test_csvs/multiple_identifiers_single_test.csv",
                                 csv_path + "/test_csvs/multiple_identifiers_pair_test.csv",
                                 csv_path + "/test_csvs/multiple_identifiers_assoc_test.csv"]

filepath_gc = [csv_path + "/test_csvs/group_test.csv"]
filepath_param_gc = [csv_path + "/test_csvs/group_param_test.csv"]

# @testset "getparams - general" begin
def test_getparams_general():
    opts = cl.ParamOptions()
    opts2 = cl.ParamOptions(ignore_missing_singleparams=jl.Array[jl.String](["emptyparam","missingparam"])) #TODO this is ugly
    allparams, allnotfoundparams = cl.Clapeyron.createparams(jl.Array[jl.String](testspecies), jl.Array[jl.String](filepath_normal), opts)
    
    params1 = cl.getparams(["water", "methanol"], jl.Array[jl.String](["SAFT/PCSAFT"]), return_sites=False)
    # test that we have a sigma inside params1
    assert "sigma" in params1
    
    for param_i in params1.values():
        repr(param_i)
    
    # this fails, because there are missing params
    with pytest.raises(Exception):  # MissingException
        cl.Clapeyron.compile_params(testspecies, allparams, allnotfoundparams, None, opts)
    
    # Check that it throws an error if ignore_missing_singleparams is not set to true.
    with pytest.raises(Exception):  # MissingException
        cl.Clapeyron.getparams(testspecies, userlocations=jl.Array[jl.String](filepath_normal), return_sites=False)
    
    # Clashing headers between association and non-association parameters are not allowed
    with pytest.raises(Exception):  # ErrorException
        cl.Clapeyron.getparams(testspecies, userlocations=jl.Array[jl.String](filepath_clashingheaders))
    
    # If parameter is not tagged as ignore_missing_singleparams, incomplete diagonal will throw an error
    with pytest.raises(Exception):  # MissingException
        cl.Clapeyron.getparams(testspecies, userlocations=jl.Array[jl.String](filepath_asymmetry), return_sites=False)
    
    with pytest.raises(Exception):  # MissingException
        cl.Clapeyron.getparams(testspecies, userlocations=jl.Array[jl.String](filepath_asymmetry), 
                    asymmetricparams=jl.Array[jl.String](["asymmetricpair", "asymmetricassoc"]))

# @testset "params - printing" begin
def test_params_printing():
    params = cl.getparams(testspecies, userlocations=jl.Array[jl.String](filepath_normal), 
                         ignore_missing_singleparams=jl.Array[jl.String](["emptyparam","missingparam"]), 
                         verbose=True)
    sites = cl.SiteParam(params["intparam"].components)
    
    # Printing: SingleParam
    assert jl.repr(params["intparam"]) == "SingleParam{Int64}(\"intparam\")[\"sp1\", \"sp2\", \"sp3\", \"sp4\", \"sp5\"]"
    
    # Printing: PairParam
    assert jl.repr(params["overwriteparam"]) == "PairParam{Float64}(\"overwriteparam\")[\"sp1\", \"sp2\", \"sp3\", \"sp4\", \"sp5\"]"
    
    # Printing: SiteParam
    assert jl.repr(sites) == "SiteParam[\"sp1\" => [], \"sp2\" => [], \"sp3\" => [], \"sp4\" => [], \"sp5\" => []]"

# @testset "params - types and values" begin
def test_params_types_and_values():
    params = cl.getparams(testspecies, userlocations=jl.Array[jl.String](filepath_normal), #TODO
                         ignore_missing_singleparams=jl.Array[jl.String](["emptyparam","missingparam"]),    #TODO 
                         verbose=True)
    
    # Check that all the types are correct.
    assert jl.isa(params["intparam"], cl.Clapeyron.SingleParam)
    assert jl.isa(params["doubleparam"], cl.Clapeyron.SingleParam)
    assert jl.isa(params["stringparam"], cl.Clapeyron.SingleParam)
    
    # If column has both strings and numbers, they should all be strings.
    assert jl.isa(params["mixedparam"], cl.Clapeyron.SingleParam)
    
    # Contains missing values
    assert jl.isa(params["missingparam"], cl.Clapeyron.SingleParam)
    
    # All missing values
    assert jl.isa(params["emptyparam"], cl.Clapeyron.SingleParam)
    
    # Overwrite Int with Float, and also an upgrade from single to pair param
    assert jl.isa(params["overwriteparam"], cl.PairParam)
    
    # test for missingness in PairParam
    expected_ismissing = np.array([[False, False, True, True, True],
                                   [False, False, True, False, True],
                                   [True, True, False, False, True],
                                   [True, False, False, False, True],
                                   [True, True, True, True, False]])
    assert np.array_equal(params["overwriteparam"].ismissingvalues, expected_ismissing)
    
    # Overwrite String with Int
    assert jl.isa(params["overwritestringparam"], cl.Clapeyron.SingleParam)
    # Overwrite Int with String
    assert jl.isa(params["overwriteassocparam"], cl.AssocParam)
    
    # Check that values of "sp1" and "sp3" has been correctly overwritten.
    assert list(params["intparam"].values) == [6, 2, 7, 4, 5]
    
    # Also check that "testsource1" and "testsource3" are not present
    assert "testsource1" not in params["intparam"].sources
    assert "testsource3" not in params["intparam"].sources
    assert "testsource5" in params["intparam"].sources
    assert "testsource19" in params["intparam"].sources
    
    # Check that missing values have been correctly defaulted.
    assert list(params["missingparam"].values) == [1, 2, 3, 0, 0]
    assert list(params["missingparam"].ismissingvalues) == [False, False, False, True, True]
    
    # Check that the promotion form 1D to 2D array is succesful
    expected_values = np.array([[1.6, 4.0, 0.0, 0.0, 0.0],
                                [4.0, 1.2, 0.0, 3.0, 0.0],
                                [0.0, 0.0, 1.3, 2.0, 0.0],
                                [0.0, 3.0, 2.0, 1.4, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.5]])
    assert np.allclose(params["overwriteparam"].values, expected_values)
    
    # assert np.allclose(cl.diagvalues(params["overwriteparam"]), [1.6, 1.2, 1.3, 1.4, 1.5]) #Internals
    # assert cl.diagvalues(1.23) == 1.23
    
    assoc_param = params["assocparam"]
    # diagonal 3-3
    assert assoc_param[("sp3","H"),("sp3","e")] == 2000
    assert assoc_param[("sp3","e"),("sp3","H")] == 2000
    assert assoc_param[("sp3","H"),("sp3","e2")] == 2700
    assert assoc_param[("sp3","e2"),("sp3","H")] == 2700
    
    # asym: 3-4
    assert assoc_param[("sp3","H"),("sp4","e")] == 2400
    assert assoc_param[("sp4","e"),("sp3","H")] == 2400
    assert assoc_param[("sp3","H"),("sp4","H")] == 2300
    assert assoc_param[("sp4","H"),("sp3","H")] == 2300
    assert assoc_param[("sp3","e"),("sp4","H")] == 0
    assert assoc_param[("sp4","H"),("sp3","e")] == 0
    assert assoc_param[("sp3","e"),("sp4","e")] == 0
    assert assoc_param[("sp4","e"),("sp3","e")] == 0
    # assert assoc_param["sp3","sp4"].shape == assoc_param["sp4","sp3"].T.shape
    
    # asym 4-5
    assert assoc_param[("sp4","H"),("sp5","e2")] == 2500
    assert assoc_param[("sp5","e2"),("sp4","H")] == 2500
    assert assoc_param[("sp4","e2"),("sp5","H")] == 0
    assert assoc_param[("sp5","H"),("sp4","e2")] == 0
    
    # asym 4-5
    assert assoc_param[("sp4","H"),("sp5","e")] == 0
    assert assoc_param[("sp5","e"),("sp4","H")] == 0
    assert assoc_param[("sp4","e"),("sp5","H")] == 2600
    assert assoc_param[("sp5","H"),("sp4","e")] == 2600
    
    # diagonal 5-5
    assert assoc_param[("sp5","H"),("sp5","e")] == 2100
    assert assoc_param[("sp5","e"),("sp5","H")] == 2100
    assert assoc_param[("sp5","e"),("sp5","e")] == 2200
    
    # Testing for asymmetric params
    asymmetricparams = cl.getparams(testspecies, userlocations=jl.Array[jl.String](filepath_asymmetry), #TODO
                                   asymmetricparams=jl.Array[jl.String](["asymmetricpair", "asymmetricassoc"]), 
                                   ignore_missing_singleparams=jl.Array[jl.String](["asymmetricpair"]))
    
    expected_asym = np.array([[0.06, 0.04, 0.0, 0.0, 0.0],
                              [0.05, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.02, 0.0],
                              [0.0, 0.03, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0]])
    assert np.allclose(asymmetricparams["asymmetricpair"].values, expected_asym)
    
    # Testing for multiple identifiers
    multiple_identifiers = cl.getparams(testspecies, userlocations=jl.Array[jl.String](filepath_multiple_identifiers)) #TODO
    assert list(multiple_identifiers["param_single"].values) == [100, 200, 200, 300, 300]
    
    expected_pair = np.array([[1000, 4000, 0, 0, 0],
                              [4000, 2000, 0, 6000, 0],
                              [0, 0, 2000, 5000, 0],
                              [0, 6000, 5000, 3000, 0],
                              [0, 0, 0, 0, 3000]])
    assert np.array_equal(multiple_identifiers["param_pair"].values, expected_pair)
    assert (multiple_identifiers["param_assoc"].values[3,2][0,0] == 10000 and
            multiple_identifiers["param_assoc"].values[5,1][0,0] == 20000)

# @testset "params - conversion,broadcasting,indexing" begin -> internals
# @testset "GroupParam" begin
def test_GroupParam():
    # GC test, 3 comps, 4 groups
    components_gc = cl.GroupParam(["test1", "test2", ("test3", {"grp1": 2, "grp2": 2, "grp3": 3, "grp4": 5})], 
                                 group_userlocations=filepath_gc)
    
    # Printing: GroupParam
    assert jl.repr(components_gc) == "GroupParam[\"test1\" => [\"grp1\" => 1, \"grp2\" => 2], \"test2\" => [\"grp2\" => 1], \"test3\" => [\"grp1\" => 2, \"grp2\" => 2, \"grp3\" => 3, \"grp4\" => 5]]"
    
    assert str(components_gc.grouptype) == "test"
    assert components_gc.components == jl.Array[jl.String](["test1", "test2", "test3"])
    
    # Build param struct using the gc components above
    param_gc = cl.getparams(components_gc, userlocations=jl.Array[jl.String](filepath_param_gc))
    assert list(param_gc["param1"].values) == [1, 2, 3, 4]

paramtable_file = cl.ParamTable(jl.Symbol("single"), jl.concretize({"species": ["sp1", "sp2"], "userparam": np.array([2, 10])})) #TODO avoid jl.Symbol?

# @testset "ParamTable" begin
def test_ParamTable():
    param_user = cl.getparams(testspecies, userlocations=jl.Array[jl.String]([paramtable_file]), 
                             ignore_missing_singleparams=jl.Array[jl.String](["userparam"]))
    assert param_user["userparam"].values[0] == 2

# @testset "direct csv parsing" begin
def test_direct_csv_parsing():
    # reading external data, via direct CSV parsing:
    csv_string = """Clapeyron Database File,
in memory like parameters
species,userparam,b,c
sp1,1000,0.05,4
sp2,,0.41,5
"""
    param_user2 = cl.getparams(["sp1", "sp2"], userlocations=jl.Array[jl.String]([csv_string]), #TODO userlocations type
                              ignore_missing_singleparams=jl.Array[jl.String](["userparam"]))
    assert param_user2["userparam"].values[0] == 1000
    
    # @REPLACE keyword
    # paramtable_file = cl.ParamTable(jl.Symbol("single"), jl.concretize({"species": ["sp1", "sp2"], "userparam": np.array([2, 10])})) #TODO avoid jl.Symbol?
    param_user3 = cl.getparams(["sp1", "sp2"], 
                              userlocations=jl.Array[jl.String]([paramtable_file, "@REPLACE/" + csv_string]), 
                              ignore_missing_singleparams=jl.Array[jl.String](["userparam"]))
    assert param_user3["userparam"].ismissingvalues[1] == True

# @testset "named tuple parsing" begin
def test_named_tuple_parsing():
    model_nt = cl.PCSAFT(["a1"], userlocations={
        "Mw": np.array([1.]),
        "epsilon": np.array([2.]),
        "sigma": np.array([3.]),
        "segment": np.array([4.]),
        "k": np.zeros((1, 1)),
        "n_H": np.array([1]),
        "n_e": np.array([1]),
        "epsilon_assoc": jl.concretize({(("a1", "e"), ("a1", "H")): 1000.}),
        "bondvol": jl.concretize({(("a1", "e"), ("a1", "H")): 0.001})
    })
    
    assert model_nt.params.Mw[1] == 1.
    assert model_nt.params.epsilon[1] == 2.
    assert model_nt.params.sigma[1] == 3e-10  # pcsaft multiplies sigma by 1e-10
    assert model_nt.params.segment[1] == 4.
    assert model_nt.params.epsilon_assoc.values.values[0] == 1000.
    assert model_nt.params.bondvol.values.values[0] == 0.001
    
    model_nt2 = cl.PCSAFT(["a1"], userlocations={
        "Mw": np.array([1.]),
        "epsilon": np.array([2.]),
        "sigma": np.array([3.]),
        "segment": np.array([4.]),
        "k": np.zeros((1, 1)),
        "epsilon_assoc": None,
        "bondvol": None
    })
    assert len(model_nt2.params.bondvol.values.values) == 0
