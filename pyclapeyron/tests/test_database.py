import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np
from juliacall import Main as jl

# @testset "Database Parsing" begin

# @testset "normalisestring" begin -> removed

# The rest of the test will be conducted with a custom dataset in the test_csvs directory.
testspecies = ["sp1", "sp2", "sp3", "sp4", "sp5"]

filepath_normal = ["test_csvs/normal_pair_test.csv",
                   "test_csvs/normal_single_test.csv",
                   "test_csvs/normal_assoc_test.csv",
                   "test_csvs/normal_single2_test.csv",
                   "test_csvs/normal_assoc2_test.csv"]

filepath_clashingheaders = ["test_csvs/headercollision_single_test.csv",
                            "test_csvs/headercollision_assoc_test.csv"]

filepath_asymmetry = ["test_csvs/asymmetry_pair_test.csv",
                      "test_csvs/asymmetry_assoc_test.csv"]

filepath_multiple_identifiers = ["test_csvs/multiple_identifiers_single_test.csv",
                                 "test_csvs/multiple_identifiers_pair_test.csv",
                                 "test_csvs/multiple_identifiers_assoc_test.csv"]
filepath_gc = ["test_csvs/group_test.csv"]
filepath_param_gc = ["test_csvs/group_param_test.csv"]

# @testset "getparams - general" begin
def test_getparams_general():
    opts = cl.ParamOptions()
    opts2 = cl.ParamOptions(ignore_missing_singleparams=jl.Array[jl.String](["emptyparam","missingparam"])) #TODO this is ugly
    allparams, allnotfoundparams = cl.createparams(testspecies, filepath_normal, opts)
    
    params1 = cl.getparams(["water", "methanol"], ["SAFT/PCSAFT"], return_sites=False)
    # test that we have a sigma inside params1
    assert "sigma" in params1
    
    for param_i in params1.values():
        repr(param_i)
    
    # this fails, because there are missing params
    with pytest.raises(Exception):  # MissingException
        cl.compile_params(testspecies, allparams, allnotfoundparams, None, opts)
    
    # Check that it throws an error if ignore_missing_singleparams is not set to true.
    with pytest.raises(Exception):  # MissingException
        cl.getparams(testspecies, userlocations=filepath_normal, return_sites=False)
    
    # Clashing headers between association and non-association parameters are not allowed
    with pytest.raises(Exception):  # ErrorException
        cl.getparams(testspecies, userlocations=filepath_clashingheaders)
    
    # If parameter is not tagged as ignore_missing_singleparams, incomplete diagonal will throw an error
    with pytest.raises(Exception):  # MissingException
        cl.getparams(testspecies, userlocations=filepath_asymmetry, return_sites=False)
    
    # Also, since a non-missing value exists on the diagonal of "asymmetricpair",
    # and the diagonal contains missing values, it should throw an error
    # when parameter is not in ignore_missing_singleparams
    with pytest.raises(Exception):  # MissingException
        cl.getparams(testspecies, userlocations=filepath_asymmetry, 
                    asymmetricparams=["asymmetricpair", "asymmetricassoc"])

# @testset "params - sites" begin
def test_params_sites():
    opts2 = cl.ParamOptions(ignore_missing_singleparams=["emptyparam","missingparam"])
    allparams, allnotfoundparams = cl.createparams(testspecies, filepath_normal, cl.ParamOptions())
    sites2 = cl.buildsites(testspecies, allparams, allnotfoundparams, opts2)
    result = cl.compile_params(testspecies, allparams, allnotfoundparams, sites2, opts2)
    
    assert [set(s) for s in sites2.sites] == [set(),
                                               set(),
                                               set(["e", "e2", "H"]),
                                               set(["e", "H"]),
                                               set(["e", "e2", "H"])]

# @testset "params - printing" begin
def test_params_printing():
    params = cl.getparams(testspecies, userlocations=filepath_normal, 
                         ignore_missing_singleparams=["emptyparam","missingparam"], 
                         verbose=True)
    sites = cl.SiteParam(params["intparam"].components)
    
    # Printing: SingleParam
    assert repr(params["intparam"]) == "SingleParam{Int64}(\"intparam\")[\"sp1\", \"sp2\", \"sp3\", \"sp4\", \"sp5\"]"
    
    # Printing: PairParam
    assert repr(params["overwriteparam"]) == "PairParam{Float64}(\"overwriteparam\")[\"sp1\", \"sp2\", \"sp3\", \"sp4\", \"sp5\"]"
    
    # Printing: SiteParam
    assert repr(sites) == "SiteParam[\"sp1\" => [], \"sp2\" => [], \"sp3\" => [], \"sp4\" => [], \"sp5\" => []]"

# @testset "params - types and values" begin
def test_params_types_and_values():
    params = cl.getparams(testspecies, userlocations=filepath_normal, 
                         ignore_missing_singleparams=["emptyparam","missingparam"], 
                         verbose=True)
    
    # Check that all the types are correct.
    assert isinstance(params["intparam"], cl.SingleParam)
    assert isinstance(params["doubleparam"], cl.SingleParam)
    assert isinstance(params["stringparam"], cl.SingleParam)
    
    # If column has both strings and numbers, they should all be strings.
    assert isinstance(params["mixedparam"], cl.SingleParam)
    
    # Contains missing values
    assert isinstance(params["missingparam"], cl.SingleParam)
    
    # All missing values
    assert isinstance(params["emptyparam"], cl.SingleParam)
    
    # Overwrite Int with Float, and also an upgrade from single to pair param
    assert isinstance(params["overwriteparam"], cl.PairParam)
    
    # test for missingness in PairParam
    expected_ismissing = np.array([[False, False, True, True, True],
                                   [False, False, True, False, True],
                                   [True, True, False, False, True],
                                   [True, False, False, False, True],
                                   [True, True, True, True, False]])
    assert np.array_equal(params["overwriteparam"].ismissingvalues, expected_ismissing)
    
    # Overwrite String with Int
    assert isinstance(params["overwritestringparam"], cl.SingleParam)
    # Overwrite Int with String
    assert isinstance(params["overwriteassocparam"], cl.AssocParam)
    
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
    
    # test that Passing a single param to a pair param doesnt erase the missings
    singletopairparam = cl.PairParam(params["missingparam"], "singletopairparam")
    diagmissingvalues = np.diag(singletopairparam.ismissingvalues)
    assert np.array_equal(diagmissingvalues, params["missingparam"].ismissingvalues)
    
    # Check that the promotion form 1D to 2D array is succesful
    expected_values = np.array([[1.6, 4.0, 0.0, 0.0, 0.0],
                                [4.0, 1.2, 0.0, 3.0, 0.0],
                                [0.0, 0.0, 1.3, 2.0, 0.0],
                                [0.0, 3.0, 2.0, 1.4, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.5]])
    assert np.allclose(params["overwriteparam"].values, expected_values)
    
    assert np.allclose(cl.diagvalues(params["overwriteparam"]), [1.6, 1.2, 1.3, 1.4, 1.5])
    assert cl.diagvalues(1.23) == 1.23
    
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
    assert assoc_param["sp3","sp4"].shape == assoc_param["sp4","sp3"].T.shape
    
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
    asymmetricparams = cl.getparams(testspecies, userlocations=filepath_asymmetry, 
                                   asymmetricparams=["asymmetricpair", "asymmetricassoc"], 
                                   ignore_missing_singleparams=["asymmetricpair"])
    
    expected_asym = np.array([[0.06, 0.04, 0.0, 0.0, 0.0],
                              [0.05, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.02, 0.0],
                              [0.0, 0.03, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0]])
    assert np.allclose(asymmetricparams["asymmetricpair"].values, expected_asym)
    
    # Testing for multiple identifiers
    multiple_identifiers = cl.getparams(testspecies, userlocations=filepath_multiple_identifiers)
    assert list(multiple_identifiers["param_single"].values) == [100, 200, 200, 300, 300]
    
    expected_pair = np.array([[1000, 4000, 0, 0, 0],
                              [4000, 2000, 0, 6000, 0],
                              [0, 0, 2000, 5000, 0],
                              [0, 6000, 5000, 3000, 0],
                              [0, 0, 0, 0, 3000]])
    assert np.array_equal(multiple_identifiers["param_pair"].values, expected_pair)
    assert (multiple_identifiers["param_assoc"].values[3,2][1,1] == 10000 and
            multiple_identifiers["param_assoc"].values[5,1][1,1] == 20000)

# @testset "params - conversion,broadcasting,indexing" begin
def test_params_conversion_broadcasting_indexing():
    comps = ["aa", "bb"]
    floatbool = cl.SingleParam("float to bool", comps, [1.0, 0.0])
    intbool = cl.SingleParam("int to bool", comps, [1, 0])
    str_single = cl.SingleParam("int to bool", comps, ["111", "222"])
    substr_single = cl.SingleParam("int to bool", comps, ["11", "22"])
    
    assert floatbool.values.shape == (2,)
    
    # SingleParam - conversion
    assert isinstance(cl.convert(cl.SingleParam, floatbool, dtype=bool), cl.SingleParam)
    assert isinstance(cl.convert(cl.SingleParam, str_single, dtype=str), cl.SingleParam)
    assert isinstance(cl.convert(cl.SingleParam, substr_single, dtype=str), cl.SingleParam)
    assert isinstance(cl.convert(cl.SingleParam, intbool, dtype=float), cl.SingleParam)
    assert isinstance(cl.convert(cl.SingleParam, floatbool, dtype=int), cl.SingleParam)
    
    # SingleParam - indexing
    assert intbool["aa"] == 1
    intbool["bb"] = 2
    assert intbool["bb"] == 2
    with pytest.raises(Exception):  # BoundsError
        intbool["cc"]
    with pytest.raises(Exception):  # BoundsError
        intbool["cc"] = 2
    
    # pair param conversion and broadcasting
    floatbool = cl.PairParam("float to bool", comps, np.array([[1.0, 0.0], [0.0, 1.0]]))
    intbool = cl.PairParam("int to bool", comps, np.array([[1, 2], [3, 4]]))
    
    assert floatbool.values.shape == (2, 2)
    assert isinstance(cl.convert(cl.PairParam, floatbool, dtype=bool), cl.PairParam)
    assert isinstance(cl.convert(cl.PairParam, intbool, dtype=float), cl.PairParam)
    assert isinstance(cl.convert(cl.PairParam, floatbool, dtype=int), cl.PairParam)
    
    # indexing for PairParam
    floatbool[1, 2] = 1000
    assert floatbool[2, 1] == 1000
    floatbool[1, 2, False] = 4000
    assert floatbool[2, 1] == 1000
    floatbool[1] = 1.2
    assert floatbool[1, 1] == 1.2
    assert floatbool["aa", "bb"] == 4000
    assert floatbool["aa"] == 1.2
    assert floatbool["aa", "aa"] == 1.2
    assert floatbool["bb", "aa"] == 1000
    
    with pytest.raises(Exception):  # BoundsError
        floatbool["cc"]
    with pytest.raises(Exception):  # BoundsError
        floatbool["cc", "aa"]
    with pytest.raises(Exception):  # BoundsError
        floatbool["aa", "cc"]
    
    # pack vectors
    s1 = cl.SingleParam("s1", comps, [1, 2])
    s2 = cl.SingleParam("s2", comps, [10, 20])
    s = cl.pack_vectors(s1, s2)
    assert list(s[1]) == [1, 10]
    
    # assoc param
    ijab = [(1, 1, 1, 2), (2, 2, 1, 2), (2, 2, 1, 3)]
    
    float_vals = [1., 2., 3.]
    int_vals = [1, 0, 1]
    str_vals = ["v1", "v2", "v3"]
    
    assoc_vals_float = cl.Compressed4DMatrix(float_vals, ijab)
    assoc_vals_int = cl.Compressed4DMatrix(int_vals, ijab)
    assoc_vals_str = cl.Compressed4DMatrix(str_vals, ijab)
    assoc_vals_substr = cl.Compressed4DMatrix(["v", "v", "v"], ijab)
    
    assoc_float = cl.AssocParam("float", ["comp1", "comp2"], assoc_vals_float)
    assoc_int = cl.AssocParam("int", ["comp1", "comp2"], assoc_vals_int)
    assoc_str = cl.AssocParam("str", ["comp1", "comp2"], assoc_vals_str)
    assoc_substr = cl.AssocParam("str", ["comp1", "comp2"], assoc_vals_substr)
    
    # AssocParam: indexing
    assert assoc_float.values.dtype == np.float64
    assert np.allclose(assoc_float[1], np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert np.allclose(assoc_float[2], np.array([[0.0, 2.0, 3.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
    assert assoc_float[1, 2].size == 0
    
    # AssocParam: Conversion
    assert isinstance(cl.convert(cl.AssocParam, assoc_float, dtype=bool), cl.AssocParam)
    assert isinstance(cl.convert(cl.AssocParam, assoc_int, dtype=float), cl.AssocParam)
    assert isinstance(cl.convert(cl.AssocParam, assoc_int, dtype=int), cl.AssocParam)
    assert isinstance(cl.convert(cl.AssocParam, assoc_substr, dtype=str), cl.AssocParam)
    assert isinstance(cl.convert(cl.AssocParam, assoc_str, dtype=str), cl.AssocParam)
    
    # single param: mixing rules with Real and Int type
    single_real = cl.SingleParam("tes", ["a", "b", "c"], [1., 2., 3.])
    single_int = cl.SingleParam("tes", ["a", "b", "c"], [1, 2, 3])
    assert isinstance(cl.sigma_LorentzBerthelot(single_real), cl.PairParam)
    assert isinstance(cl.sigma_LorentzBerthelot(single_int), cl.PairParam)
    assert isinstance(cl.lambda_LorentzBerthelot(single_real, 0.5), cl.PairParam)
    assert isinstance(cl.lambda_LorentzBerthelot(single_int, 0.5), cl.PairParam)

# @testset "GroupParam" begin
def test_GroupParam():
    # GC test, 3 comps, 4 groups
    components_gc = cl.GroupParam(["test1", "test2", ("test3", [("grp1", 2), ("grp2", 2), ("grp3", 3), ("grp4", 5)])], 
                                 group_userlocations=filepath_gc)
    
    # Printing: GroupParam
    assert repr(components_gc) == "GroupParam[\"test1\" => [\"grp1\" => 1, \"grp2\" => 2], \"test2\" => [\"grp2\" => 1], \"test3\" => [\"grp1\" => 2, \"grp2\" => 2, \"grp3\" => 3, \"grp4\" => 5]]"
    
    assert components_gc.grouptype == "test"
    assert components_gc.components == ["test1", "test2", "test3"]
    assert components_gc.groups == [["grp1", "grp2"], ["grp2"], ["grp1", "grp2", "grp3", "grp4"]]
    assert components_gc.n_groups == [[1, 2], [1], [2, 2, 3, 5]]
    
    # Check that flattening of groups is correct.
    assert components_gc.flattenedgroups == ["grp1", "grp2", "grp3", "grp4"]
    assert components_gc.n_flattenedgroups == [[1, 2, 0, 0], [0, 1, 0, 0], [2, 2, 3, 5]]
    
    # Build param struct using the gc components above
    param_gc = cl.getparams(components_gc, userlocations=filepath_param_gc)
    assert list(param_gc["param1"].values) == [1, 2, 3, 4]

# @testset "ParamTable" begin
def test_ParamTable():
    paramtable_file = cl.ParamTable("single", species=["sp1", "sp2"], userparam=[2, 10])
    # reading external data, via ParamTable
    param_user = cl.getparams(testspecies, userlocations=[paramtable_file], 
                             ignore_missing_singleparams=["userparam"])
    assert param_user["userparam"].values[1] == 2

# @testset "direct csv parsing" begin
def test_direct_csv_parsing():
    # reading external data, via direct CSV parsing:
    csv_string = """Clapeyron Database File,
in memory like parameters
species,userparam,b,c
sp1,1000,0.05,4
sp2,,0.41,5
"""
    param_user2 = cl.getparams(["sp1", "sp2"], userlocations=[csv_string], 
                              ignore_missing_singleparams=["userparam"])
    assert param_user2["userparam"].values[1] == 1000
    
    # @REPLACE keyword
    paramtable_file = cl.ParamTable("single", species=["sp1", "sp2"], userparam=[2, 10])
    param_user3 = cl.getparams(["sp1", "sp2"], 
                              userlocations=[paramtable_file, "@REPLACE/" + csv_string], 
                              ignore_missing_singleparams=["userparam"])
    assert param_user3["userparam"].ismissingvalues[2] == True

# @testset "named tuple parsing" begin
def test_named_tuple_parsing():
    model_nt = cl.PCSAFT(["a1"], userlocations={
        "Mw": jl.Array([1.]),
        "epsilon": np.array([2.]),
        "sigma": np.array([3.]),
        "segment": np.array([4.]),
        "k": np.zeros((1, 1)),
        "n_H": np.array([1]),
        "n_e": np.array([1]),
        "epsilon_assoc": {(("a1", "e"), ("a1", "H")): 1000.},
        "bondvol": {(("a1", "e"), ("a1", "H")): 0.001}
    })
    
    assert model_nt.params.Mw[1] == 1.
    assert model_nt.params.epsilon[1] == 2.
    assert model_nt.params.sigma[1] == 3e-10  # pcsaft multiplies sigma by 1e-10
    assert model_nt.params.segment[1] == 4.
    assert model_nt.params.epsilon_assoc.values.values[1] == 1000.
    assert model_nt.params.bondvol.values.values[1] == 0.001
    
    model_nt2 = cl.PCSAFT(["a1"], userlocations={
        "Mw": [1.],
        "epsilon": [2.],
        "sigma": [3.],
        "segment": [4.],
        "k": np.zeros((1, 1)),
        "epsilon_assoc": None,
        "bondvol": None
    })
    assert len(model_nt2.params.bondvol.values.values) == 0

# @testset "misc database utils" begin
# removed as only testing internals
