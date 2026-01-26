import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np
from juliacall import Main as jl
import os

def test_params():
    testspecies = ["sp1", "sp2", "sp3", "sp4", "sp5"]

    # Utils
    csv_path = os.path.dirname(os.path.abspath(__file__))
    filepath_normal = [csv_path + "/test_csvs/normal_pair_test.csv",
                    csv_path + "/test_csvs/normal_single_test.csv",
                    csv_path + "/test_csvs/normal_assoc_test.csv",
                    csv_path + "/test_csvs/normal_single2_test.csv",
                    csv_path + "/test_csvs/normal_assoc2_test.csv"]
    filepath_gc = [csv_path + "/test_csvs/group_test.csv"]
    
    # getparams
    params = cl.getparams(testspecies, userlocations=filepath_normal,
                         ignore_missing_singleparams=["emptyparam","missingparam"], 
                         verbose=True)
    
    assert jl.isa(params["intparam"], cl.Clapeyron.SingleParam)
    assert jl.isa(params["doubleparam"], cl.Clapeyron.SingleParam)
    assert jl.isa(params["stringparam"], cl.Clapeyron.SingleParam)

    # group params
    components_gc = cl.GroupParam(
        ["test1", "test2", ("test3", {"grp1": 2, "grp2": 2, "grp3": 3, "grp4": 5})], group_userlocations=filepath_gc
    )
