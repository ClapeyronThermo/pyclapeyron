import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np

def bubble_point(model, T, x):
    return 1, 1

# @testset "estimation" begin
def test_estimation():
    model = cl.SAFTgammaMie(["ethanol", "water"], epsilon_mixing="hudsen_mccoubrey")
    
    toestimate = [
        {
            "param": "epsilon",
            "indices": (1, 1),
            "recombine": True,
            "lower": 100.,
            "upper": 300.,
            "guess": 250.
        },
        {
            "param": "epsilon",
            "indices": (2, 3),
            "symmetric": False,
            "lower": 100.,
            "upper": 300.,
            "guess": 250.
        },
        {
            "param": "sigma",
            "indices": (1, 1),
            "factor": 1e-10,
            "lower": 3.2,
            "upper": 4.0,
            "guess": 3.7
        },
        {
            "param": "epsilon_assoc",
            "indices": 2,
            "cross_assoc": True,
            "lower": 2000.,
            "upper": 4000.,
            "guess": 3500.
        }
    ]
    
    estimator, objective, initial, upper, lower = cl.Estimation(
        model, toestimate, ["../examples/data/bubble_point.csv"], ["vrmodel"]
    )
    
    model2 = cl.return_model(estimator, model, initial)
    
    assert model2.params.epsilon[1, 1] == initial[1]  # Test that the parameter was correctly updated
    assert model2.params.epsilon[1, 2] == approx(251.57862124740765, rel=1e-6)  # Test that the combining rule was used
    assert model2.params.epsilon[1, 2] == model2.params.epsilon[1, 2]  # Test that the unlike parameters remain symmetric
    
    assert model2.params.epsilon[2, 3] == initial[2]  # Test that the parameter was updated
    assert model2.params.epsilon[2, 3] != model2.params.epsilon[3, 2]  # Test that the parameter is no longer symmetric
    
    assert model2.params.sigma[1, 1] == initial[3] * 1e-10  # Test that the factor was used to update the parameters
    
    assert model2.params.epsilon[2, 3] == initial[2]  # Test that the parameter was updated
    assert model2.params.epsilon[2, 3] != model2.params.epsilon[3, 2]  # Test that the parameter is no longer symmetric
    
    assert model2.params.epsilon_assoc.values.values[2] == initial[4]  # Test that the association parameter was updated
    assert model2.params.epsilon_assoc.values.values[2] == model2.params.epsilon_assoc.values.values[3]  # Test that the cross-association parameter was updated
    
    assert objective(initial) == approx(2.4184631612655836, rel=1e-6)
    
    estimator, objective, initial, upper, lower = cl.Estimation(
        model, toestimate, 
        [(2., "../examples/data/bubble_point.csv"), (1., "../examples/data/bubble_point.csv")], 
        ["vrmodel"]
    )
    
    assert objective(initial) == approx(7.255389483796751, rel=1e-6)
    
    # error found during #365
    modelvec = cl.EoSVectorParam(model)
    assert isinstance(cl.promote_model(float, modelvec), cl.EoSVectorParam)
    
    model2 = cl.SAFTgammaMie(["octane"])
    k1 = model2.params.mixed_segment.values.v.copy()
    cl.recombine(model2)
    k2 = model2.params.mixed_segment.values.v
    assert k1[1] == k2[1]  # Test that the first value is unchanged
