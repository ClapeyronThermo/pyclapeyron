import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np
import gc

# @testset "SAFT methods, single components" begin
def test_Bulk_properties_single():
    system = cl.PCSAFT(["ethanol"])
    p = 1e5
    T = 298.15
    T2 = 373.15
    v = 5.907908736304141e-5
    
    assert cl.volume(system, p, T) == approx(v, rel=1e-6)
    assert cl.volume(system, p, T, phase="v") == approx(0.020427920501436134, rel=1e-6)
    assert cl.volume(system, p, T, threaded=False) == approx(v, rel=1e-6)
    assert cl.Clapeyron.pip(system, v, T) == approx(6.857076349623449, rel=1e-6)
    assert cl.Clapeyron.is_liquid(cl.Clapeyron.VT_identify_phase(system, v, T))
    assert cl.compressibility_factor(system, p, T) == approx(0.002383223535444557, rel=1e-6)
    assert cl.pressure(system, v, T) == approx(p, rel=1e-6)
    assert cl.pressure(system, 2*v, T, np.array([2.0])) == approx(p, rel=1e-6)
    
    s = cl.entropy(system, p, T)
    assert s == approx(-58.87118569239617, rel=1e-6)
    assert cl.Clapeyron.VT_entropy_res(system, v, T) + cl.Clapeyron.VT_entropy(cl.Clapeyron.idealmodel(system), v, T) == approx(s)
    
    assert cl.chemical_potential(system, p, T)[0] == approx(-18323.877542682934, rel=1e-6)
    
    u = cl.internal_energy(system, p, T)
    assert u == approx(-35882.22946560716, rel=1e-6)
    assert cl.Clapeyron.VT_internal_energy_res(system, v, T) + cl.Clapeyron.VT_internal_energy(cl.Clapeyron.idealmodel(system), v, T) == approx(u)
    
    h = cl.enthalpy(system, p, T)
    assert h == approx(-35876.32155687084, rel=1e-6)
    assert cl.Clapeyron.VT_enthalpy_res(system, v, T) + cl.Clapeyron.VT_enthalpy(cl.Clapeyron.idealmodel(system), v, T) == approx(h)
    
    g = cl.gibbs_free_energy(system, p, T)
    assert g == approx(-18323.87754268292, rel=1e-6)
    assert cl.Clapeyron.VT_gibbs_free_energy_res(system, v, T) + cl.Clapeyron.VT_gibbs_free_energy(cl.Clapeyron.idealmodel(system), v, T) == approx(g)
    
    a = cl.helmholtz_free_energy(system, p, T)
    assert a == approx(-18329.785451419295, rel=1e-6)
    assert cl.Clapeyron.VT_helmholtz_free_energy_res(system, v, T) + cl.Clapeyron.VT_helmholtz_free_energy(cl.Clapeyron.idealmodel(system), v, T) == approx(a)
    
    assert cl.isochoric_heat_capacity(system, p, T) == approx(48.37961296309505, rel=1e-6)
    assert cl.isobaric_heat_capacity(system, p, T) == approx(66.45719988319257, rel=1e-6)
    
    Cp = cl.isobaric_heat_capacity(system, p, T2)
    Cv = cl.isochoric_heat_capacity(system, p, T2)
    assert cl.adiabatic_index(system, p, T2) == approx(Cp/Cv, rel=1e-12)
    
    assert cl.isothermal_compressibility(system, p, T) == approx(1.1521981407243432e-9, rel=1e-6)
    assert cl.isentropic_compressibility(system, p, T) == approx(8.387789464951438e-10, rel=1e-6)
    assert cl.speed_of_sound(system, p, T) == approx(1236.4846683094133, rel=1e-6)  # requires that the model has Mr
    assert cl.isobaric_expansivity(system, p, T) == approx(0.0010874255138433413, rel=1e-6)
    assert cl.joule_thomson_coefficient(system, p, T) == approx(-6.007581864883784e-7, rel=1e-6)
    assert cl.second_virial_coefficient(system, T) == approx(-0.004919678119638886, rel=1e-6)
    assert cl.inversion_temperature(system, 1.1e8) == approx(824.4137805298458, rel=1e-6)
    assert cl.fugacity_coefficient(system, p, T, phase="l")[0] == approx(0.07865326632570452, rel=1e-6)

def test_VLE_properties_single():
    system = cl.PCSAFT(["ethanol"])
    p = 1e5
    T = 298.15
    
    assert cl.saturation_pressure(system, T)[0] == approx(7972.550405922014, rel=1e-6)
    assert cl.saturation_temperature(system, p)[0] == approx(351.32529505096164, rel=1e-6)
    assert cl.saturation_temperature(system, p, 350.)[0] == approx(351.32529505096164, rel=1e-6)
    assert cl.enthalpy_vap(system, T) == approx(41712.78521121877, rel=1e-6)
    assert cl.acentric_factor(system) == approx(0.5730309964718605, rel=1e-6)
    assert cl.crit_pure(system)[0] == approx(533.1324329774004, rel=1e-6)

# @testset "SAFT methods, multi-components" begin
def test_Bulk_properties_multi():
    system = cl.PCSAFT(["methanol", "cyclohexane"])
    p = 1e5
    T = 313.15
    z = np.array([0.5, 0.5])
    
    assert cl.volume(system, p, T, z) == approx(7.779694485714412e-5, rel=1e-6)
    assert cl.speed_of_sound(system, p, T, z) == approx(1087.0303138908864, rel=1e-6)
    assert cl.activity_coefficient(system, p, T, z)[0] == approx(1.794138454452822, rel=1e-6)
    assert cl.fugacity_coefficient(system, p, T, z)[0] == approx(0.5582931304564298, rel=1e-6)
    assert cl.mixing(system, p, T, z, cl.Clapeyron.gibbs_free_energy) == approx(-178.10797973342596, rel=1e-6)
    assert cl.excess(system, p, T, z, cl.Clapeyron.volume) == approx(1.004651584989827e-6, rel=1e-6)
    assert cl.excess(system, p, T, z, cl.Clapeyron.entropy) == approx(-2.832281578281112, rel=1e-6)
    assert cl.excess(system, p, T, z, cl.Clapeyron.gibbs_free_energy) == approx(1626.6212908893858, rel=1e-6)

def test_Equilibrium_properties_multi():
    system = cl.PCSAFT(["methanol", "cyclohexane"])
    p = 1e5
    T = 313.15
    z = np.array([0.5, 0.5])
    p2 = 2e6
    T2 = 443.15
    z2 = np.array([0.27, 0.73])
    
    # Those are the highest memory-intensive routines
    assert cl.gibbs_solvation(system, T) == approx(-13131.087644740426, rel=1e-6)
    assert cl.UCEP_mix(system)[0] == approx(319.36877456397684, rel=1e-6)
    assert cl.bubble_pressure(system, T, z)[0] == approx(54532.249600937736, rel=1e-6)
    assert cl.bubble_temperature(system, p2, z)[0] == approx(435.80890506865, rel=1e-6)
    assert cl.dew_pressure(system, T2, z)[0] == approx(1.6555486543884084e6, rel=1e-6)
    assert cl.dew_temperature(system, p2, z)[0] == approx(453.0056727580934, rel=1e-6)
    
    # res_LLE_p = cl.LLE_pressure(system, T, z2) #TODO error in Julia too !?
    # assert res_LLE_p[0] == approx(737971.7522006684, rel=1e-6)
    # assert cl.pressure(system, res_LLE_p[1], T, z2) == approx(cl.pressure(system, res_LLE_p[2], T, res_LLE_p[-1]), rel=1e-6)
    # assert cl.pressure(system, res_LLE_p[1], T, z2) == approx(res_LLE_p[0], rel=1e-6)
    
    # res_LLE_T = cl.LLE_temperature(system, p, z2) #TODO error in Julia too !?
    # T_LLE = res_LLE_T[0]
    # assert res_LLE_T[0] == approx(312.9523684945214, rel=1e-6)
    # assert cl.pressure(system, res_LLE_T[1], T_LLE, z2) == approx(cl.pressure(system, res_LLE_T[2], T_LLE, res_LLE_T[-1]), rel=1e-6)
    # assert cl.pressure(system, res_LLE_T[1], T_LLE, z2) == approx(p, rel=1e-6)
    
    assert cl.azeotrope_pressure(system, T2)[0] == approx(2.4435462800998255e6, rel=1e-6)
    assert cl.azeotrope_temperature(system, p)[0] == approx(328.2431049077264, rel=1e-6)
    assert cl.Clapeyron.UCST_mix(system, T2)[0] == approx(1.0211532467788119e9, rel=1e-6)
    assert cl.VLLE_pressure(system, T)[0] == approx(54504.079665621306, rel=1e-6)
    assert cl.VLLE_temperature(system, 54504.079665621306)[0] == approx(313.1499860368554, rel=1e-6)
    assert cl.crit_mix(system, z)[0] == approx(518.0004062881115, rel=1e-6)
