import pytest
from pytest import approx
import pyclapeyron as cl
import numpy as np
import os

def test_cubic():
    cl.vdW(["ethane", "undecane"])
    cl.Clausius(["ethane", "undecane"])
    cl.Berthelot(["ethane", "undecane"])
    cl.RK(["ethane", "undecane"])
    cl.SRK(["ethane", "undecane"])
    cl.PSRK(["ethane", "undecane"])
    cl.tcRK(["ethane", "undecane"])
    cl.PR(["ethane", "undecane"])
    cl.PR78(["ethane", "undecane"])
    cl.VTPR(["ethane", "undecane"])
    cl.UMRPR(["ethane", "undecane"])
    cl.QCPR(["neon", "helium"])
    cl.QCPR(["helium"])
    cl.cPR(["ethane", "undecane"])
    cl.tcPR(["ethane", "undecane"])
    cl.tcPRW(["ethane", "undecane"])
    cl.EPPR78(["benzene", "isooctane"])
    cl.KU(["ethane", "undecane"])
    cl.RKPR(["ethane", "undecane"])
    cl.PatelTeja(["ethane", "undecane"])
    cl.PTV(["ethane", "undecane"])
    cl.YFR(["ethane", "undecane"])
    cl.YFR("water")

def test_saft_vr():
    cl.SAFTVRMie(["methanol", "water"])
    cl.SAFTVRMieGV(["benzene", "acetone"])
    cl.SAFTVRMieGV(["carbon dioxide", "benzene"])
    cl.SAFTVRMieGV(["acetone", "diethyl ether", "ethyl acetate"])
    cl.SAFTVRQMie(["helium"])
    cl.SAFTVRSMie(["carbon dioxide"])
    cl.SAFTVRMie15("water")
    cl.SAFTgammaMie(["methanol", "butane"])
    species = [
        ("ethanol", {"CH3": 1, "CH2OH": 1}, {("CH3", "CH2OH"): 1}),
        ("octane", {"CH3": 2, "CH2": 6}, {("CH3", "CH2"): 2, ("CH2", "CH2"): 5})
    ]
    cl.structSAFTgammaMie(species)

def test_saft_pc():
    cl.PCSAFT(["butane", "ethanol"])
    cl.PCSAFT(["a1"], userlocations={
        "Mw": np.array([1.]),
        "epsilon": np.array([2.]),
        "sigma": np.array([3.]),
        "segment": np.array([4.]),
        "k": np.zeros((1, 1)),
        "n_H": np.array([1]),
        "n_e": np.array([1]),
        "epsilon_assoc": {(("a1", "e"), ("a1", "H")): 1000.},
        "bondvol": {(("a1", "e"), ("a1", "H")): 0.001}
    })
    cl.PCPSAFT(["acetone", "butane", "DMSO"])
    cl.PCPSAFT(["acetone", "water", "DMSO"])
    cl.QPCPSAFT(["carbon dioxide", "acetone", "hydrogen sulfide"])
    cl.QPCPSAFT(["carbon dioxide", "chlorine", "carbon disulfide"])
    cl.gcPCPSAFT(["acetone", "ethane", "ethanol"], mixing="homo")
    cl.gcPCPSAFT(["acetone", "ethane", "ethanol"], mixing="hetero")
    cl.sPCSAFT(["propane", "methanol"])
    cl.sPCSAFT(["water", "ethanol"])
    cl.gcsPCSAFT(["acetone", "ethane"])
    species = [
        ("ethanol", {"CH3": 1, "CH2": 1, "OH": 1}, {("CH3", "CH2"): 1, ("OH", "CH2"): 1}),
        ("hexane", {"CH3": 2, "CH2": 4}, {("CH3", "CH2"): 2, ("CH2", "CH2"): 3})
    ]
    cl.gcPCSAFT(species)
    cl.CPPCSAFT(["butane", "ethanol"])
    cl.CPPCSAFT("water")
    cl.CPPCSAFT(["butane", "propane"])
    cl.CPPCSAFT(["water", "ethanol"])
    cl.GEPCSAFT(["propane", "methanol"])
    cl.ADPCSAFT(["water"])
    cl.DAPT(["water"])

def test_saft_others():
    cl.ogSAFT(["water", "ethylene glycol"])
    cl.CKSAFT(["carbon dioxide", "2-propanol"])
    cl.sCKSAFT(["benzene", "acetic acid"])
    cl.BACKSAFT(["carbon dioxide"])
    cl.LJSAFT(["ethane", "1-propanol"])
    cl.SAFTVRSW(["water", "ethane"])
    cl.softSAFT(["hexane", "1-propanol"])
    cl.softSAFT2016(["hexane", "1-propanol"])
    cl.solidsoftSAFT(["octane"])
    cl.CPA(["ethanol", "benzene"])
    cl.sCPA(["water", "carbon dioxide"])
    cl.COFFEE(["a1"], userlocations=dict(
        Mw=np.array([1.]),
        segment=np.array([1.]),
        sigma=np.array([[3.]]),
        epsilon=np.array([[300.]]),
        lambda_r=np.array([[12]]),
        lambda_a=np.array([[6]]),
        shift=np.array([3*0.15]),
        dipole=np.array([2.0*1.0575091914494172]),
    ))

def test_models_others():
    cl.Wilson(["methanol", "benzene"])
    cl.NRTL(["methanol", "benzene"])
    cl.aspenNRTL(["methanol", "benzene"])
    cl.UNIQUAC(["methanol", "benzene"])
    cl.UNIFAC(["methanol", "benzene"])
    cl.UNIFAC2([("acetaldehyde", {"CH3": 1, "HCO": 1}), ("acetonitrile", {"CH3CN": 1})], puremodel=cl.BasicIdeal())
    cl.BasicIdeal()
    cl.ogUNIFAC(["methanol", "benzene"])
    cl.ogUNIFAC2([("R22", {"HCCLF2": 1}), ("carbon disulfide", {"CS2": 1})], puremodel=cl.BasicIdeal())
    cl.UNIFACFV(["PMMA", "PS"])
    cl.UNIFACFVPoly(["PMMA", "PS"])
    cl.FH(["PMMA", "PS"], np.array([100, 100]))
    cl.COSMOSAC02(["water", "ethanol"])
    cl.COSMOSAC10(["water", "ethanol"])
    cl.COSMOSACdsp(["water", "ethanol"])
    cl.JobackIdeal(["hexane"])
    cl.JobackIdeal("acetone")
    cl.ReidIdeal(["butane"])
    cl.ShomateIdeal(["water"])
    cl.WalkerIdeal(["hexane"])
    cl.MonomerIdeal(["hexane"])
    cl.EmpiricIdeal(["water"])
    cl.Clapeyron.idealmodel(cl.MultiFluid(["water"]))
    cl.AlyLeeIdeal(["methane"])
    cl.Clapeyron.idealmodel(cl.GERG2008(["methane"]))
    cl.GERG2008(["methane"])
    cl.CPLNGEstIdeal(["a1"], userlocations={'Mw': np.array([20.5200706797])})
    cl.PPDSIdeal("krypton")
    cl.PPDSIdeal("methanol")
    cl.IAPWS95()
    cl.PropaneRef()
    cl.GERG2008(["water"])
    cl.GERG2008(["water", "carbon dioxide", "hydrogen sulfide", "argon"])
    cl.GERG2008(["methane", "ethane"], Rgas=8.2)
    cl.EOS_LNG(["methane", "butane"])
    cl.LKP(["methane", "butane"])
    cl.LKPSJT(["methane", "butane"])
    cl.LJRef(["methane"])
    cl.XiangDeiters(["water"])
    cl.MultiFluid(["carbon dioxide", "hydrogen"], verbose=True)
    cl.SingleFluid("ammonia", verbose=True)
    cl.SingleFluid("water", Rgas=10.0)
    cl.SPUNG(["ethane"])
    cl.SPUNG(["ethane"], cl.PropaneRef(), cl.PCSAFT(["ethane"]), cl.PCSAFT(["propane"]))
    cl.SanchezLacombe(["carbon dioxide"])
    cl.SanchezLacombe(["carbon dioxide", "benzoic acid"], mixing=cl.SLKRule)
    cl.SanchezLacombe(["carbon dioxide", "benzoic acid"], mixing=cl.SLk0k1lMixingRule)
    cl.AntoineEqSat(["water"])
    cl.DIPPR101Sat(["water"])
    cl.LeeKeslerSat(["water"])
    cl.COSTALD(["water"])
    cl.COSTALD(["water", "methanol"])
    cl.DIPPR105Liquid(["water"])
    cl.DIPPR105Liquid(["water", "methanol"])
    cl.RackettLiquid(["water"])
    cl.RackettLiquid(["water", "methanol"])
    cl.YamadaGunnLiquid(["water"])
    cl.YamadaGunnLiquid(["water", "methanol"])
    cl.GrenkeElliottWater()
    cl.HoltenWater()
    cl.AbbottVirial(["methane", "ethane"])
    cl.TsonopoulosVirial(["methane", "ethane"])
    cl.PR(["methane", "ethane"])
    cl.SolidHfus(["water"])
    cl.SolidKs(["water"])
    cl.IAPWS06()
    cl.JagerSpanSolidCO2()

def test_electrolytes():
    cl.ConstRSP()
    cl.ZuoFurst()
    cl.LinMixRSP(["water"], ["sodium", "chloride"])
    cl.Schreckenberg(["water"], ["sodium", "chloride"])
    cl.ePCSAFT(["water"], ["sodium", "chloride"])
    cl.eSAFTVRMie(["water"], ["sodium", "chloride"])
    cl.SAFTVREMie(["water"], ["sodium", "chloride"])
    cl.SAFTVREMie(["water"], ["sodium", "chloride"], ionmodel=cl.MSA)
    cl.SAFTVREMie(["water"], ["sodium", "chloride"], ionmodel=cl.Born)
    cl.MSAID(["water"], ["sodium", "chloride"], userlocations=dict(charge=np.array([0, 1, -1]), sigma=np.array([4., 4., 4.]), dipole=np.array([2.2203, 0., 0.])))
    cl.SAFTgammaEMie(["water"], ["sodium", "acetate"])
    cl.SAFTgammaEMie(["water"], ["calcium", "chloride"])