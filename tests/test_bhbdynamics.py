import cbhbd.clusterbh
import numpy as np


def test_bhbdynamics():
    print("Checking that BHBdynamics runs without crashing with test parameters")
    t_fin = 13
    Mcl = 8e5
    rg = 8
    rh = 1
    rho_h_i = (Mcl / 2) / ((4 / 3) * np.pi * rh ** 3)
    Z = 0.003
    seed = 12345
    verbose = True

    bbh, _ = cbhbd.bhbdynamics.run_model(t_fin, Mcl, Z, rho_h_i, rg=rg, seed=seed, verbose=verbose)
