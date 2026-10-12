import cbhbd.cbhbd


def test_cluster():
    print("Checking that the code runs without crashing when mergers are disabled")
    tend = 13e3
    M0 = 8e5
    rg = 8
    rh0 = 1
    Z = 0.003
    seed = 12345
    verbose = True

    model = cbhbd.cbhbd.CBHBD(tend=tend, M0=M0, Z=Z, rh0=rh0, compute_mergers=False, rg=rg, seed=seed, verbose=verbose)

    assert model.cluster is not None, "Cluster is not properly computed"
