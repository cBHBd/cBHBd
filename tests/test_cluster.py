import cbhbd.cbhbd


def test_cluster():
    print("Checking that the code runs without crashing when mergers are disabled")
    N0 = 1e6
    rhoh0 = 1e4
    Z = 0.003

    model = cbhbd.cbhbd.CBHBD(N=N0, rhoh0=rhoh0, Z=Z, compute_mergers=False, verbose=True)

    assert model.cluster is not None, "Cluster is not properly computed"
