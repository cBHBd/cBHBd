import cbhbd.clusterbh


def test_clusterbh():
    print("Checking that clusterBH runs without crashing with test parameters")
    N = 1e6
    rhoh = 1e4

    cbh = cbhbd.clusterbh.ClusterBH(N, rhoh)
