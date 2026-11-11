import cbhbd.cluster
import numpy as np


def test_merger():
    print("Checking that the code runs without crashing when mergers are enabled", flush=True)
    tend = 13e3
    M0 = 8e5
    rg = 8
    rh0 = 1
    Z = 0.003
    seed = 12345
    verbose = True

    model = cbhbd.cbhbd.CBHBD(tend=tend, M0=M0, Z=Z, rh0=rh0, compute_mergers=True, rg=rg, seed=seed, verbose=verbose)

    assert model.cluster is not None, "Cluster is not properly computed"
    assert model.mergers is not None, "Mergers are not properly computed"
    assert len(model.mergers) != 0, "No mergers found but some are expected for these initial conditions"

    assert model.mIMBH > 0, f"IMBH mass is not properly computed, {model.mIMBH = }"
    assert model.chiIMBH > 0, f"IMBH spin is not properly computed, {model.chiIMBH = }"
    assert model.genIMBH > 0, f"IMBH generation is not properly computed, {model.genIMBH = }"

    assert model.mIMBHej > 0, f"IMBHej mass is not properly computed, {model.mIMBHej = }"
    assert model.chiIMBHej > 0, f"IMBHej spin is not properly computed, {model.chiIMBHej = }"
    assert model.genIMBHej > 0, f"IMBHej generation is not properly computed, {model.genIMBHej = }"

    assert model.cluster.M_interp(model.cluster.tfin) >= 0, "Final cluster mass is not properly computed"
    assert model.cluster.Mbh_interp(model.cluster.tfin) >= 0, "Final BH mass is not properly computed"
    assert model.cluster.rh_interp(model.cluster.tfin) >= 0, "Final half-mass radius is not properly computed"


def test_merger_small_cluster():
    print(
        "Checking that the code runs without crashing when mergers are enabled and the cluster is too small to have BHs",
        flush=True)
    tend = 13e3
    M0 = 1e2
    rg = 8
    rh0 = 1
    Z = 0.003
    seed = 12345
    verbose = True

    model = cbhbd.cbhbd.CBHBD(tend=tend, M0=M0, Z=Z, rh0=rh0, compute_mergers=True, rg=rg, seed=seed, verbose=verbose)


def test_merger_seed():
    print(
        "Checking that the code returns the same results when run with the same seed",
        flush=True)
    tend = 13e3
    M0 = 8e5
    rg = 8
    rh0 = 1
    Z = 0.003
    seed = 12345
    verbose = True
    remnant_model = "RemnantModelSimple"

    model1 = cbhbd.cbhbd.CBHBD(tend=tend, M0=M0, Z=Z, rh0=rh0, compute_mergers=True, rg=rg, seed=seed, verbose=verbose,
                               remnant_model=remnant_model)
    model2 = cbhbd.cbhbd.CBHBD(tend=tend, M0=M0, Z=Z, rh0=rh0, compute_mergers=True, rg=rg, seed=seed, verbose=verbose,
                               remnant_model=remnant_model)

    try:
        assert np.allclose(model1.cluster.M_interp(model1.cluster.tfin), model2.cluster.M_interp(model2.cluster.tfin))
        assert np.allclose(model1.cluster.rh_interp(model1.cluster.tfin), model2.cluster.rh_interp(model2.cluster.tfin))
        assert np.allclose(model1.cluster.Mbh_interp(model1.cluster.tfin),
                           model2.cluster.Mbh_interp(model2.cluster.tfin))
        assert np.allclose(model1.cluster.m_interp(model1.cluster.tfin), model2.cluster.m_interp(model2.cluster.tfin))
        assert np.allclose(model1.cluster.mst_interp(model1.cluster.tfin),
                           model2.cluster.mst_interp(model2.cluster.tfin))
        assert np.allclose(model1.cluster.Edot_rlx_interp(model1.cluster.tfin),
                           model2.cluster.Edot_rlx_interp(model2.cluster.tfin))
    except AssertionError as e:
        print("Clusters are not the same, error likely in cluster.py")
        raise e

    try:
        assert np.allclose(model1.bhv0, model2.bhv0)
    except AssertionError as e:
        print("Initial BH population is not the same, error likely in sample_BHs in funcs.py")
        raise e

    try:
        assert np.allclose(np.log10(model1.mergers.t_merge), np.log10(model2.mergers.t_merge))
        assert np.allclose(np.log10(model1.mergers.t_sim), np.log10(model2.mergers.t_sim))
        assert np.allclose(model1.mergers.m1, model2.mergers.m1)
        assert np.allclose(model1.mergers.m2, model2.mergers.m2)
        assert np.allclose(model1.mergers.a, model2.mergers.a)
        assert np.allclose(model1.mergers.e, model2.mergers.e)
        assert np.allclose(model1.mergers.chi_f, model2.mergers.chi_f)
        assert np.allclose(model1.mergers.rh, model2.mergers.rh)
        assert np.allclose(model1.mergers.v_kick, model2.mergers.v_kick)
        assert np.allclose(model1.mergers.v_esc, model2.mergers.v_esc)
        assert (model1.mergers.merger_type == model2.mergers.merger_type).all()
        assert np.allclose(model1.mergers.gen, model2.mergers.gen)
        assert np.allclose(model1.mergers.Mbh, model2.mergers.Mbh)
        assert np.allclose(model1.mergers.S1, model2.mergers.S1)
        assert np.allclose(model1.mergers.S2, model2.mergers.S2)
        assert np.allclose(model1.mergers.m_rem, model2.mergers.m_rem)

        assert np.allclose(model1.bhv, model2.bhv, equal_nan=True)
    except AssertionError as e:
        print("Mergers are not the same, error likely in mergers.py")
        raise e
