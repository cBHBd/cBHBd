import multiprocessing as mp

import numpy as np
from tqdm import tqdm

import sample
from run_sse import get_smallest_BH_prog
from run_sse import run_sse

LEN_SAMP = 1000000
MAX_INT = 2_147_483_647

mmin = 0.08
mmax = 130
alphaIMF = 2.3  # Slope of the mass function at the highest mass range (BH progenitors)

# Metallicity
Z_step = 0.1
Z_arr = 10 ** np.arange(-3.9, np.log10(0.02), Z_step)

supernova_model = "RAPID"  # Alternatively, "DELAY"

for Z in tqdm(Z_arr, total=len(Z_arr)):
    progmmin = get_smallest_BH_prog(Z, mmin, mmax, supernova_model, mtol=0.01)

    assert progmmin >= 1, "Below the break of the IMF, check Kroupa's paper"
    msamp = sample.sample_power_law(progmmin, mmax, -alphaIMF, LEN_SAMP)

    seeds = ((MAX_INT - 1) * np.random.sample(LEN_SAMP)).astype(int)


    def parallel_run_sse(params):
        mZAMS = params[0]
        seed = params[1]
        isBH, mBH, vkick = run_sse(mZAMS, Z, supernova_model, seed)

        return isBH, mZAMS, mBH, vkick


    with mp.Pool() as pool:
        res = pool.map(parallel_run_sse, list(zip(msamp, seeds)), chunksize=1000)
        res = np.array(res)

        assert (res[:, 0] > 0.5).all(), "Found non-BH compact object"

        np.savetxt(f"../data/BHs/bh_Z{Z}.dat", res[:, 1:], header="mZAMS,mBH,vkick")
