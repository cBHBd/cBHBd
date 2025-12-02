import pathlib
import subprocess

CURRENT_PATH = pathlib.Path(__file__).parent.resolve()


def run_sse(mass, z, supernova_model, idum):
    # Mass in solar units
    # Metallicity in the range 0.0001 -> 0.03 where 0.02 is Population I
    # Random number seed

    # Maximum evolution time in Myr
    tphysf = 13000

    neta = 0.0  # Unused
    bwind = 0.0  # Unused
    hewind = 0.0  # Unused

    sigma = 265.0  # Dispersion in the Maxwellian for the SN kick speed (km/s)
    ifflag = 1
    wdflag = 1
    bhflag = 2

    # nsflag = 1/2/3/4
    # 1: Remnant-mass prescription of Belczynski et al., 2002, ApJ, 572, 407. (original BSE model)
    # 2: Remnant-mass prescription of Belczynski et al., 2008, ApJS, 174, 223. (B08 model)
    # 3: Remnant-mass prescription of Fryer et al., 2012, ApJ, 749, 91. (F12-rapid model)
    # 4: Remnant-mass prescription of Fryer et al., 2012, ApJ, 749, 91. (F12-delayed model)
    assert supernova_model in ["RAPID", "DELAY"], "Unknown supernova model"
    nsflag = 3 if supernova_model == "RAPID" else 4

    mxns = 2.5

    # psflag = 1/0
    # 1: PPSN/PSN schemes according to Belczynski et al., 2016, A&A 594, A97. (B16-PPSN/PSN)
    # 0: No PPSN/PSN
    psflag = 1

    # kmech = 1/2/3/4
    # 1: Standard, momentum-conserving kick of Belczynski et al., 2008, ApJS, 174, 223.
    # 2: Convection-asymmetry-driven kick
    # 3: Collapse-asymmetry-driven kick
    # 4: Neutrino-emission-asymmetry-driven kick
    kmech = 1

    # ecflag = 1/0
    # 1: ECS-NS formation according to Belczynski et al., 2008, ApJS, 174, 223.
    # 0: No ECS-NS formation
    ecflag = 1

    # Recommended values
    pts1 = 0.001
    pts2 = 0.01
    pts3 = 0.02

    # Set parameters
    params = (f"{mass} {z} {tphysf} "
              f"{neta} {bwind} {hewind} {sigma} "
              f"{ifflag} {wdflag} {bhflag} {nsflag} {mxns} {idum} "
              f"{psflag} {kmech} {ecflag} "
              f"{pts1} {pts2} {pts3}")
    # with open("evolve.in", "w") as f:
    #     f.write(f"{mass} {z} {tphysf}\n"
    #             f"{neta} {bwind} {hewind} {sigma}\n"
    #             f"{ifflag} {wdflag} {bhflag} {nsflag} {mxns} {idum}\n"
    #             f"{psflag} {kmech} {ecflag}\n"
    #             f"{pts1} {pts2} {pts3}")

    # Format
    #
    # mass,z,tphysf
    # neta,bwind,hewind,sigma
    # ifflag,wdflag,bhflag,nsflag,mxns,idum
    # psflag,kmech,ecflag
    # pts1,pts2,pts3

    # Run SSE
    cmd = [f"{CURRENT_PATH}/./updated-BSE/sse_new", ]
    cmd.extend(params.split())

    full_out = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    errout = full_out.stderr.decode('UTF-8')
    stdout = full_out.stdout.decode('UTF-8')
    assert errout == "", [errout, stdout]

    # Get properties after evolution
    lastline = stdout.split("\n")[-3]
    assert float(lastline.split()[-3]) == tphysf, "tfin not identified properly"
    isBH = "Black Hole" in lastline

    if not isBH:
        return isBH, None, None
    mBH = float(lastline.split()[-1])

    # Get natal kick
    firstline = stdout.split("\n")[1]
    assert "MASS KS FBFAC FBTOT MCO VKICK KMECH" in firstline, f"First line of output not identified properly:\n{firstline}"
    vkick = float(firstline.split()[-2])

    return isBH, mBH, vkick


def get_smallest_BH_prog(Z, mmin, mmax, supernova_model, mtol=0.01):
    # Get the mass of the smallest star that turns into a BH

    seed = 1111  # Irrelevant
    assert not run_sse(mmin, Z, supernova_model, seed)[0], f"A star with m0 = mmin ({mmin = }) turns into a BH"
    assert run_sse(mmax, Z, supernova_model, seed)[0], f"A star with m0 = mmax ({mmax = }) does not turn into a BH"

    while mmax - mmin > mtol:
        mcand = (mmin + mmax) / 2
        isBH, _, _ = run_sse(mcand, Z, supernova_model, seed)
        if isBH:
            mmax = mcand
        else:
            mmin = mcand

    return mmax
