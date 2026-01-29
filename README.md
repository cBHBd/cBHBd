# ᴄBHBᴅ
A fast code for simulating star clusters and black hole dynamics.

ᴄBHBᴅ is composed of two coupled codes

1) ᴄʟᴜsᴛᴇʀBH: Fast code for the evolution of a star cluster with black holes in a tidal field
2) BHBᴅʏɴᴀᴍɪᴄs: Fast code for population synthesis of binary black hole mergers in a star cluster

If you use ᴄBHBᴅ in any publication, please cite it as follows:
> ᴄBHBᴅ (Antonini & Gieles [2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.2936A/abstract); Antonini et al. [2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522..466A/abstract)), with the updates of Fronimos Pouliasis et al. (2025)

## Installation

The cBHBd package can be installed directly from this repository using pip:
```
python -m pip install git+https://github.com/cBHBd/cBHBd.git
```
All required dependencies will be installed automatically.

You can also simply download the ᴄBHBᴅ package source from this repository and install or place it into your path:
```
git clone https://github.com/cBHBd/cBHBd
```

## Running ᴄʟᴜsᴛᴇʀBH
You can run ᴄʟᴜsᴛᴇʀBH as
```
import cBHBd.clusterbh

cBHBd.clusterbh.clusterBH(N, rhoh)
```
with the following arguments: 
- `N`: Initial number of stars
- `rhoh`: Half-mass density [Msun/pc^3]
- `kwargs`: Additional parameters to override defaults (see code comments)


## Running BHBᴅʏɴᴀᴍɪᴄs
You can run BHBᴅʏɴᴀᴍɪᴄs as
```
import cBHBd.bhbdynamics

cBHBd.bhbdynamics.run_model(t_fin, Mcl0, Z, Z_file, rhoh0)
```
with the following arguments: 
- `t_fin`: Final time of the simulation [Gyr]
- `Mcl0`: Initial mass of the cluster [Msun]
- `Z`: Metallicity
- `Z_file`: Path to file with sampled BH masses and kicks, generated from `generate_metallicity_files.py`
- `rhoh0`: Initial density within the half-mass radius [Msun/pc^3]
- `rg`: Galactocentric radius [kpc]
- `output_dataframe`: If `True`, return is a pandas dataframe, otherwise a list
- `verbose`: If `True`, print extra output.
- `seed`: Seed for random number generator. Use `None` to get a random seed.
- `kwargs`: Additional arguments to pass to clusterBH.
