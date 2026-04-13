# ᴄBHBᴅ

A fast code for simulating star clusters and black hole dynamics.

ᴄBHBᴅ can be used for

1) the evolution of a star cluster with black holes in a tidal field
2) population synthesis of binary black hole mergers in a time-evolving star cluster

If you use ᴄBHBᴅ in any publication, please cite it as follows:
> ᴄBHBᴅ (Antonini & Gieles [2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.2936A/abstract); Antonini et
> al. [2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522..466A/abstract)), with the updates of Fronimos Pouliasis et
> al. (2025)

ᴄBHBᴅ stands for the combination of ᴄʟᴜsᴛᴇʀBH (the code for the cluster evolution) and BHBᴅʏɴᴀᴍɪᴄs (the code for the
black hole mergers).

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

## Running ᴄBHBᴅ

You can run ᴄBHBᴅ as

```
import cbhbd.cbhbd

cbhbd.cbhbd.CBHBD()
```

with the following arguments:

- `N`: Initial number of stars _or_ `M0`: Initial cluster mass
- `rhoh0`: Initial half-mass density [Msun/pc^3] _or_ `rh0`: Initial half-mass radius [pc]
- `Z`: Metallicity _or_ `FeH`: Metallicity as in [Fe/H]
- `kwargs`: Additional parameters to override defaults (see code comments in the `CBHBD` class and the `Cluster` class)

If you only want to run the cluster evolution, you can use `compute_mergers=False` for efficiency