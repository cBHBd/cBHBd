# ᴄBHBᴅ

A fast code for simulating star clusters and black hole dynamics.

ᴄBHBᴅ can be used for

1) the evolution of a star cluster with black holes in a tidal field
2) population synthesis of binary black hole mergers in a time-evolving star cluster

If you use ᴄBHBᴅ in any publication, please cite it as follows:
> ᴄBHBᴅ (Antonini & Gieles [2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.2936A/abstract); Antonini et
> al. [2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522..466A/abstract)), with the updates of Fronimos Pouliasis et
> al. ([2025](https://ui.adsabs.harvard.edu/abs/2026arXiv260528088F/abstract)),
> Chattopadhyay et al. ([2026](https://ui.adsabs.harvard.edu/abs/2026arXiv260409773C/abstract)), and Marín Pina et al.
> (in prep)

ᴄBHBᴅ stands for the combination of ᴄʟᴜsᴛᴇʀBH (the code for the cluster evolution) and BHBᴅʏɴᴀᴍɪᴄs (the code for the
black hole mergers).

## Installation

The cBHBd package can be installed using pip:

```
pip install cbhbd
```

All required dependencies will be installed automatically.

You can also simply download the ᴄBHBᴅ package source from this repository and install or place it into your path:

```
git clone https://github.com/cBHBd/cBHBd
```

## Running ᴄBHBᴅ

You can run ᴄBHBᴅ as

```
from cbhbd.cbhbd import CBHBD

c = CBHBD(...)
```

with the following arguments:

- `N`: Initial number of stars _or_ `M0`: Initial cluster mass
- `rhoh0`: Initial half-mass density [Msun/pc^3] _or_ `rh0`: Initial half-mass radius [pc]
- `Z`: Metallicity _or_ `FeH`: Metallicity as in [Fe/H]
- `kwargs`: Additional parameters to override defaults (see code comments in the `CBHBD` class and the `Cluster` class)

If you only want to run the cluster evolution, you can use `compute_mergers=False` for efficiency

## Analysing the ᴄBHBᴅ output

The output of the simulations is stored in a `CBHBD` object. The cluster properties can be accessed from `c.cluster`.
See `cluster.py` for the full documentation.

```
c.cluster.Mst # Total stellar mass in the cluster as function of time
c.cluster.Mbh # Total BH mass in the cluster as function of time
c.cluster.rh # Half-mass radius of the cluster as function of time
...
c.cluster.Mst_interp(t) # For convenience, you can also interpolate the cluster properties at any time using the interpolating functions
```

The merger properties are stored in a `pandas.Dataframe` in `c.mergers`. See `mergers.py` for the full documentation.

```
c.mergers # The merger data, use print(c.mergers) to see a table of the merger properties
c.mergers.m1 # The mass of the primary
c.mergers.merger_type # The type of merger (ejected, gw_capture, ...), see the code for the full documentation
... 
c.mergers.to_numpy() # You can also export the merger data to a numpy array
```

