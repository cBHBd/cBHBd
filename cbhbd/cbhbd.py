import numpy as np
from . import cluster
from . import mergers


class CBHBD:
    remnant_model_simple = None
    remnant_model_NR = None

    def __init__(self, **kwargs):
        self._run_mergers = mergers._run_mergers
        self._get_IMBH_data = mergers._get_IMBH_data

        self.tend = 13.8e3  # [Myr] Requested final time for integration. Actual final time can be smaller if e.g. the cluster dissolves
        self.M0 = None  # [Msun] Initial mass of the cluster.
        self.N = None  # Initial number of stars in the cluster
        self.rh0 = None  # [pc] Initial half-mass radius
        self.rhoh0 = None  # [Msun/pc^3] Initial density within the half-mass radius
        self.rg = 8  # [kpc] Galactocentric distance of the cluster. For eccentric orbits, consider it as a(1 - e) where a is the semi-major axis, e is eccentricity.
        self.Z = None  # Metallicity of the cluster, where Z=0.02 is solar metallicity
        self.FeH = None  # Metallicity of the cluster, where FeH=0 is solar metallicity

        self.Zsolar = 0.02  # Solar metallicity.

        self.verbose = True  # If True, print extra output.
        self.seed = None  # Seed for random number generator. Use None to get a random seed.
        self.output_dataframe = True  # If True, self.mergers is a pandas dataframe, otherwise a list
        self.SN_model = "rapid"  # Model for the supernova explosion ["rapid" or "delay"]
        self.simple_remnant_model = True  # If True, use a simplified GW remnant model. If False, use the full NR GW remnant model.
        self.compute_mergers = True  # If true, compute the mergers. If false, only compute the cluster model.

        self.dtout = None  # [Myr] Time step for integration. If None, computations are faster.
        self.dense_output = True  # Condition for solve_ivp to give continuous solutions. Facilitates interpolation. Can be used instead of specifying t_eval. Necessary if compute_mergers==True.
        self.BH = True  # Condition to work with BHs in clusterBH.
        self.ssp = True  # Condition to use the SSP tools to extract the BHMF at any moment. Default option uses such tools.
        self.kick = True  # Condition to include natal kicks. Affects the BH population obtained. Default option considers kicks.

        # The user can either specify the initial average mass beforehand, or it is extracted from the IMF.
        # If m0 is not None, the IMF must be well defined so that it matches.
        self.m0 = None  # [Msun] Average mass obtained for this particular IMF.

        # Pass in lists for a_slopes, m_breaks and nbins. Any number of IMF intervals, based on size of lists. m_breaks must be size N + 1.

        # Slopes of mass function.
        self.a_slopes = [-1.3, -2.3, -2.3]
        self.a_kroupa = self.a_slopes.copy()  # In case the user inserts a different set of slopes (and mass breaks), the stellar mass loss rate ν (and time tsev) is different. If ssp is True, this is acoounted, if not, bottom-light IMFs are studied with sev_tune=True.

        # Mass function break masses.
        self.m_breaks = [0.08, 0.5, 1, 150]
        self.m_kroupa = self.m_breaks.copy()  # The user can insert different masses. The slopes and mass breaks of the Kroupa should not change.

        # Number of bins per interval of the mass function.
        self.nbins = [5, 5, 20]

        # Read the user's input parameters
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        self._construct_input_params()

        # Add the unchanged defaults to kwargs, to then pass it to ClusterBH
        for attr, value in self.__dict__.items():
            if attr[0] != "_":
                kwargs[attr] = value

        # Output
        self.cluster, self.mergers = None, None
        self.mIMBH, self.chiIMBH, self.genIMBH = None, None, None  # Properties of the most massive BH in the cluster.
        self.mIMBHej, self.chiIMBHej, self.genIMBHej = None, None, None  # Properties of the most massive BH ejected from the cluster.
        self.bhv = None

        self._run(**kwargs)

    def _construct_input_params(self):
        # Construct the mass and IMF
        if (self.M0 is None) == (self.N is None):
            raise ValueError("Please provide either M0 or N. "
                             "To change the mean mass, modify the IMF breaks and limits (m_break) or slopes (a_slope)")
        if self.m0 is None:
            self.m0 = cluster.initial_average_mass(self.a_slopes, self.m_breaks)
        if self.M0 is None:
            self.M0 = self.m0 * self.N
        else:
            self.N = self.M0 / self.m0
        self.mmin = self.m_breaks[0]
        self.mmax = self.m_breaks[-1]

        # Construct the half-mass radius and density
        if (self.rh0 is None) == (self.rhoh0 is None):
            raise ValueError("Please provide either rh0 or rhoh0.")
        if self.rhoh0 is None:
            self.rhoh0 = 3 * self.M0 / (8 * np.pi * self.rh0 ** 3)
        else:
            self.rh0 = (3 * self.M0 / (8 * np.pi * self.rhoh0)) ** (1 / 3)

        # Construct the metallicity
        if (self.Z is None) == (self.FeH is None):
            raise ValueError("Please provide either Z or FeH.")
        if self.Z is None:
            self.Z = self.Zsolar * 10 ** self.FeH
        else:
            self.FeH = np.log10(self.Z / self.Zsolar)

        # General sanity checks
        assert self.rh0 > 0
        assert self.rhoh0 > 0
        assert self.M0 > 0
        assert self.N > 0
        if self.compute_mergers:
            assert self.ssp, "Mergers are only available when SSP is enabled"
            assert self.BH, "Mergers are only available when BHs are enabled"
            assert self.dense_output, "Mergers are only available when dense_output is enabled"

    def _run(self, **kwargs):
        if self.compute_mergers:
            try:
                self.mergers = self._run_mergers(self, **kwargs)  # TODO: allow output to file like in clusterBH
                self._get_IMBH_data(self)
            except Exception as err:
                print("Error in model with", flush=True)
                print(f"\t Mass = {self.M0} M_sun", flush=True)
                print(f"\t Metallicity = {self.Z}", flush=True)
                print(f"\t Final time = {self.tend} Myr", flush=True)
                print(f"\t Initial density = {self.rhoh0:.3g} M_sun/pc^3", flush=True)
                print(f"\t SN model = {self.SN_model}", flush=True)
                print(f"\t Seed = {self.seed}", flush=True)
                print(err, flush=True)
                raise err

        else:
            self.cluster = cluster.Cluster(**kwargs)
