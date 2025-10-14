from __future__ import division
import numpy
from numpy import log, sqrt, pi, log10, exp, tanh, arctan, arctanh
from scipy.special import erf, hyp2f1, spence, gammainc, gammaincc, gamma, beta, betainc
from scipy.integrate import solve_ivp, cumulative_trapezoid, quad
from scipy.optimize import fsolve
import warnings
from scipy.integrate import IntegrationWarning
warnings.filterwarnings("ignore", category=IntegrationWarning)
numpy.seterr(invalid='ignore', divide='ignore')
warnings.simplefilter("ignore", category=RuntimeWarning)

"""
- The current parameters only work with the default settings. If a different model is chosen, a different set of values may be needed.
- The default option requires the Simple Stellar Population package. To install it, run 'pip install astro-ssptools==2.0.1'. Version 2.0.1 uses Zsolar=0.02. If not, simply state ssp=False in kwargs before a run.
  For more details, visit SMU-clusters/ssptools. 
"""

class clusterBH:
    def __init__(self, N, rhoh, **kwargs):
        """
        Initialize the star cluster model.

        Parameters:
        - N (int): Initial number of stars.
        - rhoh (float): Half-mass density [Msun/pc^3].
        - kwargs: Additional parameters to override defaults.
        """
        
        # Physical constants.
        self.G  = 0.004499 # [pc^3 /Msun /Myr^2] Gravitational constant. It is 0.004302 in [(km/s)**2 pc/Msun].
        self.Zsolar = 0.02 # Solar metallicity.
        self.ym = 1 # Exponent to be used in the future for properly dissolving clusters. Affects xi.
        
        # Cluster ICs.
        self.N = N # Initial number of stars.
        self.m0 = None # [Msun] Initial average mass. If unspecified, it is computed directly from the IMF. The computation is skipped if the user inserts it as argument.
        self.fc = 1 # Factor for escape velocity. It is different for other King models. By default, W0 is set to 5 (normalized) and scales roughly as sqrt(W0/5).
        self.rg = 8 # [kpc] Galactocentric distance of the cluster. For eccentric orbits, consider it as a(1 - e) where a is the semi-major axis, e is eccentricity.
        self.Z = 0.0002 # Metalicity of the cluster. Default to low-metallicity. 
        self.omega0 = 0 # [1 / Myrs] Angular frequency of cluster due to rotation.
        
        # BHMF. It is needed only if the user does not use the available Simple Stellar Population (SSP) tools.
        # The current treatment for the BHMF is a simple power-law. At each iteration after core collapse, the BHMF is updated. The upper mass decreases with time, as dictated by Mbh.
        self.f0 = 0.06 # Initial BH fraction. Decreases with metallicity and is important only when the SSP are not used.
        self.mlo = 3 # [Msun] Lowest BH mass in the BHMF.
        self.mup = 30 # [Msun] Upper BH mass in the BHMF
        self.mb = 0. # [Msun] Lowest mass unaffected by kicks. Default option is for no kicks. If kicks are activated, it is computed subsequently.
        self.alpha_BH = 0.5 # Slope in the BHMF. If kicks are considered, the slope cannot be equal to -5 - 3k, where k is a positive integer. The special value of -2 (k=-1) is considered, but other values are not. Can be used without kicks.
        self.fretm = 1 # Retention fraction of BHs. The default value neglects kicks. Changes the initial BH mass only trhough the prescription implemented in this script.
        self.t_bhcreation = 8 # [Myrs] Time required to create all BHs.
        self.N_points = 500 # Number of points used for even spacing.
        
        # Model parameters. 
        self.mns = 1.4 # [Msun] Mass of Neutron Stars (NS).
        self.mst_inf = 1.4 # [Msun] Maximum upper stellar mass at infinity. Serves as the upper boundary for the average stellar mass. Default value to stellar remnants, subject to change for IMFs that produce heavy stars.
        self.sigmans = 265 # [km/s] Velocity dispersion for NS. 
        self.gamma = 0.02 # Parameter for Coulomb logarithm.
        self.x = 3./4 # Exponent used for finite escape time from Lagrange points. Introduces energy dependece on evaporation time scale.
        self.r = 0.8 # Prefactor for the energy of the cluster.
        self.tcross_index = 2 * sqrt(2 * pi / 3) # Prefactor for half-mass crossing time defined as tcr = 1/sqrt(G ρ).
        self.c_bh = 10 # Exponent for tides used in the BH evaporation rate. Currently deactivated. Used for describing the impact of the tidal field on the BH population. 
        self.rp = 0.8 # [kpc] Constant parameter for galactic potentials. Serves as an effective scale.
        self.rpc = 1 # [pc] Constant parameter for cluster potentials. Also an effective scale.
        self.Rmaxc = 1e4 # [pc] Maximum distance used for the potential of a cluster. Inserted for numerical purposes for models with infinite mass if the used selects such model.
        self.gammapc = 2.5 # Exponent that can be used in cluster profiles.
        self.gammapc1 = 1 # Exponent used for cluster model potential.
        self.gammapc2 = 1 # Exponent used for cluster model potential. Combined with the other 2.
        self.Vcc = 10 # [km / s] Velocity that can be used in cluster profiles.
        self.fmin = 0 # Constant evaporation rate. Can be used in order to have a constant evaporation rate for cases where the ratio rh / rt is not strong enough. Currently deactivated.
        self.Scrit = 0 # Constant value below which Spitzer's parameter is kept fixed. Facilitates ejection of last few BHs if a varying ejection rate is selected.
        
        # This part includes parameters that are on trial, intended for future additions. Some may be subjects to future fittings.
        #_________________________________________________
        self.gamma1 = 3 / 2 # Exponent for the evaporation rate in the rotational component. Default value keeps the ratio of xi over trh constant, with changes affecting only the evolution of rh.
        self.gamma2 = 1 # Parameter used for describing the BH ejection rate. Applies to extended options.
        self.gamma3 = 3 # Constant used in the balance function. Currently deactivated.
        self.gamma4 = 2 # Second constant for the balance function, working with gamma3 to smoothly connect the two phases.
        self.c1 = 0.1 # Parameter for stellar evolution. Can be used for a age dependent ν.
        self.c2 = 0.03 # Second parameter for stellar evolution. 
        self.d1 = 0.18 # First exponent to describe the evolution of parameter fc. Can be used to specify how W0 evolves for a given cluster as function fo M and rh.
        self.d2 = 0.14 # Second exponent to describe the evolution of parameter fc.
        self.chi_ev = 0. # Prefactor that adjusts the average stellar mass with respect to tides, important for small galactocentric distances.
        self.chi_ej = 0 # Prefactor used for decreasing the average stellar mass due to ejections.
        self.chi_gmc = 0 # Prefactor used for increasing the average stellar mass from interactions with GMC if chosen.
        self.chi_disk = 0 # Prefactor used for increasing the average stellar mass due to disk shocks. Light stars in the outskirts should leave first. 
        self.find_max = 0.8 # Maximum index for induced mass loss.
        self.Rht_crit = 0.3 # Critical ratio of rh /rt for induced mass loss. It describes Rv filling clusters.
        self.n_delay = 3 # Delay factor for induced mass loss. Connects delay time with crossing time.
        self.k_bh = 1 # Parameter used for properly extracting average mass that is evaporated. Can be used for large BH fractions.
        self.gamma_n = 0.62 # Exponent for interaction of the cluster with GMCs.
        self.Sigma_n = 170 # [Msun / pc^2] Surface density in the solar neighborhood. Used for interactions with GMCs.
        self.rho_n = 0.03 # [Msun / pc^3] Total density of GMC in the solar neighborhood.
        self.Sigma_disk0 = 10 ** 3 # [Msun / pc^2] Central surface density of disk.
        self.Rd = 2.65 # [kpc] Length scale for the disk. Surface density decreases exponentially.
        self.xi_shock = 0.01 # Parameter used for disk shocks to model mass loss rate. This is on top of evaporation which is enhanced from disk shockings that provide energy (increase rh). 
        self.Sseg = 0 # Initial degree of segregation. Values range from [0,1] with the bounds marking homologous distribution and fully segregated cluster.
        self.aseg = 3 / 2 # Exponent used for core collapse, should the initial degree of segregation is nonzero. 
        
        #_________________________________________________
        
        # Parameters fitted to N-body / Monte Carlo models.
        self.zeta = 0.0981 # Energy loss per half-mass relaxation. It is assumed to be constant.
        self.beta = 0.06 # BH ejection rate from the core per relaxation, considered constant for all clusters, assuming a large BH population in the core, regardless of initial conditions. Shows the efficiency of ejections. Subjects to change in the following functions.
        self.nu = 0.073 # Mass loss rate of stars due to stellar evolution. It is the same for all clusters and does not depend on metallicity. Changes can be applied in the following functions.
        self.tsev = 1.19 # [Myrs]. Time instance when stars start evolving. Typically it should be a few Myrs.
        self.a0 = 1 # Fix zeroth order in ψ, if approximations are made. In reality, it serves as the contribution of the stellar population to the half-mass mass spectrum.
        self.a11 = 2 # First prefactor in ψ. Relates the BH mass fraction within the half-mass to the total fraction.
        self.a12 = 0.5769 # Second prefactor in ψ. Relates the mass ratio of BHs over the total average mass, within the half-mass radius and the total. It is the product of a11 and a12 that matters in the following computations.
        self.a3 = 1 # [Msun] Prefactor of average mass within the half-mass radius compared to the total. The values of a11, a12 and a3 are taken for clusters with large initial relaxations.
        self.n = 3.34 # Exponent in the power-law parametrization of tides. It determines how quickly the Virial radii lose mass. Its value relative to 1.5 indicates the rate at which mass is lost.
        self.c = 1 / 0.0555 # Parameter for exponential tides. Can be used for other tidal models available.
        self.Rht = 0.1195 # Ratio of rh/rt, used to calculate the correct mass loss rate due to tides. It serves as a reference scale, not necessarily representing the final value of rh/rt.
        self.ntrh = 0.5937 # Number of initial relaxations to compute core collapse instance. Marks also the transition from the unbalanced to the balanced phase.
        self.alpha_ci = 0.0 # Initial ejection rate of stars, when BHs are present. Currently deactivated, as it only accounts for a small percentage of the total stellar mass loss each relaxation.
        self.alpha_cf = 0.0 # Final ejection rate of stars, when all BHs have been ejected (or the final binary remains) and the stellar density increases in the core. Represented by different values, but in principle they may be equal.
        self.kin = 1 # Kinetic term of evaporating stars. Used to compute the energy carried out by evaporated stars. The deafult value neglects this contribution. The case of a varying value is available in later parts.
        self.b = 0.25 # Exponent for parameter ψ. The choices are between 2 and 2.5, the former indicating equal velocity dispersion between components while the latter complete equipartition.
        self.b_min = 0 # Minimum exponent for parameter ψ. Can be used when the exponent in ψ is varying with the BH fraction.
        self.b_max = 0.5 # Maximum exponent for parameter ψ. Taken from Wang (2020), can be used for the case of a running exponent b.
        self.b0 = 2 # Exponent for parameter ψ. It relates the fraction mst / m (average stelar mass over average mass) within rh and the total fraction.
        self.b1 = 1 # Correction to exponent of fbh in parameter ψ. It appears because the properties within the half-mass radius differ.
        self.b2 = 1.0 # Exponent of fraction mbh / m (average BH mass over average mass) in ψ. Connects properties within rh with the global. Current value is used for sparse clusters.
        self.b3 = 1.0 # Exponent of average mass within rh compared to the global average mass. All exponents are considered for the case of clusters with large relaxation.
        self.b4 = 0.17 # Exponent for the BH ejection rate. Participates after a critical value, here denoted as fbh_crit.
        self.b5 = 0 # 0.4 # Second exponent for the BH ejection rate. Participates after a critical value for the BH fraction, here denoted as qbh_crit. It is deactivated for now.
        self.Mval0 = 3. # Initial value for mass segregation. For a homologous distribution of stars, set it equal to 3.
        self.Mvalf = 3. # Final parameter for mass segregation. Describes the maximum value for the level of segregation. It is universal for all clusters and appears at the moment of core collapse. Currently deactivated.
        self.p = 0.1 # Parameter for finite time stellar evaporation from the Lagrange points. Relates escape time with relaxation and crossing time. 
        self.fbh_crit = 0.005 # Critical value of the BH fraction to use in the ejection rate of BHs. Decreases the fractions E / M φ0 properly, which for BH fractions close/above O(1)% approximately can be treated as constant.
        self.qbh_crit = 25 # Ratio of mbh / m when the BH ejection rate starts decreasing. Also for the ratio E / M φ0, however now it is deactivated. Should introduce an effective metallicity dependence.
        self.S0 = 0.6881 # Parameter used for describing BH ejections when close to equipartition. Useful for describing clusters with different metallicities using the same set of parameters, as well as for clusters with small BH populations which inevitably slow down the ejections to some extend.
        self.gamma_exp = 10 # Parameter used for obtaining the correct exponent for parameter ψ as a function of the BH fraction. Uses an exponential fit to connect minimum and maximum values of b.
        self.cg = 0 # Constant used for the contribution of the tidal field on the energy of the cluster.
        
        # Integration parameters.
        self.tend = 13.8e3 # [Myrs] Final time instance for inegration. Here taken to be a Hubble time.
        self.dtout = 2 # [Myrs] Time step for integration. If None, computations are faster. 
        self.Mst_min = 100 # [Msun] Stop criterion for stars. Below this value, the integration stops.
        self.Mbh_min = 550 # [Msun] Stop criterion for BHs. Below this value the integration stops. The integrator stops only if both the stellar option is selected and the specified conditions are met.
        self.integration_method = "RK45" # Integration method. Default to a Runge-Kutta.
        self.vectorize = False # Condition for solve_ivp. Solves faster coupled differential equations when the derivatives are arrays.
        self.dense_output = True # Condition for solve_ivp to give continuous solution. Facilitates interpolation. Can be used instead of specifying t_eval.
        self.rtol, self.atol = 1e-6, 1e-7 # Relative and absolute tolerance for integration.
        
        # Output.
        self.output = False # A Boolean parameter to save the results of integration along with a few important quantities.
        self.outfile = "cluster.txt" # File to save the results, if needed.

        # Conditions.
        self.ssp = True # Condition to use the SSP tools to extract the BHMF at any moment. Default option uses such tools. 
        self.sev = True # Condition to consider effects of stellar evolution. Affects total mass and expansion of the cluster. Default option considers stellar evolution.
        self.sev_tune = True # Condition to consider changes in the sev parameters should the user insert an IMF different than a Kroupa. Default option to True. If false, the user can insert a different IMF with different set of parameters.
        self.kick = True # Condition to include natal kicks. Affects the BH population obtained. Default option considers kicks.
        self.tidal = True # Condition to activate tides. Default option considers the effect of tides.
        self.rt_approx = True # Condition to use approximate expression for tidal radius. If deactivated, the tidal radius for a circular orbit is extracted numerically for a given cluster model.
        self.escapers = False # Condition for escapers to carry negative energy as they evaporate (only) from the cluster due to the tidal field. Affects mainly the expansion of sparse clusters at small galactocentric distances. Default option is deactivated.
        self.varying_kin = False # Condition to include a variation of the kinetic term with the number of stars. It is combined with 'escapers'. Default option is deactivated.
        self.two_relaxations = True # Condition to have two relaxation time scales for the two components. Differentiates between ejections and evaporation. Default option is activated.
        self.mass_segregation = False # Condition for mass segregation. If activated, parameter Mval evolves from Mval0 up until Mvalf within one relaxation. Default option is deactivated.
        self.sigma_exponent_run = False # Condition to have a running exponent in velocity dispersion based on the BH fraction. Default option is deactivated.
        self.finite_escape_time = False # Condition to implement escape from Lagrange points for stars which evaporate. Introduces an additional dependence on the number of particles on the tidal field, prolongs the lifetime of clusters. Default option is deactivated.
        self.running_bh_ejection_rate_1 = True # Condition for decreasing the ejection rate of BHs due to varying E / M φ0. Default option considers the effect of decreasing fbh only. Currently activated.
        self.running_bh_ejection_rate_2 = True # Condition for decreasing the ejection rate of BHs due to equipartition. Uses a smooth function for decreasing the BH ejection efficiency. Currently activated.
        self.running_stellar_ejection_rate = False # Condition for changing the stellar ejection rate compared to the tidal field. Currently deactivated.
        self.rotation = False # Condition to describe rotating clusters. Currently it is inserted as a simple extension, more study is required. It is deactivated.
        self.dark_clusters = False # Condition for describing dark clusters (stellar population depletes first). When used, the approximations in ψ are not used. It is regarded as future extension and needs to be augmented. Currently it is deactivated
        self.induced_loss = False # Condition to study induced mass loss for RV filling star clusters. Currenly deactivated, surves as a future extension.
        self.GMC = False # Condition to allow for interaction with Giant Molecular Clusters.
        self.disk = False # Condition to include shocks from the disk.
        self.tidal_spiralling = False # Condition to assume tidal spiraling of the cluster.
        
        # Stellar evolution model. Used for selecting a model for ν.
        self.sev_model = 'constant' # Default option is constant mass loss rate.
        
        # Motion of cluster.
        self.Vc = 220. # [km/s] Circular velocity of cluster. Used for singular isothermal sphere (SIS). 
        self.X0 = 1 # Ratio V^2 / (2 sigma^2) for Point mass galaxy and isotropic profile. Can be used for tidal spiraling should the user select a Point mass galaxy.
        self.rhoG = 0.1 # [Msun / pc^3] Density for the case of Point mass galaxy. Can be used for tidal spiraling.
        self.Rmax = 100 # [kpc] Maximum radius in the model. Can be used in NFW models to specify the density.
        self.gammap = 2.5 # Exponent that can be used for galactic models involving power-laws.
        self.gammap1 = 4 # Exponent used for galactic model. Second exponent for Zhao.
        self.gammap2 = 1 # Exponent used for galactic model. Third exponent for Zhao.
       
        # Galactic model. Used for specifying the tidal radius.
        self.galactic_model, self.rt_index = 'SIS', 2 # Default to 'SIS'. Affects the tidal radius.
        
        # Tidal model. Used for specifying the evaporation rate ξ.
        self.tidal_model = 'Exponential' # Default to 'Exponential'. Affects tidal mass loss and thus the evolution of the half-mass radius.
        
        # Cluster model. Used for specifying the potential of the cluster.
        self.cluster_model = 'Point_mass' # Default to 'Point_mass'. Currently it affects only the energy contribution of evaporating stars, if activated.
        
        # Model for the BH ejection rate β.
        self.beta_model = 'exponential' # Default option to the exponential model. Affects the dependence of the BH ejection rate on Spitzer's parameter and thus the extrapolation to small BH populations or large metallicity clusters.
        
        # Balance model. It can be used for a smooth connection of the unbalanced phace with post core collapse.
        self.balance_model = 'step' # Default option to heaviside. Uses a function in order to smoothly connect the unbalanced phase to post core collapse. Other options can make the differential equations continuous.
        
        # Pass in lists for a_slopes, m_breaks and nbins. Any number of IMF intervals, based on size of lists. m_breaks must be size N+1
       
        # Slopes of mass function.
        self.a_slopes = [-1.3, -2.3, -2.3]
        self.a_kroupa = self.a_slopes.copy() # In case the user inserts a different set of slopes (and mass breaks), the stellar mass loss rate ν is different.

        # Mass function break masses.
        self.m_breaks = [0.08, 0.5, 1., 150.]
        self.m_kroupa = self.m_breaks.copy() # The user can insert different masses. The slopes and mass breaks of the Kroupa should not change.

        # Number of bins per interval of the mass function.
        self.nbins = [5, 5, 20]

        # All other arguments to be passed to InitialBHPopulation, including IFMR and natal kick options. See SSPtools documentation for details.
        self.ibh_kwargs = dict()
        
        # Check input parameters, then proceed with the computations.
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        self.nu_factor = 0 # Factor that corrects solution for Mst, mst for the case of a different IMF that has similar upper part with Kroupa. Default to 0.
        # Check that this is not exactly a Kroupa IMF.
        if (not (numpy.array_equal(self.a_slopes, self.a_kroupa) and numpy.array_equal(self.m_breaks, self.m_kroupa))) and self.sev_tune and self.sev:

           # Check if IMF high mass slope/breaks match Kroupa above 1Msun
           if ((self.a_slopes[-1] == self.a_kroupa[-1]) and numpy.array_equal(self.m_breaks[-2:], self.m_kroupa[-2:])):

               self.nu_factor = 1 - self._sev_factor(self.a_kroupa, self.a_slopes, self.m_kroupa, self.m_breaks) # Approximate expression for auxiliary factor. It is inserted in the differential equations for Mst, mst.
            
           # If the upper part of the IMF changes, tsev needs to change as well.
           else:
               warnings.warn('IMF slope > 1Msun is different than a Kroupa IMF, new stellar evolution rate cannot be automatically computed. '
                             f'Using given {self.nu=} and {self.tsev=}.')
               
        # Define models for stellar evolution. Coefficients can capture different dependences on metallicity, or an explicit dependence can be inserted.
        self.sev_dict={
            'constant': lambda Z, t: self.nu, # M(t) is a power-law.
            'power_law': lambda Z, t: self.nu * (t / self.tsev) ** (- self.c1), # M(t) is exponential. nu=0.087, tsev=1.49, c1=0.046 agree with a Kroupa IMF.
            'exponential': lambda Z, t: self.nu * exp(-self.c1 * t / self.tsev), # M(t) depends on exponential integral Ei.
            'logarithmic': lambda Z, t: self.nu * (1 + self.c1 * log(t / self.tsev) - self.c2 * log(t / self.tsev) ** 2) # M(t) is power-law products. nu=0.079, tsev=1.4, c1=0.004, c2=0.003 seem to work for a Kroupa IMF 
        }
        
        # Define the tidal models dictionary.
        self.tidal_models = { # Both the half-mass and the tidal radius should be in [pc]. All the rest parameters are fixed. In a different scenario, they should be treated as variables.
            'Power_Law': lambda rh, rt: 3 * self.zeta / 5 * (rh / rt / self.Rht) ** self.n, # Simple insertion. The case of n=1.5 suggests \dot Mst is independend of half-mass radius.
            'Exponential': lambda rh, rt: 3 * self.zeta / 5 * self.Rht * exp(self.c * rh / rt), # General formula. The evaporation rate is constant for dense clusters.
            'Constant': lambda rh, rt: self.Rht # Constant evaporation rate.
        }
        
        # Galactic model dictionary. 
        self.galactic_model_dict = { 
            # Dictionary for spherically symmetric galactic potentials. The index is used for the tidal radius only and it is dimensionless. The potential has units are [pc^2 / Myr^2]. 
            # Derivative of the potential is [pc / Myrs^2] and is used to specify the velocity profile. X2 is the ratio of the velocity profile squared over twice the velocity dispersion squared for isotropic models, currently used for tidal spiraling only. Density is in [Msun / pc^3] and is used in tidal spiraling.
            # Distance r is inserted in kpc everywhere. X2 is valid only for a Maxwellian distribution. Current treatment does not allow for combinations at different ranges of the galactocentric distance.
            'SIS': {
                'rt_index': lambda r: 2,
                'Phi':lambda r: (1.023 * self.Vc) ** 2 * log(r / self.Rmax),
                'dPhi_dr': lambda r: 1e-3 * (1.023 * self.Vc) ** 2 / r,
                'd2Phi_dr2': lambda r: - 1e-6 * (1.023 * self.Vc) ** 2 / r ** 2,
                'X2': lambda r: 1,
                'rho': lambda r: 1e-6 * (1.023 * self.Vc) ** 2 / (4 * pi * self.G * r ** 2)
                }, # Singular isothermal sphere. It is the default.
            
            'Point_mass': {
                'rt_index':lambda r: 3,
                'Phi': lambda r: - 1e-3 * self.G * self.Mg / r,
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / r ** 2,
                'd2Phi_dr2': lambda r: - 2e-9 * self.G * self.Mg / r ** 3,
                'X2': lambda r: self.X0,
                'rho': lambda r: self.rhoG
                }, # Point mass galaxy.
            
            'Hernquist': {
                'rt_index':lambda r: (3 * r + self.rp) / (r + self.rp), 
                'Phi': lambda r: - 1e-3 * self.G * self.Mg / (r + self.rp),
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / (r + self.rp) ** 2,
                'd2Phi_dr2': lambda r: - 2e-9 * self.G * self.Mg / (r + self.rp) ** 3,
                'X2': lambda r: 6 * r / self.rp / ((1 + r / self.rp) ** 2 * (12 * r / self.rp * (1 + r / self.rp) ** 3 * log(1 + self.rp / r) - r / (r + self.rp) * (25 + 52 * r / self.rp + 42 * (r / self.rp) ** 2 + 12 * (r / self.rp) ** 3))),
                'rho': lambda r: 1e-9 * self.Mg / (2 * pi * self.rp ** 2) * 1 / (r * (1 + r / self.rp) ** 3)
                }, # Hernquist model.
            
            'Plummer': {
                'rt_index':lambda r: 3 * r ** 2 / (r ** 2 + self.rp ** 2),
                'Phi':lambda r: - 1e-3 * self.G * self.Mg / sqrt(r ** 2 + self.rp ** 2),
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg * r / (r ** 2 + self.rp ** 2) ** (3 / 2),
                'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg * (2 * r ** 2 - self.rp ** 2) / (r ** 2 + self.rp ** 2) ** (5 / 2),
                'X2': lambda r: 3 * (r / self.rp) ** 2 / (1 + (r / self.rp) ** 2), 
                'rho': lambda r: 1e-9 * 3 * self.Mg * self.rp ** 2 / (4 * pi * (self.rp ** 2 + r ** 2) ** (5 / 2))
                }, # Plummer model.
            
            'Jaffe': {
                'rt_index':lambda r: (3 * r + 2 * self.rp) / (r + self.rp),
                'Phi':lambda r: - 1e-3 * self.G * self.Mg / self.rp * log(1 + self.rp / r),
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / (r * (r + self.rp)),
                'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg *  (self.rp + 2 * r) / (r ** 2 * (self.rp + r) ** 2),
                'X2': lambda r: ((1 + r / self.rp) * (12 * (r / self.rp) ** 2 * (1 + r / self.rp) ** 2 * log(1 + self.rp / r) - 12 * (r / self.rp) ** 3 - 18 * (r / self.rp) ** 2 - 4 * r / self.rp + 1)) ** (-1),
                'rho': lambda r: 1e-9 * self.Mg * self.rp / (4 * pi * r ** 2 * (r + self.rp) ** 2)
                }, # Jaffe model.
            
            'NFW': {
                'rt_index':lambda r: 1 + (2 * log(1 + r / self.rp) - 2 * r / (r + self.rp) - (r / (r + self.rp)) ** 2 ) / (log(1 + r / self.rp) - r / (r + self.rp)),
                'Phi':lambda r: - 1e-3 * self.G * self.Mg / r * log(1 + r / self.rp) / (log(1 + self.Rmax / self.rp) - self.Rmax / (self.rp + self.Rmax)),
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg * (log(1 + r /  self.rp) - r / (r + self.rp)) / r ** 2 / (log(1 + self.Rmax / self.rp) - self.Rmax / (self.rp + self.Rmax)),
                'd2Phi_dr2': lambda r: - 1e-9 * 2 * self.G * self.Mg / r ** 3 * (log(1 + r / self.rp) - r / (r + self.rp) - 0.5 * (r / (r + self.rp)) ** 2) / (log(1 + self.Rmax / self.rp) - self.Rmax / (self.rp + self.Rmax)),
                'X2': lambda r: (log(1 + r / self.rp) - r / (r + self.rp)) / ((r / self.rp) ** 2 * (1 + r / self.rp) ** 2 * (pi ** 2 + log(1 + self.rp / r) + 3 * log(1 + r / self.rp) ** 2 + 6 * spence(1 + r / self.rp)) + (1 + r / self.rp) ** 2 * log(1 + r / self.rp) - r / self.rp * (1 + 9 * r / self.rp + 7 * (r / self.rp) ** 2 + 2 * log(1 + r / self.rp) * (3 * (r / self.rp) ** 2 + 5 * r / self.rp + 2))),
                'rho': lambda r: 1e-9 / (r * (r + self.rp) ** 2) * self.Mg / (4 * pi * (log(1 + self.Rmax / self.rp) - self.Rmax / (self.rp + self.Rmax)))
                }, # Navarro-Frenk-White model. At infinity it diverges so a truncation is inserted at Rmax.
            
            'Isochrone': {
                'rt_index': lambda r: 1 + 1 / (self.rp + sqrt(r ** 2 + self.rp ** 2)) / sqrt(r ** 2 + self.rp ** 2) * (2 * r ** 2 - self.rp ** 2 * (self.rp + sqrt(r ** 2 + self.rp ** 2)) / sqrt(self.rp ** 2 + r ** 2)),
                'Phi':lambda r: - 1e-3 * self.G * self.Mg / (self.rp + sqrt(r ** 2 + self.rp ** 2)),
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg * r / ((self.rp + sqrt(self.rp ** 2 + r ** 2)) ** 2 * sqrt(self.rp ** 2 + r ** 2)), 
                'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg * ((2 * r ** 2 - self.rp ** 2) * sqrt(r ** 2 + self.rp ** 2) - self.rp ** 3) / ((self.rp + sqrt(self.rp ** 2 + r ** 2)) ** 3 * (self.rp ** 2 + r ** 2) ** (3 / 2)),
                'X2': lambda r: (r / self.rp) ** 2 * (1 + 2 / 3 * (r / self.rp) ** 2 + sqrt(1 + (r / self.rp) ** 2)) / (2 * (1 + (r / self.rp) ** 2) ** 2 * (1 + sqrt(1 + (r / self.rp) ** 2)) ** 5 * (2 / 3 * log(1 + sqrt(1 + (r / self.rp) ** 2)) - 1 / 3 * log(1 + (r / self.rp) ** 2) + 1 / 6 * 1 / (1 + sqrt(1 + (r / self.rp) ** 2)) ** 2 + 1 / 9 * 1 / (1 + sqrt(1 + (r / self.rp) ** 2)) ** 3 - 2 / 3 / sqrt(1 + (r / self.rp) ** 2) + 1 / 6 / (1 + (r / self.rp) ** 2))), 
                'rho': lambda r: 1e-9 * 3 * self.Mg * (1 + 2 / 3 * (r / self.rp) ** 2 + sqrt(1 + (r / self.rp) ** 2)) / (4 * pi * self.rp ** 3 * (1 + (r / self.rp) ** 2) ** (3 / 2) * (1 + sqrt(1 + (r / self.rp) ** 2)) ** 3)
                }, # Isochrone potential.
            
            'Dehnen': {
                'rt_index': lambda r: (3 * r + self.gammap * self.rp) / (r + self.rp),
                'Phi': lambda r: - 1e-3 * self.G * self.Mg / (self.rp * (2 - self.gammap)) * (1 - (r / (r + self.rp)) ** (2 - self.gammap)),
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg * r ** (1 - self.gammap) / (r + self.rp) ** (3 - self.gammap),
                'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg / (r ** self.gammap * (r + self.rp) ** (4 - self.gammap)) * (2 * r + (self.gammap - 1) * self.rp),
                'X2': lambda r: 5 / (2 * hyp2f1(1, 7 - 2 * self.gammap, 6, 1 / (1 + r / self.rp))),
                'rho': lambda r: (3 - self.gammap) * 1e-9 * self.Mg * self.rp / (4 * pi * r ** self.gammap * (r + self.rp) ** (4 - self.gammap))
                }, # Dehnen model. Generalization of the Jaffe and Hernquist. Because of the form, it is finite as infinity. Avoid values for gammap equal to (7 + k) / 2 where k is integer.
            
            'Veltmann': {
                'rt_index': lambda r: (3 * r ** self.gammap + (2 - self.gammap) * self.rp ** self.gammap) / (r ** self.gammap + self.rp ** self.gammap),
                'Phi': lambda r: - 1e-3 * self.G * self.Mg / (r ** self.gammap + self.rp ** self.gammap) ** (1 / self.gammap),
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg * r ** (self.gammap - 1) / (r ** self.gammap + self.rp ** self.gammap) ** (1 + 1 / self.gammap),
                'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg * r ** (self.gammap - 2) * (2 * r ** self.gammap + (1 - self.gammap) * self.rp ** self.gammap) / (r ** self.gammap + self.rp ** self.gammap) ** (2 + 1 / self.gammap),
                'X2': lambda r: (4 + self.gammap) / (2 * hyp2f1(1, 3 + 2 / self.gammap, 2 * (1 + 2 / self.gammap), 1 / (1 + (r / self.rp) ** self.gammap))),
                'rho': lambda r: 1e-9 * (1 + self.gammap) * self.Mg * self.rp ** self.gammap / (4 * pi * r ** (2 - self.gammap) * (self.rp ** self.gammap + r ** self.gammap) ** (2 + 1 / self.gammap))
                }, # Veltmann model. Generalization of the Hernquist and Plummer models. It is finite as infinity.
            
            'Soft': {
                'rt_index': lambda r: (3 * r + 8 * self.rp) / (r + 2 * self.rp),
                'Phi': lambda r: - 1e-3 * self.G * self.Mg / r * (1 + self.rp / r),
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / r ** 2 * (1 + 2 * self.rp / r),
                'd2Phi_dr2': lambda r: - 1e-9 * 2 * self.G * self.Mg / r ** 3 * (1 + 3 * self.rp / r),
                'X2': lambda r: (15 * (2 + r / self.rp)) / (2 * (5 + 3 * r / self.rp)),
                'rho': lambda r: 1e-9 * self.rp * self.Mg / (2 * pi * r ** 4)
                }, # Soft model.
            
            'Power_law': {
                'rt_index': lambda r: self.gammap,
                'Phi': lambda r: - 1e-3 * self.G * self.Mg / (self.rp * (self.gammap - 2)) * r ** (2 - self.gammap) * (self.rp / self.Rmax) ** (3 - self.gammap),
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg * r ** (1 - self.gammap) / self.rp ** (3 - self.gammap) * (self.rp / self.Rmax) ** (3 - self.gammap),
                'd2Phi_dr2': lambda r: - 1e-9 * (self.gammap - 1) * self.G * self.Mg / self.rp ** 3 * (r / self.rp) ** (- self.gammap) * (self.rp / self.Rmax) ** (3 - self.gammap),
                'X2': lambda r: self.gammap - 1,
                'rho': lambda r: 1e-9 * self.Mg / (4 * pi * self.rp ** 3 / (3 - self.gammap) / (self.rp / self.Rmax) ** (3 - self.gammap)) * (self.rp / r) ** self.gammap
                }, # Power-law profile. Exponent must be within [2, 3] so that it is well defined. A truncation at Rmax is needed.
            
            'MIS': {
                'rt_index': lambda r: 3 - (r / self.rp) ** 3 / (1 + (r / self.rp) ** 2) /((r / self.rp) - arctan((r / self.rp))),
                'Phi': lambda r: - 1e-3 * self.G * self.Mg / self.rp / (self.Rmax / self.rp - arctan(self.Rmax / self.rp)) * (1 / (r / self.rp) * ((r / self.rp) - arctan((r / self.rp))) - log(sqrt(1 + (r / self.rp) ** 2) / sqrt(1 + (self.Rmax / self.rp) ** 2))),
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / r ** 2 * (r / self.rp - arctan(r / self.rp)) / (self.Rmax / self.rp - arctan(self.Rmax / self.rp)),
                'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg / r ** 3 * (r / self.rp - arctan(r / self.rp)) / (self.Rmax / self.rp - arctan(self.Rmax / self.rp)) * (2 - (r / self.rp) ** 3 / (1 + (r / self.rp) ** 2) / ((r / self.rp) - arctan(r / self.rp))),
                'X2': lambda r: (r / self.rp - arctan(r / self.rp)) / ((1 + (r / self.rp) ** 2) * (r / self.rp * ( pi ** 2 / 4 - arctan(r / self.rp) ** 2) - 2 * arctan(r / self.rp))),
                'rho': lambda r: 1e-9 * self.Mg / (4 * pi * self.rp ** 3 * (self.Rmax / self.rp - arctan(self.Rmax / self.rp))) * 1 / (1 + (r / self.rp) ** 2)
                }, # Modified Isothermal Sphere.
            
            'Perfect_sphere': {
                'rt_index': lambda r: 3 - 2 * (r / self.rp) ** 3 / (1 + (r / self.rp) ** 2 ) ** 2 / (arctan(r / self.rp) - (r / self.rp) / (1 + (r / self.rp) ** 2)),
                'Phi': lambda r: - 1e-3 * self.G * self.Mg / r / (pi / 2) * arctan(r / self.rp) ,
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / r ** 2 * (arctan(r / self.rp) - (r / self.rp) / (1 + (r / self.rp) ** 2)) / (pi / 2),
                'd2Phi_dr2': lambda r: - 2e-9 * self.G * self.Mg * (arctan(r / self.rp) - (r / self.rp) / (1 + (r / self.rp) ** 2)) / (pi / 2) / r ** 3 * (1 - (r / self.rp) ** 3 / (1 + (r / self.rp) ** 2) ** 2 / (arctan(r / self.rp) - (r / self.rp) / (1 + (r / self.rp) ** 2))),
                'X2': lambda r: (arctan(r / self.rp) - (r / self.rp) / (1 + (r / self.rp) ** 2)) / ((r / self.rp) * (3 * (r / self.rp) ** 2 + 4) / 2 + 3 / 2 * (r / self.rp) * ((1 + (r / self.rp) ** 2) * arctan(r / self.rp)) ** 2 + (3 * (r / self.rp) ** 4 + 5 * (r / self.rp) ** 2 + 2) * arctan(r / self.rp) - 3 * pi ** 2 / 8 * (r / self.rp) * (1 + (r / self.rp) ** 2) ** 2),
                'rho': lambda r: 1e-9 * self.Mg / (pi ** 2 * self.rp ** 3) * 1 / (1 + (r / self.rp) ** 2) ** 2
                }, # Perfect Sphere. 
            
            'Modified_Hubble': {
                'rt_index': lambda r: 3 - (r / self.rp) ** 3 / (1 + (r / self.rp) ** 2) ** (3 / 2) / (arctanh((r / self.rp) / sqrt(1 + (r / self.rp) ** 2)) - (r / self.rp) / sqrt(1 + (r / self.rp) ** 2)),
                'Phi': lambda r: - 1e-3 * self.G * self.Mg / r * arctanh(r / self.rp / sqrt(1 + (r / self.rp) ** 2)) / (arctanh(self.Rpmax / self.rp /  sqrt(1 + (self.Rpmax / self.rp) ** 2)) - self.Rpmax / self.rp / sqrt(1 + (self.Rpmax / self.rp) ** 2)),
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / r ** 2 * (arctanh((r / self.rp) / sqrt(1 + (r / self.rp)** 2)) - (r / self.rp) / sqrt(1 + (r / self.rp) ** 2)) / (arctanh((self.Rmax / self.rp) / sqrt(1 + (self.Rmax / self.rp) ** 2)) - (self.Rmax / self.rp) / sqrt(1 + (self.Rmax / self.rp) ** 2)) ,
                'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg / r ** 3 * (arctanh((r / self.rp) / sqrt(1 + (r / self.rp)** 2)) - (r / self.rp) / sqrt(1 + (r / self.rp) ** 2)) / (arctanh((self.Rmax / self.rp) / sqrt(1 + (self.Rmax / self.rp)** 2)) - (self.Rmax / self.rp) / sqrt(1 + (self.Rmax / self.rp) ** 2)) * (2 - (r / self.rp) ** 3 / (1 + (r / self.rp) ** 2) ** (3 / 2) / (arctanh((r / self.rp) / sqrt(1 + (r / self.rp) ** 2)) - (r / self.rp) / sqrt(1 + (r / self.rp) ** 2))),
                'X2': lambda r: (arctanh((r / self.rp) / sqrt(1 + (r / self.rp) ** 2)) - (r / self.rp) / sqrt(1 + (r / self.rp) ** 2)) / ((r / self.rp) * sqrt(1 + (r / self.rp) ** 2) + (2 * (r / self.rp) ** 2 + 1) * (1 + (r / self.rp) ** 2) * log(((r / self.rp) + sqrt(1 + (r / self.rp) ** 2)) / (sqrt(1 + (r / self.rp) ** 2) - (r / self.rp))) - (r / self.rp) * (1 + (r / self.rp) ** 2) ** (3 / 2) * (4 * log(2) + 2 * log(1 + (r / self.rp) ** 2))),
                'rho': lambda r: 1e-9 * self.Mg / (4 * pi * self.rp ** 3 * (arctanh((self.Rmax / self.rp) / (sqrt(1 + (self.Rmax / self.rp) ** 2))) - (self.Rmax / self.rp) / (sqrt(1 + (self.Rmax / self.rp) ** 2)))) / (1 + (r / self.rp) ** 2) ** (3 / 2)
                }, # Hubble.
            # MIS, Perfect sphere and Hubble are just a few examples that have an analytical formula for the velocity dispersion. Other options are available in the Zhao model, however numerically.
            
            'Moore': {
                'rt_index': lambda r: 3 - (3 - self.gammap) / log(1 + (r / self.rp) ** (3 - self.gammap)) * (r / self.rp) ** (3 - self.gammap) / (1 + (r / self.rp) ** (3 - self.gammap)),
                'Phi': lambda r: - 1e-3 * self.G * self.Mg / self.rp / log(1 + (self.Rmax / self.rp) ** (3 - self.gammap)) * (self.rp / r * log(1 + (r / self.rp) ** (3 - self.gammap)) + beta(1 / (3 - self.gammap), (2 - self.gammap) / (3 - self.gammap)) * betainc(1 / (3 - self.gammap), (2 - self.gammap) / (3 - self.gammap), 1 / (1 + (r / self.rp) ** (3 - self.gammap)))),
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / r ** 2 * log(1 + (r / self.rp) ** (3 - self.gammap)) / log(1 + (self.Rmax / self.rp) ** (3 - self.gammap)),
                'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg / r ** 3 / log(1 + (self.Rmax / self.rp) ** (3 - self.gammap)) * (2 - (3 - self.gammap) / log(1 + (r / self.rp) ** (3 - self.gammap)) * (r / self.rp) ** (3 - self.gammap) / (1 + (r / self.rp) ** (3 - self.gammap))),
                'X2': numpy.vectorize(lambda r: (3 - self.gammap) * log(1 + (r / self.rp) ** (3 - self.gammap)) / (- 2 * (r / self.rp) ** (1 + self.gammap) * (1 + (r / self.rp) ** (3 - self.gammap)) * quad(lambda t: t ** (-1 - (1 + self.gammap) / (3 - self.gammap)) * (1 - t) ** (4 / (3 - self.gammap) - 1) * log(1 - t), (r / self.rp) ** (3 - self.gammap) / (1 + (r / self.rp) ** (3 - self.gammap)), 1, limit=1000, epsabs=1e-6, epsrel=1e-6)[0])),
                'rho': lambda r: 1e-9 * (3 - self.gammap) * self.Mg / (4 * pi * self.rp ** 3 * log(1 + (self.Rmax / self.rp) ** (3 - self.gammap))) / (r / self.rp) ** self.gammap / (1 + (r / self.rp) ** (3 - self.gammap))
                }, # Moore / Generalized NFW. Mass diverges so it is truncated up to Rmax. It is generelized, but the user can select gammap=1.5.
            
            'Truncated_Einasto': {
                'rt_index': lambda r: 3 - self.gammap * (r / self.rp) ** (3 - self.gammap1) * exp(- (r / self.rp) ** self.gammap) / gamma((3 - self.gammap1) / self.gammap) / gammainc((3 - self.gammap1) / self.gammap, (r / self.rp) ** self.gammap),
                'Phi': lambda r: - 1e-3 * self.G * self.Mg / self.rp * (self.rp / r * gammainc((3 - self.gammap1) / self.gammap, (r / self.rp) ** self.gammap) + gamma((2 - self.gammap1) / self.gammap) * gammaincc((2 - self.gammap1) / self.gammap, (r / self.rp) ** self.gammap) / gamma((3 - self.gammap1) / self.gammap)),
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg * gammainc((3 - self.gammap1) / self.gammap, (r / self.rp) ** self.gammap) / r ** 2,
                'd2Phi_dr2': lambda r: - 1e-9 * self.G * self.Mg * gammainc((3 - self.gammap1) / self.gammap, (r / self.rp) ** self.gammap) / r ** 3 * (2 - self.gammap * (r / self.rp) ** (3 - self.gammap1) * exp(-(r / self.rp) ** self.gammap) / gamma((3 - self.gammap1) / self.gammap) / gammainc((3 - self.gammap1) / self.gammap, (r / self.rp) ** self.gammap)),
                'X2': numpy.vectorize(lambda r: self.gammap * gammainc((3 - self.gammap1) / self.gammap, (r / self.rp) ** self.gammap) * exp(- (r / self.rp) ** self.gammap) / (2 * (r / self.rp) ** (1 + self.gammap1) * quad(lambda t: exp(-t) * t ** (- (1 + self.gammap1) / self.gammap - 1) * gammainc((3 - self.gammap1) / self.gammap, t), (r / self.rp) ** self.gammap, numpy.inf, limit=1000, epsabs=1e-6, epsrel=1e-6)[0])),
                'rho': lambda r: 1e-9 * self.Mg * self.gammap / 4 / pi / self.rp ** 3 / gamma((3 - self.gammap1) / self.gammap) * (r / self.rp) ** (- self.gammap1) * exp(-(r / self.rp) ** self.gammap)
                }, # Einasto profile for gammap1=0.
            
            'Zhao': {
                'rt_index': lambda r: 3 - self.gammap * (r / self.rp) ** (3 - self.gammap2) / ((1 + (r / self.rp) ** self.gammap) ** ((self.gammap1 - self.gammap2) / self.gammap) * betainc((3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap, (r ** self.gammap / (self.rp ** self.gammap + r ** self.gammap))) * beta((3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap)),
                'Phi': lambda r: - 1e-3 * self.G * self.Mg / self.rp * ( self.rp / r * betainc((3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap, (r ** self.gammap / (self.rp ** self.gammap + r ** self.gammap)))  + betainc((self.gammap1 - 2) / self.gammap, (2 - self.gammap2) / self.gammap, (self.rp ** self.gammap / (self.rp ** self.gammap + r ** self.gammap))) * beta((self.gammap1 - 2) / self.gammap, (2 - self.gammap2) / self.gammap) / beta((3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap)),
                'dPhi_dr': lambda r: 1e-6 * self.G * self.Mg / r ** 2 * betainc((3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap, (r ** self.gammap / (self.rp ** self.gammap + r ** self.gammap))) ,
                'd2Phi_dr2': lambda r: - 2e-9 * self.G * self.Mg / r ** 3 * betainc((3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap, (r ** self.gammap / (self.rp ** self.gammap + r ** self.gammap))) + 1e-9 * self.gammap * self.G * self.Mg / r ** 3 * ((r / self.rp) ** (3 - self.gammap2)) / (1 + (r / self.rp) ** self.gammap) ** ((self.gammap1 - self.gammap2) / self.gammap) / beta((3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap) ,
                'X2': numpy.vectorize(lambda r: self.gammap * betainc((3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap, (r ** self.gammap / (self.rp ** self.gammap + r ** self.gammap))) / (2 * (r / self.rp) ** (1 + self.gammap2) * (1 + (r / self.rp) ** self.gammap) ** ((self.gammap1 - self.gammap2) / self.gammap) * quad(lambda t: betainc((3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap, t) * t ** (- 1 - (1 + self.gammap2) / self.gammap) * (1 - t) ** (- 1 + (self.gammap1 + 1) / self.gammap), (r / self.rp) ** self.gammap / (1 + (r / self.rp) ** self.gammap), 1, limit=1000, epsabs=1e-6, epsrel=1e-6)[0])),
                'rho': lambda r: 1e-9 * self.gammap * self.Mg / (4 * pi * self.rp ** 3) / (r / self.rp) ** (self.gammap2) / (1 + (r / self.rp) ** self.gammap) ** ((self.gammap1 - self.gammap2) / self.gammap) / beta((3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap)
                } # Zhao model. Generalizes the families introduced before. For example, it agrees with Hernquist, Plummer, Jaffe, pergect sphere. NFW and Moore struggle with the incomplete beta. The user should keep in mind to always insert well behaved exponents so that the special functions are well defined, that is gammap > 0, gammap1 > 3, gammap2 < 3. The family of models with gammap2=0 does not have an analytic expression for the velocity dispersion squared so it is not presented separately. The same applies to generalized Moore models with gammap -> 3 - gammap2 + epsilon, gammap1 -> 3 + epsilon.
        } 
        
        # Future extension: Add a SMBH in the model. This shifts rt_index, Vc(r) or M(r) and X2. 
        # Future extension2: Add a dictionary with anisotropic models and the solution for the radial velocity dispersion from Jean's equations.
        
        # Cluster model dictionary. NFW, Power-law, MIS and Hubble use a fixed maximum distance to estimate the potential and its derivative, to avoid divergences.
        self.cluster_model_dict = { # Dictionary for spherically symmetric cluster potentials. Units are [pc^2 / Myr^2]. Radius should be in [pc], mass in [Msun]. Currently used for for estimating the 
            'SIS': {'phi': lambda r, M: (1.023 * self.Vcc) ** 2 * log(r / self.Rmaxc), 'dPhi_dr': lambda r: (1.023 * self.Vcc) ** 2 / r},
            
            'Point_mass': {'phi': lambda r, M: - self.G * M / r, 'dPhi_dr': lambda r, M: self.G * M / r ** 2},
            
            'Plummer': {'phi': lambda r, M: - self.G * M / sqrt(r ** 2 + self.rpc ** 2), 'dPhi_dr': lambda r, M: self.G * M * r / (r ** 2 + self.rpc ** 2) ** (3 / 2)},
            
            'Hernquist': {'phi': lambda r, M: - self.G * M / (r + self.rpc), 'dPhi_dr': lambda r, M: self.G * M / (r + self.rpc) ** 2},
            
            'Jaffe': {'phi':lambda r, M: - self.G * M / self.rpc * log(1 + self.rpc / r), 'dPhi_dr': lambda r, M: self.G * M / (r * (r + self.rpc))},
            
            'NFW': {'phi': lambda r, M: - self.G * M / r * log(1 + r / self.rpc) / (log(1 + self.Rmaxc / self.rpc) - self.Rmaxc / (self.rpc + self.Rmaxc)), 'dPhi_dr': lambda r, M: self.G * M * (log(1 + r /  self.rpc) - r / (r + self.rpc)) / (log(1 + self.Rmaxc / self.rpc) - self.Rmaxc / (self.rpc + self.Rmaxc)) / r ** 2},
            
            'Isochrone': {'phi': lambda r, M: - self.G * M / (self.rpc + sqrt(r ** 2 + self.rpc ** 2)), 'dPhi_dr': lambda r, M: self.G * M * r / ((self.rpc + sqrt(self.rpc ** 2 + r ** 2)) ** 2 * sqrt(self.rpc ** 2 + r ** 2))},
            
            'Dehnen': {'phi': lambda r, M: - self.G * M / (self.rpc ** (2 - self.gammapc)) * (1 - (r / (r + self.rpc)) ** (2 - self.gammapc)), 'dPhi_dr': lambda r, M: self.G * M * r ** (1 - self.gammapc) / (r + self.rpc) ** (3 - self.gammapc)},
            
            'Veltmann': {'phi': lambda r, M: - self.G * M / (r ** self.gammapc + self.rpc ** self.gammapc) ** (1 / self.gammapc), 'dPhi_dr': lambda r, M: self.G * M * r ** (self.gammapc - 1) / (r ** self.gammapc + self.rpc ** self.gammapc) ** (1 + 1 / self.gammapc)},
            
            'Soft': {'phi': lambda r, M: - self.G * M / r * (1 + self.rpc / r), 'dPhi_dr': lambda r, M: self.G * M / r ** 2 * (1 + 2 * self.rpc / r)},
            
            'Power_law': {'phi': lambda r, M: - self.G * M / (self.rpc * (self.gammapc - 2)) * r ** (2 - self.gammapc) * (self.rpc / self.Rmaxc) ** (3 - self.gammapc), 'dPhi_dr': lambda r, M: self.G * M * r ** (1 - self.gammapc) / self.rpc ** (3 - self.gammapc) * (self.rpc / self.Rmaxc) ** (3 - self.gammapc)},
            
            'MIS': {'phi': lambda r, M: - self.G * M / self.rpc / (self.Rmaxc / self.rpc - arctan(self.Rmaxc / self.rpc)) * (1 / (r / self.rpc) * ((r / self.rpc) - arctan((r / self.rpc))) - log(sqrt(1 + (r / self.rpc) ** 2) / sqrt(1 + (self.Rmaxc / self.rpc) ** 2))), 'dPhi_dr': lambda r, M: self.G * M / r ** 2 * (r / self.rpc - arctan(r / self.rpc)) / (self.Rmaxc / self.rpc - arctan(self.Rmaxc / self.rpc))},
            
            'Perfect_sphere': {'phi': lambda r, M: - self.G * M / r / (pi / 2) * arctan(r / self.rpc), 'dPhi_dr': lambda r, M: self.G * M / r ** 2 * (arctan(r / self.rpc) - (r / self.rpc) / (1 + (r / self.rpc) ** 2)) / (pi / 2)},
            
            'Modified_Hubble': {'phi': lambda r, M: - self.G * M / r * arctanh(r / self.rpc / sqrt(1 + (r / self.rpc) ** 2)) / (arctanh(self.Rpmaxc / self.rpc /  sqrt(1 + (self.Rpmaxc / self.rpc) ** 2)) - self.Rpmaxc / self.rpc / sqrt(1 + (self.Rpmaxc / self.rpc) ** 2)), 'dPhi_dr': lambda r, M: self.G * M / r ** 2 * (arctanh((r / self.rpc) / sqrt(1 + (r / self.rpc)** 2)) - (r / self.rpc) / sqrt(1 + (r / self.rpc) ** 2)) / (arctanh((self.Rmaxc / self.rpc) / sqrt(1 + (self.Rmaxc / self.rpc) ** 2)) - (self.Rmaxc / self.rpc) / sqrt(1 + (self.Rmaxc / self.rpc) ** 2))},
            
            'Moore': {'phi': lambda r, M: - self.G * M / self.rpc / log(1 + (self.Rmaxc / self.rpc) ** (3 - self.gammapc)) * (self.rpc / r * log(1 + (r / self.rpc) ** (3 - self.gammapc)) + beta(1 / (3 - self.gammapc), (2 - self.gammapc) / (3 - self.gammapc)) * betainc(1 / (3 - self.gammapc), (2 - self.gammapc) / (3 - self.gammapc), 1 / (1 + (r / self.rpc) ** (3 - self.gammapc)))), 'dPhi_dr': lambda r, M: self.G * M / r ** 2 * log(1 + (r / self.rpc) ** (3 - self.gammapc)) / log(1 + (self.Rmaxc / self.rpc) ** (3 - self.gammapc))},
            
            'Truncated_Einasto': {'phi': lambda r, M:  - self.G * M / self.rpc * (self.rpc / r * gammainc((3 - self.gammapc1) / self.gammapc, (r / self.rpc) ** self.gammapc) + gamma((2 - self.gammapc1) / self.gammapc) * gammaincc(2 / self.gammapc, (r / self.rpc) ** self.gammapc) / gamma(3 / self.gammapc)),'dPhi_dr': lambda r, M: self.G * M * gammainc((3 - self.gammapc1) / self.gammapc, (r / self.rpc) ** self.gammapc) / r ** 2},
            
            'Zhao': {'phi': lambda r, M: - self.G * M / self.rpc * ( self.rpc / r * betainc((3 - self.gammapc2) / self.gammapc, (self.gammapc1 - 3) / self.gammapc, (r ** self.gammapc / (self.rpc ** self.gammapc + r ** self.gammapc)))  + betainc((self.gammapc1 - 2) / self.gammapc, (2 - self.gammapc2) / self.gammapc, (self.rpc ** self.gammapc / (self.rpc ** self.gammapc + r ** self.gammapc))) * beta((self.gammap1 - 2) / self.gammap, (2 - self.gammap2) / self.gammap) / beta((3 - self.gammap2) / self.gammap, (self.gammap1 - 3) / self.gammap)), 'dPhi_dr': lambda r, M: self.G * M * r ** (1 - self.gammapc) / self.rpc ** (3 - self.gammapc)},
            
        }
        
        # The dictionary is subject to change, should a better description for BH ejections when fbh close to 0 is found.
        # Dictionary that introduces a scaling on the BH ejection rate with respect to Spitzer's paramater. Can be used as a proxy for different parametrizations as well. Can be used for describing equipartition.
        self.beta_dict = {
           'exponential': lambda S: 1 - exp(- (S / self.S0) ** self.gamma2),
           'logistic': lambda S: (S ** self.gamma2 / (S ** self.gamma2 + self.S0 ** self.gamma2)),
           'error_function': lambda S:  erf(S / self.S0),
           'hyperbolic': lambda S: tanh(S / self.S0)
        }
        
        # Dictionary for balancing functions. Functions must be 0 at t=0 and for initial times, and 1 at tcc<=t. Default to step like.
        self.balance_dict = {
            'step':lambda t: numpy.heaviside(t - self.tcc, 1),
            'error_function':lambda t:  0.5 * (1 + erf(self.gamma4 * self.gamma3 * (t - self.gamma4 * self.tcc / self.gamma3 ) / self.tcc)) ,
            'hyperbolic': lambda t: 0.5 * (1 + tanh(self.gamma3 * self.gamma4 * (t - self.gamma4 * self.tcc / self.gamma3) / self.tcc)),
            'exponential': lambda t: 1 - exp(- self.gamma3 * (t / self.tcc) ** self.gamma4)
        }
        
        # The user can either specify the initial average mass beforehand, or it is extracted from the IMF.
        # If m0 is specified in the beginning, the IMF must be well defined so that it matches.
        if self.m0 is None:
           self.m0 = self._initial_average_mass(self.a_slopes, self.m_breaks) # [Msun] Average mass obtained for this particular IMF. 
        
        self.FeH = log10(self.Z / self.Zsolar) # Metallicity in solar units.
        
        self.M0 = self.m0 * N # [Msun] Total mass of stars (cluster) initially. This is used for computations, even if the BH population is present in clusterBH at t=0.
        self.rh0 = (3 * self.M0 / (8 * pi * rhoh)) ** (1./3) # [pc] Initial half-mass radius.
        self.vesc0 = self._vesc(self.M0, self.rh0) # [km/s] Initial escape velocity.
        
        # Check the BH ejection model.
        if self.sev_model not in self.sev_dict:
            raise ValueError(f"Invalid model for stellar evolution: {self.sev_model}.")
        
        # Check the BH ejection model.
        if self.beta_model not in self.beta_dict:
            raise ValueError(f"Invalid model for BH ejections: {self.beta_model}.")
        
        # Check the balance model. Future implementation.
        if self.balance_model not in self.balance_dict:
            raise ValueError(f"Invalid model for balancing: {self.balance_model}.")
        
        # Check if there is a step for integration. If so, at each step the results are saved. Otherwise a dense output is used.
        if self.dtout is None:
            self.t_eval = None
        else:
            self.t_eval = numpy.arange(0, self.tend, self.dtout) # Area for integration.
        
        self.mst_sev, self.t_mst = self._mst_sev() # [Msun, Myrs] Solution for average stellar mass if stellar evolution is considered solely. Time is kept for interpolation.
        
        # Check whether SSP are used for the BHMF.
        if self.ssp:
            import ssptools # Import the package. Must be installed first.
            
            self.ibh_kwargs.setdefault('kick_vdisp', self.sigmans) # Default arguments for ssp.
            
            # Implement kicks, if activated, for this IMF, number of stars, with such metallicity, central escape velocity and BHMF conditions.
            self.ibh = ssptools.InitialBHPopulation.from_powerlaw(self.m_breaks, self.a_slopes, self.nbins, self.FeH, N0=N, vesc=self.vesc0, natal_kicks=self.kick, **self.ibh_kwargs)
            self.Mbh0 = self.ibh.Mtot  # [Msun] Expected initial mass of BHs.
            self.f0 = self.Mbh0 / self.M0 # Initial fraction of BHs. Should be close to 0.06 for poor-metal clusters. 
            
            self.Nbh0 = self.ibh.Ntot # Initial number of BHs. Round the number for consistency.
            self.mbh0 = self.Mbh0 / self.Nbh0 # [Msun] Initial average BH mass.
            self.mlo = self.ibh.m.min() # [Msun] Minimum BH mass in the BHMF.
            self.mup = self.ibh.m.max() # [Msun] Maximum BH mass in the BHMF.
            self.Mst_lost = self.ibh.Ms_lost # [Msun] Mass of stars lost in order to form BHs.
            self.t_bhcreation = self.ibh.age # [Myrs] Time needed to form these astrophysical mass BHs.
            self.mst_sev_BH = numpy.interp(self.t_bhcreation, self.t_mst, self.mst_sev) # [Msun] Average stellar mass when all BHs are created. Effect of tides is neglected for simplicity.
            
        # Should the user wishes to exclude the ssp tools and use another approach, they can first define f0 and then compute Mbh0. A simple approach would be the same value regardless of kicks.
        else:
            mmax_ = numpy.logspace(log10(self.mlo), log10(self.mup), self.N_points) # List of possible values for the maximum BH mass at a given time instance.
            self.b_BH = (self.alpha_BH + 2) / 3 # Auxiliary exponent.
            self.qul = self.mup / self.mlo # Ratio of upper and lower mass in the BHMF.
            self.qul3, self.mlo3b, self.mup3b = self.qul ** 3, self.mlo ** (3 * self.b_BH), self.mup ** (3 * self.b_BH) # Constants that are used multiple times. They are defined to avoid subsequent recomputations.
            if self.alpha_BH != -2: self.b_BH3 = 1./(3 * self.b_BH) # Exponent that participates in computations for alpha_BH different than -2, for no kicks.
            
            self.Mbh0 = self.f0 * self.M0 # [Msun] Prediction for the initial BH mass if SSP are not used and kicks are deactivated.
            
            if self.kick: # Depending on the condition for kicks, a different retention fraction is used.
               
               # Checks if the user has actually inserted a bad exponent for the BHMF that cannot be used.
               k = (self.alpha_BH + 5) / -3  # Solve for k in equation: self.alpha_BH = -5 - 3k. These are values that give negative integers in the third input of the hypergeometric functions used.
    
               if k.is_integer() and k >= 0:  # Checks whether k is a non-negative integer.
                  raise ValueError(f"Exponent {self.alpha_BH} cannot be used with kicks and power-law BHMF since the hypergeometric functions used are ill defined. Choose other conditions.")

               self.mb = (9 * pi / 2) ** (1./6) * self.sigmans * self.mns / self.vesc0 # [Msun] Correct the lowest mass unaffected by kicks. All masses below this threshold are affected by kicks.
               self.qub, self.qlb = self.mup / self.mb, self.mlo / self.mb # Mass ratios.
               self.qub3, self.qlb3, self.qul3b = self.qub ** 3, self.qlb ** 3, self.qul ** (3 * self.b_BH) # Ratios that appear multiple times in calculations.
               self.h2 = hyp2f1(1, self.b_BH, self.b_BH + 1, -self.qlb3) # Hypergeometric function. Appears several times.
               self.mloh2 = self.mlo3b * (1 - self.h2) # Constant prefactor which appears several times.
               
               # Function that returns the result of integrating the power-law mass function. It is normalized to the prefactor of the mass function. Valid only for power-law BHMF.
               def integr(mm, qmb):

                   if self.alpha_BH == -2: # This excludes only the alpha_BH = -2, but in principle other choices should be excluded, those that are problematic for hypergeometric functions (negative integers for b + 1). 
                      return log((1 + qmb ** 3) / (1 + self.qlb3))  # Integral of power-law mass function without the constant prefactor.
                   
                   else:
                       
                      h1 = hyp2f1(1, self.b_BH, self.b_BH + 1, -qmb ** 3) # Hypergeometric function used for the solution.
                      return mm ** (3 * self.b_BH) * (1 - h1) - self.mloh2 # BH mass extracted by integrating the mass function, neglecting the constant prefactor.

               self.qmb = mmax_ / self.mb # Auxiliary mass rations that can be used in the solution.

               if self.alpha_BH != -2: # It is important so that the hypergeometric function is well defined.
                   h1 = hyp2f1(1, self.b_BH, self.b_BH + 1, -self.qub3) # Hypergeometric function, part of the solution.
                   self.fretm = 1 - (self.qul3b * h1 - self.h2) / (self.qul3b - 1) # Retention fraction.
               
               else:
                   
                   self.fretm = log( (self.qub3 + 1) / (self.qlb3 + 1) ) / log(self.qul3) # Retention fraction.
               
               self.Mbh0 *= self.fretm # [Msun] Correct the Initial BH mass since the retention fraction is now lowered.
               
            # Compute function values to be used in integrals.
            num_values = numpy.array(mmax_ ** (self.alpha_BH + 1) / ((self.mb / mmax_) ** 3 + 1))
            den_values = numpy.array(mmax_ ** self.alpha_BH / ((self.mb / mmax_) ** 3 + 1))

            # Precompute integrals for average mass.
            self.num_integral = cumulative_trapezoid(num_values, mmax_, initial=0) # [Msun] Total mass in the BHMF. 
            self.den_integral = cumulative_trapezoid(den_values, mmax_, initial=0) # Total number of BHs in the BHMF.
            self.mmax_ = mmax_ # Array will be recalled.
            self.Mbh_ = self.Mbh0 * self.num_integral / self.num_integral[-1] #  [Msun] BH mass points that are generated from the possible max BH mass points.
            self.Mbh_[self.Mbh_ < self.mlo] = self.mlo # Ensure that no value is below the lowest BH mass.
            
            # Further properties.
            self.mbh0 = self._mbh(self.Mbh0) # [Msun] Initial average BH mass.
            self.Nbh0 = numpy.round(self.Mbh0 / self.mbh0).astype(int) # Initial number of BHs.
            self.mst_sev_BH = numpy.interp(self.t_bhcreation, self.t_mst, self.mst_sev) # [Msun] Average stellar mass when all BHs are created. Effect of tides is neglected for simplicity.
            self.Mst_lost = (self.m0 - self.mst_sev_BH) * N # [Msun] Approximate value for the stellar mass lost to create all BHs.
        
        if self.Mbh0 < self.mlo: self.Mbh0, self.mbh0, self.Nbh0 = 1e-99, 1e-89, 0 # This condition checks whether BHs will be created. If not, clusterBH works only with stars.
       
        # Initial relaxation is extracted with respect to the BH population that will be obtained. It is an approximation such that large metallicity clusters have larger relaxation, as expected. This approach inserts an indirect metallicity dependence.
        if not 0 <= self.Sseg <= 1: # Initial segregation is constrained.
            raise ValueError(f'Degree of segregation {self.Sseg} is invalid. The value must be within [0, 1].')
        
        self.psi0 = self._psi(self.Mbh0 / (self.M0 + 0*self.Mbh0), self.M0 + 0*self.Mbh0, self.mbh0, self.m0) # Use as time instance the moment when all BHs are formed. In this regime, tides are not so important so all the mass-loss comes from stellar evolution. 
        # If self.mst_sev_BH is used in trh0 and psi0, self.ntrh is shifted mildly.
        self.trh0 = 0.138 / (self.m0 * self.psi0 * log(self.gamma * self.N)) * sqrt(self.M0 * self.rh0 ** 3 / self.G ) * (1 - 2 * self.omega0 ** 2 * self.rh0 ** 3 / (self.G * self.M0)) ** (3 / 2) # [Myrs] We use m0 since at t=0 the mass distribution is the same everywhere. Valid for unsegregated clusters.
        self.tcc = self.ntrh * self.trh0 * (1 - self.Sseg) ** self.aseg # [Myrs] Core collapse. Effectively it depends on metallicity, size and mass of the cluster. Changes with the level of segregation.
      
        # Check if the galactic model, cluster model, tidal model have changed. Based on the available choices in the dictionaries, change the values accordingly.
        if self.tidal: # Valid check only if we have tides.
            if self.galactic_model in self.galactic_model_dict:
                
                # Check if the Zhao model is selected and if so, whether the parameters are well behaved. In a different scenario the potential diverges and a numerical solution is needed. Currently not available in clusterBH.
                if self.galactic_model=='Zhao' and (self.gammap < 0 or self.gammap1 <= 3 or self.gammap2 >= 3):
                    raise ValueError(f'Invalid parameters for the Zhao model: {self.gammap}, {self.gammap1}, {self.gammap2}. Potential diverges. Use values in the vicinity {self.gammap}>0, {self.gammap1}>3, {self.gammap2}<3.')
               
                # Galactic properties.
                # To obtain the correct galactic mass, the user can either specify it as arguments or specify the circular velocity Vc in [km/s] at distance rg in [kpc] and the computation is performed automatically.
                self.Mg = 1.023 ** 2 * self.Vc ** 2 * self.rg * 1e3 / self.G if 'Mg' not in kwargs else kwargs['Mg'] # [Msun] Mass of the galaxy inside the orbit of the cluster. Derived from orbital properties.
                self.rt_index = self.galactic_model_dict[self.galactic_model]['rt_index'] # Index used in the tidal radius. Applies to circular orbits only. Can vary with time in the case of tidal spiraling.
                self.dPhi_dr_func = self.galactic_model_dict[self.galactic_model]['dPhi_dr'] # [pc/Myrs^2] Derivative of the galactic potential. Can be used to obtain the rotational curve for a galactic model at any distance.
                self.d2Phi_dr2_func = self.galactic_model_dict[self.galactic_model]['d2Phi_dr2'] # [pc/Myrs^2] Derivative of the galactic potential. Can be used to obtain the rotational curve for a galactic model at any distance.
                self.Vc_func = lambda r: sqrt(1e3 * r * self.dPhi_dr_func(r)) / 1.023 # [km / s] Velocity profile required to maintain a circular orbit.
                self.rhoG_func = self.galactic_model_dict[self.galactic_model]['rho'] # [Msun / pc^3] Density profile of the galactic model. Currently implemented only for tidal spiraling.
                self.X2_func = self.galactic_model_dict[self.galactic_model]['X2'] # Ratio of squared velocity profile over velocity dispersion.
                self.L = 1.023 * self.Vc_func(self.rg) * self.rg * 1e3 # [pc^2 / Myrs] Orbital angular momentum at t=0. Conserved if no tidal spiralling is assumed.  
                
            else:
                raise ValueError(f"Invalid galactic model: {self.galactic_model}.") 
                
            # Currently the cluster model is needed only if tides are activated and the evaporation of stars changes the energy budget.
            if self.cluster_model not in self.cluster_model_dict:
                raise ValueError(f"Invalid cluster model: {self.cluster_model}.") 
        
            if self.tidal_model not in self.tidal_models:
                raise ValueError(f"Invalid tidal model: {self.tidal_model}.")
        
        # Specify the functions that shall be used. They are available in the dictionaries.
        self.nu_function = self.sev_dict[self.sev_model] # Mass loss rate of stars due to stellar evolution.
        self.xi_function = self.tidal_models[self.tidal_model] # Evaporation mass loss rate.
        self.beta_function = self.beta_dict[self.beta_model] # Ejection rate of BHs.
        self.Phic_function = self.cluster_model_dict[self.cluster_model]['phi'] # [pc^2/Myrs^2] Potential of the cluster.
        self.dPhic_dr_func = self.cluster_model_dict[self.cluster_model]['dPhi_dr'] # [pc/Myrs^2] Derivative of the galactic potential.
        self.balance_function = self.balance_dict[self.balance_model] # Function used to make the differential equation continuous.

        self._evolve(N, rhoh) # Runs the integrator, generates the results.
        
    # Functions:

    def _sev_factor(self, a_slopes1, a_slopes2, m_breaks1, m_breaks2):
        """
        Computes the scaling factor between two different IMFs
        by normalizing and integrating them over the given mass ranges.
        
        This function first determines normalization constants to ensure IMF continuity,
        then integrates each IMF to compute their total mass, and finally calculates 
        the fraction of mass in the high-mass end of the second IMF relative to the first.
        
        Parameters
        ----------
        a_slopes1 : list of float
            The power-law slopes for the first IMF.
        a_slopes2 : list of float
            The power-law slopes for the second IMF.
        m_breaks1 : list of float
            The mass breakpoints corresponding to `a_slopes1`. [Msun]
        m_breaks2 : list of float
            The mass breakpoints corresponding to `a_slopes2`. [Msun]
        
        Returns
        -------
        float
            The ratio of the two IMFs' total mass, adjusted by the mass fraction 
            in the upper limit of the second IMF relative to the first. First IMF is the Kroupa.
        """
        
        def integrate_IMF(m_low, m_high, p):
            if p == -2:  # Handles special case where p = -2 to avoid division by zero.
                return log(m_high / m_low)
            else:
                return (m_high ** (p + 2) - m_low ** (p + 2)) / (p + 2)
        
        # Normalization constants make the IMF continuous.
        def norm_constants(a_slopes, m_breaks):
            c_values = [1.0]  # c1 = 1.0 (initial normalization). Does not matter.
            norm_constants = []
        
            for i in range(1, len(a_slopes)):
                c_values.append(m_breaks[i] ** (a_slopes[i - 1] - a_slopes[i]))
        
            for i in range(len(c_values)):
                norm_constants.append(numpy.prod(c_values[:i + 1]))
        
            return norm_constants
    
        # Compute normalization constants for both sets of slopes.
        norm_consts1 = norm_constants(a_slopes1, m_breaks1)
        norm_consts2 = norm_constants(a_slopes2, m_breaks2)
        
        # Compute the integrals for both IMF sets. They denote the total mass.
        integral1, integral2 = 0, 0
        for i in range(len(a_slopes1)):
            integral1 += norm_consts1[i] * integrate_IMF(m_breaks1[i], m_breaks1[i + 1], a_slopes1[i])

        for i in range(len(a_slopes2)):
            integral2 += norm_consts2[i] * integrate_IMF(m_breaks2[i], m_breaks2[i + 1], a_slopes2[i])
        
        # Fraction of mass in the upper limit of the Kroupa.
        r = norm_consts2[-1] * integrate_IMF(m_breaks2[-2], m_breaks2[-1], a_slopes2[-1])
        r /= norm_consts1[-1] * integrate_IMF(m_breaks1[-2], m_breaks1[-1], a_slopes1[-1])
       
        # Return the ratio of the two integrals.
        return r * integral1 / integral2

    # Compute initial average mass.
    def _initial_average_mass(self, a_slopes, m_breaks):
       """
       Calculates the initial average mass of stars based on a piecewise power-law IMF.
    
       Parameters:
       -----------
       a_slopes : list of floats
          Slopes of the IMF in different mass ranges. Each slope corresponds to a segment of the piecewise IMF.
       m_breaks : list of floats
          Breakpoints of the mass ranges [Msun]. The length of this list should be one more than `a_slopes`.

       Returns:
       --------
       float
        The average mass of stars calculated using the IMF and its normalization [Msun].
    
       Notes:
       ------
       - This function integrates the IMF.
       - It assumes the IMF is continuous across the mass breakpoints.
       - Special handling is included for slopes to avoid division by zero.
       """
       
       def integrate_IMF(m_low, m_high, p):
        
           if p == -1:  # Special case where p = -1 to avoid division by zero, should the user select such value.
              return log(m_high / m_low) 
           else:
              return (m_high ** (p + 1) - m_low ** (p + 1)) / (p + 1)
       
       # Normalization constants to ensure continuity of the IMF.
       c_values = [1.0]  # c1 = 1.0. First is irrelevant.
       normalization_constants = []
       
       # Calculate the rest of the c values based on slopes and mass breakpoints. This is for continuity.
       for i in range(1, len(a_slopes)):
           c_values.append(m_breaks[i] ** (a_slopes[i - 1] - a_slopes[i])) # Previous exponent minus current one, to compute the current prefactor.

       # Compute the cumulative products to form the normalization_constants array. Ensures continuity.
       for i in range(len(c_values) - 1):
         normalization_constants.append(c_values[i] * c_values[i + 1]) # For every prefactor, we need to take the product with the previous one. We have 2 breakpoints in 1 interval.
       
      # The prefactors are dimensionless here, if a normalization to 1Msun is assumed in the power-laws.
       normalization_constants.insert(0, c_values[0]) # First entry is 1. 
       
       # Set initial values to 0.
       stellar_mass = 0 
       stellar_number = 0
       
       # For every pair, compute the number of massess and stars.
       for i, a in enumerate(a_slopes):
          m_low = m_breaks[i]
          m_high = m_breaks[i + 1]
          c_i = normalization_constants[i]
          
          # Compute numerator.
          stellar_mass += c_i * integrate_IMF(m_low, m_high, a + 1)
         
          # Compute denominator.
          stellar_number += c_i * integrate_IMF(m_low, m_high, a)
    
       # Calculate the average mass. A well defined IMF should have a nonzero denominator.
       return stellar_mass / stellar_number # [Msun]
    
    # Function to determine by how much the BHMF changes, given a particular mass loss.
    def _deplete_BHMF(self, M_eject, M_BH, N_BH):
        """
        Depletes the BHMF by ejecting mass starting from the heaviest bins. This is subject to change once dark clusters are considered.
        Possible extension. Insert M_eject and also the type. Ejections are subtracted top-down while evaporation bottom-up.
    
        Parameters:
         -----------
        M_eject : float
           Total mass to eject from the BHMF [Msun].
        M_BH : numpy.ndarray
           Array representing the total mass in each mass bin [Msun].
        N_BH : numpy.ndarray
           Array representing the number of BHs in each mass bin.

        Returns:
        --------
        Tuple of numpy.ndarray
           Updated arrays for the mass in each BH bin (M_BH) and the number of BHs (N_BH).
    
        Notes:
        ------
        - The function starts ejecting mass from the heaviest bin and proceeds to lighter bins.
        - If the mass to eject (`M_eject`) exceeds the total mass in a bin, the entire bin is depleted.
        - For partially depleted bins, the corresponding number of BHs (`N_BH`) is adjusted proportionally.
        - The method avoids modifying the original arrays by creating copies of `M_BH` and `N_BH`.
        """
        # Avoid altering initial BH bins.
        M_BH, N_BH = M_BH.copy(), N_BH.copy() # [Msun, dimensionless]

        # Remove BH starting from Heavy to Light.
        j = M_BH.size
 
        # Bins are emptied for as long as the mass ejected is nonzero.
        while M_eject != 0: # [Msun]
            j -= 1

            # Stop ejecting if trying to eject more mass than there is in BHs.
            if j < 0:
                break
             
            # Remove entirety of this mass bin. Adjust the ejected mass.
            if M_BH[j] < M_eject: # [Msun]
                M_eject -= M_BH[j] # [Msun]
                M_BH[j] = 0 # [Msun]
                N_BH[j] = 0 # Dimensionless
                continue

            # Remove required fraction of the last affected bin.
            else:
                m_BH_j = M_BH[j] / N_BH[j] # [Msun] The average BH mass in a given bin is assumed to be constant.
                M_BH[j] -= numpy.asarray(M_eject).item()  # [Msun] Convert to scalar explicitly, allows for multiple values to be computed.
                N_BH[j] -= numpy.asarray(M_eject / m_BH_j).item() # Number of remaining BHs.
                break

        return M_BH, N_BH
    
    # Calculates the upper mass in the BHMF at any given moment.
    def _find_mmax(self, Mbh):
        """
        Computes the maximum BH mass at each time instance for two cases of BHMFs. Either directly from the SSP tools, or from a simple power-law the user has specified.
    
        Parameters:
        -----------
        Mbh : float
            The current total BH mass [Msun].
    
        Returns:
        --------
        mmax : float
            The maximum individual BH mass at the current time instance [Msun]. It changes only by assuming ejections. A different prescription is needed if the tidal field is important for the BH subsystem.
        """
        
        # Check the routine that is used for the BHMF.
        if self.ssp:
            
            M_eject = self.Mbh0 - Mbh # [Msun] Mass that has been ejected.
            M_BH, N_BH = self._deplete_BHMF(M_eject, self.ibh.M, self.ibh.N) # [Msun, dimensionless]. BHMF at a given time instance.
            valid_bins = N_BH > 0 # Create a mask to see which bins are not empty. These have BHs.
            # Find the maximum individual BH mass in the valid bins.

            if valid_bins.any() > 0: # Ensure valid_bins is not empty
               mmax = numpy.max(self.ibh.m[valid_bins]) # [Msun] Consider only non-zero bins.
            else:
               mmax = 0 # All BHs have been ejected.
        
        else:
           
            # Arrays Mbh_ and mmax_ are constructed accordingly for all cases of alpha_BH, apart from those that are ill defined. No need to check.
            mmax = numpy.interp(Mbh, self.Mbh_, self.mmax_)
            
        return mmax
    
    # Average BH mass.
    def _mbh(self, Mbh):
        """
        Calculates the updated average BH mass after ejecting a specified amount of BH mass. 
        The computation is done based on which BHMF is used (either from the SSP or a simple power-law specified by the user).
    
        Parameters:
        -----------
        Mbh : float
          Total BH mass after ejection [Msun].

        Returns:
        --------
        float
          Average BH mass after ejections [Msun].
    
        Notes:
        ------
        - The function first calculates the total mass to be ejected (`M_eject`) in [Msun].
        - It then updates the BHMF by removing mass using `_deplete_BHMF`. The removal it top-down.
        - Finally, the average mass of the remaining BHs is computed by dividing the total mass by the total number. Units in [Msun].
        """
        
        # Check which BHMF is used.
        if self.ssp:
           # Determine amount of BH mass (total) to eject.
           M_eject = self.Mbh0 - Mbh # [Msun]
            
           # Deplete the initial BH MF based on this M_eject.
           M_BH, N_BH = self._deplete_BHMF(M_eject, self.ibh.M, self.ibh.N) # [Msun, dimensionless]
         
           return M_BH.sum() / N_BH.sum() # [Msun]
        
        else: 
            
            mmax = self._find_mmax(Mbh) # [Msun] Upper BH mass in the BHMF. Assumes a simple power-law function for the MF with slope a_BH.
            mmax_ = self.mmax_ # Call the array for upper masses. It is used multiple times.
            
            # Interpolate for each mmax. Find the total mass and number of BHs. The integrals are precomputed.
            numerator = numpy.interp(mmax, mmax_, self.num_integral)
            denominator = numpy.interp(mmax, mmax_, self.den_integral)
            # Perform the integration. Assume a power-law BHMF, corrected for kicks. If kicks are deactivated, mb=0 by default. For a different model instead of power-law, the integrands change but also the routine for kicks in _find_mmax.
         
            return numerator / numpy.maximum(denominator, 1e-99) # [Msun]
    
    # Tidal radius. Applies for any orbit on spherically symmetric potentials. If axisymmetric are used, a different prescription is needed.
    def _rt(self, M, RG): 
        """
        Calculates the tidal radius of a star cluster based on its mass and orbital properties.

        Parameters:
        -----------
        M : float
           Total mass of the cluster [Msun].
           
        RG : float
            Galactocentric distance [kpc].

        Returns:
        --------
        float
           Tidal radius of the cluster [pc].

        Notes:
        ------
        - The tidal radius depends on the cluster's mass (`M`) [Msun], the angular velocity squared (`O2`) [1/Myrs^2], and a scaling parameter (`rt_index`) differing between galactic potentials.
        - The formula assumes a spherically symmetric potential, applicable to both circular and eccentric orbits (using an effective distance for the latter).
        """
        
        # Angular velocity squared.
        O2 = (self.Vc_func(RG) * 1.023 / (RG * 1e3)) ** 2 # [ 1 / Myrs^2]
        
        # Prefactor for tidal radius.
        nu = self.rt_index(RG)
        
        # Tidal radius.
        rt = (self.G * M / (nu * O2)) ** (1./3) # [pc] The expression is valid for circular orbits in spherically symmetric potentials. The point mass approximation for the potential of the cluster is assumed.
        
        if not self.rt_approx and self.cluster_model != 'Point_mass':
            eq = lambda x: self.dPhic_dr_func(x, M) - x * nu * O2 # Condition extracted from working on circular orbits in spherically symmetric potentials. Allows for the use of different potentials of the cluster.
            rt = fsolve(eq, x0=rt)[0] # Initial guess is the value from Point mass approximation for simplicity.
            
        return rt

    # Average mass of stars, due to stellar evolution.    
    def _mst_sev(self):
        """
        Solves the differential equations for mst and Mst over the time interval,
        considering mst = m0, Mst=M0 for t < self.tsev. 

        Returns:
        -------
        Array: The average stellar mass [Msun] as it would evolve only through stellar evolution.
        Array: The time steps of the solution [Myrs] to be used for interpolation.
        """
        
        def evolution(t, y):
            mst, Mst = y
            if not self.sev:
               return [0, 0]
            sev_value = numpy.maximum(self.sev_dict[self.sev_model](self.Z, t + 1e-99), 0)
            factor = sev_value / (t + 1e-99) * numpy.heaviside(t - self.tsev, 1)
            return [-factor * (mst - mst * self.M0 / Mst * self.nu_factor),
                -factor * (Mst - self.M0 * self.nu_factor)]

        sol = solve_ivp(evolution, [0, self.tend], [self.m0, self.M0], 
                    method=self.integration_method, t_eval=self.t_eval, 
                    rtol=self.rtol, atol=self.atol, vectorized=self.vectorize, 
                    dense_output=self.dense_output)
       
        # Return the value of mst and time as arrays. The latter will be used for interpolation.
        return sol.y[0], sol.t
    
    # Exponent connecting velocity dispersion to mass.
    def _b(self, fbh):
        """
        Calculates the exponent connecting velocity dispersion to mass.
        
        Parameters
        ----------
        fbh : float
            BH fraction.

        Returns
        -------
        b : float
            Exponent. It varies with the BH fraction.

        """
        b = self.b if not self.sigma_exponent_run else self.b_min + (self.b_max - self.b_min) * (1 - fbh) ** self.gamma_exp # No BHs, starts with equal velocity dispersion. As time flows by it approaches equipartition.
        
        return b
         
    # Friction term ψ in the relaxation. It is characteristic of mass spectrum within rh, here due to BHs. We neglect O(2) corrections in the approximate form.
    def _psi(self, fbh, M, mbh, mst): 
        """
        Calculates ψ for the cluster based on various parameters.

        Parameters:
        -----------
        fbh : float
           A factor representing the BH fraction in the range [0, 1].
    
        M : float
           Total mass of the cluster, in solar masses [Msun].
    
        mbh : float
           Average BH mass in solar masses [Msun].
    
        mst : float
           Average stellar mass [Msun].

        Returns:
        --------
        float
           The calculated ψ value, which depends on the BH fraction, cluster mass and average masses.

        Notes:
        ------
        - The function assumed that the input values are well defined (for Mbh < mbh, both should be 0, same for Mst < mst).
        - The number of particles `Np` is calculated using the `_N` function, which depends on the cluster's mass, 
          the BH fraction, and the average masses of the populations.
        - The average stellar mass `mav` is derived by dividing the total cluster mass by the number of particles.
        - The exponent `gamma` is either a constant value `b` or it evolves based on the BH fraction (if `psi_exponent_run` is True),
          starting from equal velocity dispersion for different particles and evolving towards equipartition as time passes.
        - The expression for `psi` includes a combination of parameters like `a0`, `a11`, `a12`, `b1`, `b2`, and others, 
          which relate the properties within the cluster's half-mass radius (`rh`) to the global properties of the cluster.
        - For dark clusters (`dark_clusters` is True), a more complex expression for `psi` is used, 
          incorporating contributions from both the stellar and BH populations.
        - From t=0 until tcc, a better expression may be needed, one that accounts for initial stellar mass spetrum and how the two populations evolve up until tcc. This will allow to bypass the insertion of psi0 in trh0, before tcc is computed initially.
        """
        
        # Number of particles.
        Np = M * (fbh / mbh + (1 - fbh) / mst)
       
        # Average mass and stellar mass.
        mav = M / Np # [Msun]
       
        # Exponent.
        gamma = self._b(fbh)
        
        # Result.
      #  if not self.dark_clusters:
        # Approximate form is the default option. Parameters a11, a12, b1 and b2 relate the properties within rh to the global properties.
        psi = self.a0  + self.a11 * self.a12 ** (gamma + 1) * fbh ** self.b1 * (mbh / mav) ** ((gamma + 1) * self.b2)  
        #else:
            # Total expression. Can be used for instance if the stellar population depletes first.
         #   psi = (self.a0 * (mst / mav) ** self.b0) ** (gamma + 1) * (1 - self.a11 * fbh ** self.b1) + self.a11 * self.a12 ** (gamma + 1) * fbh ** self.b1 * (mbh / mav) ** (self.b2 * (gamma + 1)) # Complete expression if we include the contribution from stars as well.
        # The prefactors defined as a0, a11, a12 are in priciple variables. The proper model requires knowledge about the number densitites to avoid this issue. This is just a trial model.
        # For fbh=0, in reality the result should be different from 0. If the remaining stellar mass spectrum is narrow, this approach suffices.  
        return psi
    
    # Relaxation as defined by Spitzer. Here we consider the effect of mass spectrum due to BHs. 
    def _trh(self, M, rh, fbh, mbh, mst):
        """
        Calculates the relaxation timescale (`trh`) for a cluster, taking into account mass, radius, 
        BH fraction and time evolution.

        Parameters:
        -----------
        M : float
          Total mass of the cluster [Msun].
    
        rh : float
           Half-mass radius of the cluster [pc].
    
        fbh : float
           The BH fraction in the range [0, 1].
    
        mbh : float
           Average BH mass [Msun].
    
        mst : float
          Average stellar mass [Msun].

        Returns:
        --------
        float
           The relaxation timescale (`trh`) of the cluster, in [Myrs].

        Notes:
        ------
        - The function assumed that the input values are well defined (for Mbh < mbh, both should be 0, same for Mst < mst, while cases where M < 0 or rh < 0 should not be used).
        - The number of particles `Np` is computed using the `_N` function, which accounts for both stars and BHs in the cluster.
        - The average mass within `rh` is computed using a power-law fit based on the total mass and the number of particles.
        - The relaxation timescale is calculated using a formula that depends on the total mass `M`, the half-mass radius `rh`, the average mass `mav`, particle number `Np`
          and the gravitational constant `G`. The timescale is further modified by the cluster's `psi` value, which depends on the BH fraction.
        - If the cluster undergoes rigid rotation (`self.rotation` is True), the relaxation timescale is adjusted to account for the effect of rotation, 
          using a simplified model. The expression by King is used.
        """
        
        # Number of particles.
        Np = M * (fbh / mbh + (1 - fbh) / mst) 
        
        # Average mass within rh. We use a power-law fit.
        mav =  self.a3 * (M / Np) ** self.b3  # [Msun]
        
        # Mass spectrum within rh.
        psi = self._psi(fbh, M, mbh, mst)
        
        # Relaxation.
        trh = 0.138 * sqrt(M * rh ** 3 / self.G) / (mav * psi * log(self.gamma * Np)) # [Myrs]. Solution stops when the number of particles is low so the logarithm is well defined.
        
       # if self.rotation: # Effect of rigid rotation.
       #     trh *= (1 - 2 * self.omega0 ** 2 * rh ** 3 / (self.G * M)) ** (3 / 2) # [Myrs] Assume constant rotation for now. In principle, omega should evolve, \dot omega/ omega = -\dot M / M - 2 \dot rh / rh + tau / I omega. Torque should vanishe if purely spherical object is assumed.
        
        return trh
        
    # Relaxation for evaporation depending on the assumptions. 
    def _tev(self, M, rh, fbh, mbh, mst): 
        """
        Calculates the evaporation timescale (`tev`) for a cluster, taking into account mass and radius.

        Parameters:
        -----------
        M : float
          Total mass of the cluster [Msun].
    
        rh : float
           Half-mass radius of the cluster [pc].
    
        fbh : float
           The BH fraction in the range [0, 1].
    
        mbh : float
           Average BH mass [Msun].
    
        mst : float
          Average stellar mass [Msun].

        Returns:
        --------
        float
           The evaporation timescale (`trhstar`) of the cluster, in [Myrs].

        Notes:
        ------
        - The function assumed that the input values are well defined (for Mbh < mbh, both should be 0, same for Mst < mst, while cases where M < 0 or rh < 0 should not be used).
        - The number of particles `Np` is computed using the `_N` function, which accounts for both stars and BHs in the cluster.
        - The average mass evaporated is estimated from the average mass of each population.
        - The relaxation timescale is calculated using a formula that depends on the total mass `M`, the half-mass radius `rh`, the average mass `mav`, particle number `Np`
          and the gravitational constant `G`. 
        - If the cluster undergoes rigid rotation (`self.rotation` is True), the relaxation timescale is adjusted to account for the effect of rotation, 
          using a simplified model assuming constant rotation.
        """
        
        # Number of particles.
        Np = M * (fbh / mbh + (1 - fbh) / mst)
     #   psi = self._psi(fbh, M, mbh, mst)
        
        # Average mass evaporated in the region.
        mev = mst #if not self.dark_clusters else mst + (mbh * psi - mst) * exp(self.k_bh * self.c_bh * (fbh - 1)) # Exponential function for smooth transition. When the BH fraction increases, the evaporation mass should change. The limit should be tev = trh. The formula neglects stellar mass spectrum.
       # A similar expression should be included for the mass and number of particles evaporated. 
       # Relaxation for evaporation.
        tev = 0.138 * sqrt(M * rh ** 3 / self.G) / (mev *  log(self.gamma * Np)) # [Myrs]
        # If this timescale is used for evaporation. It may require corrections (ψ) for the case of dark clusters, it is included through mev. 
       
     #   if self.rotation: # Effect of rigid rotation.
     #       tev *= (1 - 2 * self.omega0 ** 2 * rh ** 3 / (self.G * M)) ** (3 / 2) # [Myrs] Assume constant rotation for now.
          
        return tev
          
    # Crossing time within the half-mass radius in Myrs.
    def _tcr(self, M, rh, k=1): 
        """
        Calculates the crossing timescale (`tcr`) for a cluster, which is related to the time it takes 
        for particles to cross the half-mass radius of the cluster.

        Parameters:
        -----------
        M : float
           Total mass of the cluster [Msun].
    
        rh : float
           Half-mass radius of the cluster [pc].

        Returns:
        --------
        float
           The crossing timescale (`tcr`) of the cluster, in [Myrs].

        Notes:
        ------
        - The function assumed that the input values are well defined ( M < 0 or rh < 0 should not be used).
        - The value of `k` is adjusted for different cases, for instance at the tidal radius.
        - The crossing timescale is calculated using the formula: `tcr = k * sqrt(rh^3 / (G * M))`, where `rh` is the half-mass radius and `M` is the total mass of the cluster.
          It is derived from `tcr = 1 / sqrt(G rhoh)` or `tcr = rh / sqrt(-2E)` according to the Virial theorem. Both give the same scaling.
        """
        
        tcr = k * sqrt(rh ** 3 / (self.G * M)) # [Myrs] With rotation, the numerical values of rh, M change and this impacts crossing time indirectly.
      
        return tcr
    
    def _tdf(self, M, RG, L, rh):
        """
        Calculates the time scale for dynamical friction.
        
        Parameters
        ----------
        M : float
            Total mass of the cluster [Msun].
        RG : float
            Galactocentric distance [kpc].
        L : float
            Orbital angular momentum of cluster [pc^2 / Myrs].
        
        rh : float
             Half-mass radius of the cluster [pc].

        Returns
        -------
        float
           The time scale for dynamical friction (`tdf`) [Myrs].

        """

        X = sqrt(self.X2_func(RG)) # Ratio Vc / (sqrt(2) sigma). Does not accound for RGdot, the velocity in the radial change. It is assumed to be negligible.
        G = self.G # [pc^3/Msun Myrs^2]
        vc2 = (1.023 * self.Vc_func(RG)) ** 2 # [pc / Myrs] Velocity profile squared.
        rhoG = self.rhoG_func(RG) # [Msun / pc^3] Density of galaxy at distance r.
        log_Coulomb = log(RG * 1e3 / numpy.maximum(rh, G * M / vc2)) # Coulomb logarithm in Chandrasekhar's formula. Maximum impact parameter is taken to be the galactocentric distance, the minimum impact paramter is chosen as the maximum value of the half-mass radius since the cluster has a finite size, or the typical length scale.
        
        # Time scale for dynamical friction. Works only for Maxwellian distribution. The anisotropic model has the same dependence on X, only its value changes.
        tdf = L * vc2 / (4 * pi * 1e3 * RG * G ** 2 * M * rhoG * log_Coulomb * (erf(X) - 2 * X / sqrt(pi) * exp(- X ** 2)))
        # Future extension: Define g(X) for two cases, for isotropic and anisotropic velocity distribution (isotropic model is erf(X) - 2 * X / sqrt(pi) * exp(- X ** 2)), and include it in the time scale. Then for anisotropic models, call the respective X from a different dictionary depending on the anisotropic model, for instance constant, power-law (or combine these two).
        # Most configurations that are currently available have analytic forms for constant anisotropy. Power-law has some analytic forms, only if the anisotropy length scale is equal to the configuration length scale which is not always the case.
        return tdf
    
    def _tdis(self, M):
        """
        Computes the disruption time scale for interactions with GMC.
        
        Parameters
        ----------
        M : float
            Mass of the cluster [Msun]

        Returns:
        -------
        float
            Disruption time scale [Myrs].

        """
        
        # Time scale for disruption.
        tdis = 2e3 * (5.1 / (self.Sigma_n * self.rho_n)) * (M / 10 ** 4) ** self.gamma_n # [Myrs] 
        
        return tdis
    
    def _tshock(self, M, rh, RG):
        """
        Estimates the time scale for tidal shocks produced y the disk
        Parameters
        ----------
        M : float
            Mass of the cluster [Msun].
        rh : float
            Half-mass radius [pc].
        RG : float
            Galactocentric distance [kpc].

        Returns
        -------
        float
            Time scale for disk shock [Myrs].
            
        """
        
        Vc = 1.023 * self.Vc_func(RG) # [pc / Myrs] Velocity profile.
        
        # Disk density decreases exponentially with the galactocentric distance.
        Sigma_disk = self.Sigma_disk0 * exp(- RG / self.Rd) # [Msun / pc ^2] Surface density. 
        
        # Time scale for shocks.
        tshock = M * RG * 1e3 * Vc / (40 * pi * self.G * rh ** 3 * Sigma_disk ** 2) # [Myrs]
        
        return tshock
        
    # Central escape velocity.
    def _vesc(self, M, rh):
        """
        Calculates the central escape velocity (`vesc`) for a cluster, based on its total mass and half-mass radius.

        Parameters:
        -----------
        M : float
           Total mass of the cluster [Msun].
    
        rh : float
           Half-mass radius of the cluster [pc].

        Returns:
        --------
        float
           The escape velocity (`vesc`) at the center of the cluster, in [km/s].

        Notes:
        ------
        - The density `rhoh` is computed as the mass enclosed within the half-mass radius (`rh`), using the formula: 
          `rhoh = 3 * M / (8 * pi * rh ** 3)`. The density is in units of [Msun / pc^3].
        - The escape velocity `vesc` is then calculated using the relation: 
          `vesc = 50 * (M / 1e5) ** (1/3) * (rhoh / 1e5) ** (1/6)`, where the units for `vesc` are in [km/s]. 
          This formula expresses the central escape velocity based on the mass and the density of the cluster.
        - The escape velocity is further augmented by multiplying it by the factor `fc`, which adjusts the value according to the specific King model used for the cluster.
        - Time dependence if `fc` is neglected, however it should increase over time. This is introduced, currently as a trial, in a different function.
        """
        
        # Density.
        rhoh = 3 * M / (8 * pi * rh ** 3) # [Msun / pc^3]
        
        # Escape velocity.
        vesc = 50 * (M / 1e5) ** (1./3) * (rhoh / 1e5) ** (1./6) # [km/s] Central escape velocity as a function of mass and density. If it is needed in [pc / Myrs], multiply with 1.023.
        vesc *= self.fc # Augment the value for different King models. This in principle evolves with time but the constant value is an approximation.
       
        return vesc

    # Evolution of prefactor fc over time. A trial dependence on the macroscopic properties of the cluster is assumed.
    def _fc(self, M, rh):
        
        """
        Estimates the change in parameter fc, and effectively on King's parameter W0, and its impact on the central escape velocity.
        
        Parameters:
        -----------
        M : float
           Total mass of the cluster [Msun].
    
        rh : float
           Half-mass radius [pc]
        
        Returns:
        --------
        float
           Multiplication factor for the central escape velocity. In principle, this parameter scales with sqrt(W0).
        
        """
        
        fc = (self.M0 / M) ** self.d1 * (rh / self.rh0) ** self.d2 # Such parameter should increase because the central region becomes denser over time. Will be studied further in the future.
        
        return fc
    
    # Construct the differential equations to be solved. 
    def _odes(self, t, y):
        """
        Computes the time derivatives of the system's state variables, which describe the evolution of a star cluster 
        under various physical processes including stellar evolution, tidal effects, ejections, mass segregation, and core collapse.

        Parameters:
        -----------
        t : float
           Time elapsed since the beginning of the cluster's evolution, in [Myrs].
    
        y : array
           A sequence representing the state variables:
           - y[0] : float : Stellar mass, `Mst` [Msun].
           - y[1] : float : Black hole mass, `Mbh` [Msun].
           - y[2] : float : Half-mass radius, `rh` [pc].
           - y[3] : float : Parameter describing mass segregation, `Mval`.
           - y[4] : float : Average stellar mass `mst` [Msun].
           - y[5] : float : Galactocentric distance `RG` [kpc].

        Returns:
        --------
        numpy.array
           An array containing the time derivatives of the state variables:
           - `Mst_dot`: Stellar mass loss rate [Msun/Myr].
           - `Mbh_dot`: BH mass loss rate [Msun/Myr].
           - `rh_dot` : Rate of change of the half-mass radius [pc/Myr].
           - `Mval_dot` : Evolution of the mass segregation parameter [1/Myr].
           - `mst_dot`: Average stellar mass loss rate [Msun/Myr]
           - `RG_dot` : Rate of change of galactocentric distance [kpc/Myr].

        Notes:
        ------
        - This function models the dynamical evolution of the star cluster using multiple physical processes.
        - The calculations include:
        1. **Core Collapse**:
         - After core collapse time (`tcc`), the parameter for mass segregation, `Mval`, transitions to a constant value. In the future, where a smoothing transition will be inserted, this contribution will not be needed.
        2. **Stellar Evolution**:
         - Stellar mass loss due to stellar evolution is modeled using mass-loss rate (`nu`). 
         - Mass segregation is evolved if the option is selected.
         - The average stellar mass also evolves due to sev. It accounts the proper solution, if a running ν is selected.
         - RV filling clusters can be described as well. An induced mass loss rate is available.
        3. **Tidal Effects**:
         - If tides are active, they contribute to mass loss and affect the half-mass radius evolution.
         - Finite escape time and rotation are also included in the tidal mass-loss calculation if selected. Rotation is implemented in a naive way and more study is needed.
         - The impact of tides on the average stellar mass can be included. Should affect low galactocentric distances and light star clusters mainly.
        4. **BH Mass Loss**:
         - BH mass loss includes mechanisms for ejection based on cluster parameters and equipartition states. Tidal effects are also available but not properly implemented, as the average BH mass remains unaffected. More study is needed.
        5. **Cluster Expansion**:
         - Expansion due to energy loss is modeled using Hénon's parameter (`zeta`). A constant value is assumed. In the future, a smoother connection will be made between the unbalanced and the balanced phase.
         - The contribution of evaporating stars to half-mass radius changes is also included, if the option is activated.
        6. **Combined Mass Loss**:
         - Total mass loss and its impact on the cluster size are calculated for both stars and BHs.
        7. Additional cases such as interactions with GMCs or tidal spirallings are available should the user wish to activate them.
        - The returned derivatives (`Mst_dot`, `Mbh_dot`, `rh_dot`, `Mval_dot`, `r_dot`, `mst_dot`, `RG_dot`) can be used in numerical integrators to evolve the cluster state over time.
        """
        
        Mst = y[0] # [Msun] Mass of stars.
        Mbh = y[1] # [Mun] Mass of BHs.
        rh = y[2] # [pc] Half-mass radius.
        Mval = y[3] # Parameter for mass segregation.
        mst = y[4] # [Msun] Average stellar mass.
        RG = y[5] # [kpc] Galactocentric distance.
        
        # Time instances are repeated multiple times. It is faster to assign them in local variables.        
        tcc = self.tcc # [Myrs] Core collapse.
        tsev = self.tsev # [Myrs] Stellar evolution.
     #   tbh = self.t_bhcreation # [Myrs] Time instance when BHs have been created.

        mbh = self._mbh(Mbh) # [Msun] Extract the average BH mass.
        mst_inf = self.mst_inf # [Msun] Highest mass of remnant stars.
        if 0 < Mbh < mbh: Mbh, mbh = 1e-99, 1e-89 # Useful statements for functions to be called later.
        if 0 < Mst < mst_inf: Mst, mst = 1e-99, 1e-89 # Suggests that all stars have left the cluster
        if rh < 0: rh = 1e-10 # Half-mass is strictly non-negative.
       
        M = Mst + Mbh # [Msun] Total mass of the cluster. It is overestimated a bit initially because we assume Mbh > 0 from the start.
        fbh = Mbh / M if M > 0 else 0 # Fraction of BHs. Before core collapse is it overestimated. It is corrected after the first few Myrs.
        
        Np = M * (fbh / mbh + (1 - fbh) / mst) # Total number of particles.
        m = M / Np # [Msun] Total average mass.
        
        trh = self._trh(M, rh, fbh, mbh, mst) # [Myrs] Relaxation.
        tcr = self._tcr(M, rh, k=0.138) # [Myrs] Crossing time. Prefactor is 0.138 to cancel with relaxation in ratios.
        
        Mst_dot = rh_dot = Mbh_dot = 0 # [Msun / Myrs], [pc / Myrs], [Msun / Myrs] At first the derivatives are set equal to zero.
        Mval_dot = mst_dot = 0 # [1 / Myrs], [1 / Myrs], [Msun / Myrs] Derivatives for parameter of mass segregation, ratio rh / rv and average stellar mass.
       
        zeta = self.zeta # Rate of energy emission each relaxation.
        G, x = self.G, self.x # Gravitational constant and exponent for finite escape time. Appear multiple times.
        kin, index = 1, 0 # Parameters that reappear. Used for the energy budget.
        psi_st = 1 # Contribution of stars in ψ. Assumed to be constant.
        F = self.balance_function(t) # Balancing function used to smoothly connect the two periods, (prior to post core collapse).
        cg , rt_index, r, rt = 0, 1, self.r, 10 ** 6 # Parameters related to energy of the cluster due to the tidal field and tidal radius. 
        
        cg_factor = 0 # Combination of parameters, used for estimating the effect of the tidal field on the energy of the cluster.
        self.rhrt_crit = 1 # Critical ratio for rh/rt beyond which the result is unpgysical.
        
        Mst_dotev, mst_dotev = 0, 0 # [Msun / Myrs] Rates of change of stellar mass due to tides.
        index_cg, denom = 1, 1
        
        # Add tidal mass loss.
        if self.tidal: # Check if we have tides. 
           cg , rt_index = self.cg, self.rt_index(RG) # Change their values. They are important only if we have tides and if they are activated.
           
           rt = self._rt(M, RG) # [pc] Tidal radius.
           tev = self._tev(M, rh, fbh, mbh, mst) if self.two_relaxations else trh # [Myrs] Evaporation time scale. Check whether it is different from relaxation within rh.
          
           cg_factor = 4 * cg / r * (rh / rt) ** 3 # Factor depends on rh / rt so it becomes important only towards the end of the life of a cluster. 
           denom = 1 - 2 * cg_factor # Denominator that appears multiple times.
           if cg > 0: self.rhrt_crit = (r / (8 * cg)) ** (1 / 3) # The critical ratio decreases if cg > 0.
           
           xi = max(self.xi_function(rh, rt), self.fmin) # Tides. The max option allows the user to insert a constant evaporation if the tidal function is weak for small rh/rt.
           xi *= (self.M0 / 2e5) ** (1 - self.ym) # Correction term to xi. To be used for dissolving clusters. Default to ym=1.
        
           if self.finite_escape_time: # Check if evaporation is energy dependent.
              Omega = (1.023 * self.Vc_func(RG) / (1e3 * RG)) # [1 / Myrs] Angular frequency.
              lambda1, lambda3 = Omega ** 2 - self.d2Phi_dr2_func(RG), - Omega ** 2 # [1 / Myrs^2] Eigenvalues used for specifying the evaporation rate. Considers differences in galactic models.
             
              P = self.p * (tev / tcr) ** (1 - x)  # Check if we have finite escape time from the Lagrange points.
              xi *= P / (1 - lambda3 / lambda1) ** (0.5 * (1 - x)) # Perhaps the denominator needs a factor of 3 / 4 so that the contribution is 1 for point mass?
              
      #     if self.rotation: # If the cluster rotates, the evaporation rate changes. A similar change to relaxation is used. Trial stage.
      #         xi *= (1 - 2 * self.omega0 ** 2 * rh ** 3 / (G * M)) ** self.gamma1 # Parameter gamma1 should be positive, since rotation makes it more difficult to reach the tail of the Maxwellian distribution. 
             
           # Effect of evaporating stars on the half mass radius. Keep only evaporation.   
           if self.escapers: # It is connected to the model for the potential of the cluster. 
              index = numpy.abs(self.Phic_function(rt, M)) / (G * M) # [pc^2 / Myrs^2 ] Get the value from the dictionary. It changes with respect to the potential assumed for the cluster.
              kin = self.kin if not self.varying_kin else 2 * (tcr / tev) ** (1 - x)  # Kinetic term for evaporating stars. Can be either a constant value or effectively a function of the number of particles.
              
           if Mst > mst_inf: # Tidal mass loss appears only when stars are present.                 
              mst_dotev += self.chi_ev * (1 - self.m_breaks[0] / mst) * (1 - mst / mst_inf) * xi * mst / tev # [Msun / Myrs] Rate of change of average stellar mass due to tides. Can be avoid if the SSP tools were used and mass subtraction was performed bottom-up, however solely for tidal mass loss. 
              Mst_dotev -= xi * Mst / tev # [Msun / Myrs] Mass loss rate of stars due to tides (and central ejections).
              rh_dot += 2 * Mst_dotev / M * rh * (1 + cg_factor / 2) / denom  # [pc / Myrs] Impact of tidal mass loss to the size of the cluster.
              rh_dot += 6 * xi / r / tev * (1 - kin) * rh ** 2 * index * Mst / M  / denom  # [pc / Myrs] If escapers carry negative energy as they leave, the half mass radius is expected to increase since it is similar to emitting positive energy. The effect is proportional to tides.
           """   
           # If the tidal field was important, an additional mass-loss mechanism would be needed. Trial stage.
           if self.dark_clusters and Mbh > mbh and t >= tbh: # Appears when BHs have been created.
              psi_st = (self.a0 * (mst / m) ** self.b0) ** (self._b(fbh) + 1) * (1 - self.a11 * fbh ** self.b1) # Contribution of stars in psi.
              gbh = exp(- self.c_bh * (1 - fbh)) # A simple dependence on fbh is shown. Trial model, will be studied extensively in the future.
              xi_bh = xi * gbh # Modify the tidal field that BHs feel such that it is negligible when many stars are in the cluster, and it starts increasing when only a few are left. 
              Mbh_dot -= xi_bh * Mbh / tev # [Msun / Myrs] Same description as stellar mass loss. Use relaxation in the outskirts. When BH eveporation becomes important, psi=1 approximately so the expressions should match.
             # mbh should increase with the tidal field. Another prescription is needed for mbh, or neglect it approximately. 
             # Now if BH evaporation is important and escapers is activated, it should be included here.
              rh_dot += 6 * xi_bh / r / tev * (1 - kin) * rh ** 2 * index * fbh / (1 - 2 * cg_factor)   # [pc / Myrs] The kinetic term is the same here, it does not distinguish particle types.
            """
        # Unbalanced phase. 
        
        if self.mass_segregation:
           Mval_dot += Mval * (self.Mvalf - Mval) / trh * numpy.heaviside(tcc - t, 0)  # [1 / Myrs] Evolution of parameter describing mass segregation. 
        
        # Balanced Phase.
             
        rh_dot += zeta * rh / trh * F * (1 + index_cg * cg_factor) / denom # [pc / Myrs] Instead of the if statement, simply multiplying with F should work.
        
        S = max(self._psi(fbh, M, mbh, mst) - psi_st, self.Scrit) # Parameter similar to Spitzer's parameter used for equipartition. A threshold is inserted, however it is 0 by default.
        beta_f = self.beta_function(S) # Factor used to modulate BH and stellar ejections. Depends on Spitzer´s parameter, and becomes constant when equipartition is reached.
        beta_bh = 0 # Ejection rate of BHs.
        
        if Mbh > mbh: # Check if BHs are present. BHs evolve only in the balanced phase and lose mass through ejections.
                  
            beta_bh = self.beta * F # Ejection rate of BHs. Initially it is set as a constant. It should be changed when fbh increases a lot and the light component does not dominate anymore.
            # If tcc < tsev, tbh, it ejects BH progenitors.
            
            if fbh < self.fbh_crit and self.running_bh_ejection_rate_1: # Condition for decreasing the ejection rate of BHs due to E / M φ0. A condition for mbh_crit may be needed.
                beta_bh *=  (fbh / self.fbh_crit) ** self.b4 * (mbh / m / self.qbh_crit) ** self.b5 # A dependence on the average mass ratio mbh / m (or metallicity) may be needed.
           
            if self.running_bh_ejection_rate_2: # Decrease the ejection rate for clusters that are close to reaching equipartition.
               beta_bh *= beta_f # Change the ejection rate with respect to S.
               
            Mbh_dot -= beta_bh * zeta * M / trh  # [Msun / Myrs] Ejection of BHs each relaxation. 
            rh_dot += 2 * Mbh_dot / M * rh * (1 + cg_factor / 2) / denom # [pc / Myrs] Contraction since BHs are removed.
            
        Mst_dotgmc, mst_dotgmc = 0, 0 # [Msun / Myrs] Mass loss rates from interactions with Giant Mollecular Clusters. Introduces additional mass-loss. 
        Mst_dotdisk, mst_dotdisk = 0, 0 # [Msun / Myrs] Mass loss rates from disk shocks.
        Mst_dotej, mst_dotej = 0, 0 # [Msun / Myrs] Mass loss rates from stellar ejections.
       
        if Mst > mst_inf: # A proper description should start with zero alpha_c, increase slightly up until core collapse, be a function of fbh, mbh and when we have no BHs it maximizes.
            
            alpha_c = self.alpha_ci * F # Initial ejection rate of stars. With F, it should be alpha_c += self.alpha_ci * F 
           
            if self.running_bh_ejection_rate_2: # Check if the ejection rate varies with the BH population.
                alpha_c += (self.alpha_cf * F - alpha_c) * (1 - beta_f) # This expression states that with many BHs, the ejection rate of stars should be small, and when the BHs vanish it increases. A similar rate of change as for the BH ejection rate is used. 
           
            if self.running_stellar_ejection_rate: # Check if the ejection rate decreases for increasing tides.
                alpha_c *= (1 - 5 * xi / (3 * zeta)) # Alter alpha_c based on tides as introduced in EMACSS.
               
            Mst_dotej -= alpha_c * zeta * M / trh # [Msun / Myrs] Contribution of central stellar ejections.
            mst_dotej -= alpha_c * zeta * self.chi_ej * (mst - self.m_breaks[0]) / trh  # [Msun / Myrs] Mass loss rate should be top_down. This is a trial method for decreasing the average stellar mass.  
            rh_dot += 2 * Mst_dotej / M * rh * (1 + cg_factor / 2) / denom  # [pc / Myrs] Impact of ejections to the size of the cluster.
            """
            if self.GMC: # Check if interactions with molecular clusters have been activated. Trial stage. BHs are not affected for now.
                tdis = self._tdis(M) # [Myrs] Applied to stellar mass only. It is inserted continuously.
                mst_dotgmc += self.chi_gmc * (1 - self.m_breaks[0] / mst) * (1 - mst / mst_inf) * mst / tdis # [Msun / Myrs] Average stellar mass increases since low mass stars vanish.
                Mst_dotgmc -= Mst / tdis # [Msun / Myrs]
                rh_dot += 2 * Mst_dotgmc / M * rh # [pc / Myrs]
            # Similar extension for tidal shocks can be inserted.
            
            if self.disk: # Trial stage. BHs are not affected.
                tshock = self._tshock(M, rh, RG) # [Myrs] Time scale for shock
                Mst_dotdisk -= self.xi_shock * Mst / tshock # [Msun / Myrs] Trial model assumes a constant prefactor.
                rh_dot += 2 * Mst_dotdisk / M * rh + rh / tshock # [pc / Myrs] Second term comes from the fact that shocks provide energy, cluster expands, some particles may evaporate.
                mst_dotdisk += self.chi_disk * (1 - self.m_breaks[0] / mst) * (1 - mst / mst_inf) * mst * self.xi_shock / tshock # [Msun / Myrs]
            """
        Mst_dotsev, mst_dotsev = 0, 0 # [Msun / Myrs] Mass loss rates from stellar evolution.
        Mst_dotsev_ind = 0 # [Msun / Myrs] Induced mass loss rate for RV filling clusters. 
        
        # Stellar mass loss. Impact of stellar winds.
        if self.sev and t >= tsev and Mst > mst_inf: # This contribution is present only when Mst is nonzero.
            nu = numpy.max(self.nu_function(self.Z, t), 0) # Rate of stellar mass loss due to stellar winds. Taken from a dictionary, ensures that it is non-negative.
            mst_dotsev -= nu * (mst - mst * self.M0 / Mst * self.nu_factor) / t # [Msun/Myrs] When we consider stellar evolution, the average stellar mass changes through this differential equation. It is selected so that the case of a varying nu is properly described.
            Mst_dotsev -= nu * (Mst - self.M0 * self.nu_factor) / t  # [Msun / Myrs] Stars lose mass due to stellar evolution. If it is the sole mechanism for mass loss, it implies that N is constant since mst decreases with the same rate.
            rh_dot -= (Mval * (1 + index_cg * cg_factor) - 2 * (1 + cg_factor / 2)) / denom *  Mst_dotsev / M * rh # [pc / Myrs] The cluster expands for this reason. It is because a uniform distribution is assumed initially. 
            
            if self.induced_loss and self.tidal and rh / rt > self.Rht_crit: # Check if the cluster is RV filling. Not important at large time scales. Trial stage.
                t_delay = self.n_delay * self._tcr(M, rt, k=1) # [Myrs] Compute the crossing time at the tidal radius.
                f_ind = self.find_max * (1 - exp(- t / t_delay)) # Function for induced mass loss.
                Mst_dotsev_ind += f_ind * Mst_dotsev # [Msun / Myrs] Add this contribution to the stellar mass loss.
                rh_dot += 2 * Mst_dotsev_ind / M * rh * (1 + cg_factor / 2) / denom # [pc / Myrs] If it is, the half-mass radius decreases due to induced mass loss. It is derived from conservation of energy.
            
        Mst_dot += Mst_dotev + Mst_dotsev + Mst_dotej + Mst_dotsev_ind + Mst_dotgmc + Mst_dotdisk # [Msun / Myrs] Correct the total stellar mass loss rate. This way, resummation in rhdot is avoided.
        mst_dot += mst_dotev + mst_dotsev + mst_dotej + mst_dotgmc + mst_dotdisk # [Msun / Myrs] Correct the average stellar mass loss rate by considering all contributions.
        
        RG_dot = 0 # [kpc / Myrs]
        
        # Condition to check whether the galactocentric distance changes due to inspiral.
        if self.tidal_spiralling and self.tidal:
            L = 1.023 * self.Vc_func(RG) * 1e3 * RG # [pc^ / Myrs] Angular frequency of the orbit. At each point the angular frequency changes with radius.
            tdf = self._tdf(M, RG, L, rh) # [Myrs] Time scale for dynamical friction.
            RG_dot -= 2 / (4 - self.rt_index(RG)) * RG / tdf * numpy.heaviside(RG - 0.01, 0) # [kpc / Myrs] Galactocentric distance cannot be negative. Truncate to the tidal radius for small values.
            rh_dot += rh * RG_dot / RG * cg_factor / denom * (- rt_index + (rt_index - 3) / rt_index * (rt_index - 2) ) # [pc / Myrs] Should be rt_index - 2 + RG MG''(RG) / MG'(RG), MG taken from the galactic profile. To be included in the future for the rest potentials. Now works only for SIS.
        
        derivs = numpy.array([Mst_dot, Mbh_dot, rh_dot, Mval_dot, mst_dot, RG_dot], dtype=object) # Save all derivatives in a sequence.
        
        return derivs # Return the derivatives in an array.

    # Extract the solution using the above differential equations.
    def _evolve(self, N, rhoh):
        """
        Simulates the dynamical evolution of a star cluster over time using the specified initial conditions and physical models.

        Parameters:
        -----------
        N : int
          Total number of stars in the cluster initially.
        rhoh : float
          Initial half-mass density of the cluster [Msun/pc^3].

        Method:
        -------
        - Uses `solve_ivp` to numerically integrate the system's differential equations (`odes` method).
        - Implements an event to terminate the simulation when the stellar mass drops below a minimum threshold (`Mst_min`) and the BH population is also below a threshold (`Mbh_min`).

        Output:
        ------
        - Cluster parameters are computed and can be saved in a text file if selected.
        
        Notes:
        ------
        - Initial conditions for stellar mass, BH mass, half-mass radius, segregation parameter, ratio `rh/rv` and average stellar mass are set to 
          `self.M0`, `self.Mbh0`, `self.rh0`, `self.Mval0`, `self.r0`, `self.m0` respectively.
        - Results are stored as attributes of the object for further analysis.
        """
        
        Mst = [self.M0] # [Msun] Initial stellar mass.
        Mbh = [self.Mbh0] # [Msun] Initial BH mass.
        rh = [self.rh0] # [pc] Initial half-mass radius.
        Mval = [self.Mval0] # Initial parameter for mass segregation.
        mst = [self.m0] # [Msun] Initial average stellar mass.
        RG = [self.rg] # [kpc] Galactocentric distance.
       
        y = [Mst[0], Mbh[0], rh[0], Mval[0], mst[0], RG[0]] # Combine them in a multivariable.

        def event(t, y):
            # Check if either condition is still valid.
            condition_1 = y[0] - self.Mst_min  # Positive if stars are above the threshold.
            condition_2 = y[1] - self.Mbh_min  # Positive if BHs are above the threshold.

            # Event function returns the maximum of the two conditions. Stops only when both are negative.
            return numpy.max([condition_1, condition_2]) # Maximum is chosen because it suggests that this population dominates.
       
        def tidal_overflow_event(t, y):
            # Condition for dissolved clusters. Future extension.
            if self.tidal:
                rt = self._rt(y[0] + y[1], y[5])  # [pc] Tidal radius.
                return -y[2]/rt + self.rhrt_crit  # Stop the integration in unphysical regimes.
            else:
                return 1.0  # Does not trigger the effect

        tidal_overflow_event.terminal = True  # Stop integration
        tidal_overflow_event.direction = -1   # Trigger when crossing from positive to negative

        # Configure the event properties.
        event.terminal = True # Stop integration when the event is triggered.
        event.direction = -1 # Trigger the event from above. Means that the solution y is large initially and it stops when the difference becomes negative.

        # Solution.
        sol = solve_ivp(self._odes, [0, self.tend], y, method=self.integration_method, t_eval=self.t_eval, events=[event, tidal_overflow_event], rtol=self.rtol, atol=self.atol, dense_output=self.dense_output, vectorized=self.vectorize) 

        self.t = sol.t / 1e3 # [Gyrs] Time.
        self.Mst = sol.y[0] # [Msun] Stellar mass.
        self.Mbh = sol.y[1] # [Msun] BH mass.
        self.rh = sol.y[2] # [pc] Half-mass radius.
        self.Mval = sol.y[3] # Parameter for segregation.
        self.mst = sol.y[4] # [Msun] Average stellar mass.
        
        self.mbh = numpy.array([self._mbh(x) for x in self.Mbh]) # [Msun] Average BH mass.
        mask_Mbh, mask_Mst = self.Mbh < self.mbh, self.Mst < self.mst_inf 
        self.Mbh[mask_Mbh] = 1e-99 # [Msun] BH mass corrected for mbh > Mbh.
        self.mbh[mask_Mbh] = 1e-89 # [Msun] Correct the average BH mass.
        
        self.Mst[mask_Mst] = 1e-99 # [Msun] Stellar mass corrected for low stellar masses.
        self.mst[mask_Mst] = 1e-89 # [Msun] Correct the average stellar mass.
        
        # Quantities for the cluster.
        self.M = self.Mst + self.Mbh # [Msun] Total mass of the cluster. BHs are included already. A small error in the first few Myrs is present.
        self.fbh = self.Mbh / self.M # BH fraction.
        self.mbh_max = numpy.array([self._find_mmax(x) for x in self.Mbh]) # [Msun] Maximum BH mass in the BHMF as a function of time. 
        self.mbh_max = numpy.where(self.Mbh >= self.mbh, self.mbh_max, 0) # Correct the upper BH mass to account for the case of no BHs left in the cluster.
        
        # Further properties.
        self.psi = self._psi(self.fbh, self.M, self.mbh, self.mst) # Friction term ψ.
        self.Np = self.M * ( self.fbh / self.mbh + (1 - self.fbh) / self.mst) # Number of components. 
        self.mav = self.M / self.Np # [Msun] Average mass of cluster over time, includes BHs. No significant change is expected given that Nbh <= O(1e3), apart from the beginning where the difference is a few percent.
        self.mev = self.mst if not self.dark_clusters else self.mst + (self.mbh * self.psi - self.mst) * 2  / (1 + exp(self.k_bh * self.c_bh * (1 - self.fbh))) # [Msun] Average mass evaporated.
        self.Nbh = self.Mbh / self.mbh # Number of BHs.
        self.E = - self.r * self.G * self.M ** 2 / (4 * self.rh) # [pc^2 Msun / Myrs^2] External energy of the cluster
        
        if self.tidal: 
            self.RG = sol.y[5] # [kpc] Galactocentric distance.
            self.Ltot = 1.023 * self.Vc_func(self.RG) * 1e3 * self.RG if self.tidal else None # [pc^2 / Myrs] Angular frequency of the model.
            self.rt = numpy.vectorize(self._rt)(self.M, self.RG) # [pc] Tidal radius.
            self.xi = numpy.maximum(self.xi_function(self.rh, self.rt), self.fmin) # Evaporation rate.
            self.tev = self._tev(self.M, self.rh, self.fbh, self.mbh, self.mst) # [Myrs] Relaxation within rh. Describes evaporation.
            if self.tidal_spiralling: self.tdf = self._tdf(self.M, self.RG, self.Ltot, self.rh) # [Myrs] Dynamical friction time scale.
            self.phi_c = self.Phic_function(self.rt, self.M) # [pc^2/Myr^2] Cluster potential at the tidal radius.
            self.Vg = - self.cg / self.rt_index(self.RG) * self.G * self.M ** 2 / self.rh * (self.rh / self.rt) ** 3
        else:
            self.RG = numpy.full(len(self.t), numpy.inf) # [kpc] Galactocentric distance is set at infinity.
            self.rt = numpy.full(len(self.t), numpy.inf) # [pc] No tidal radius ir predicted for isolated clusters.
            self.phi_c = numpy.full(len(self.t), 0) # [pc^2/Myrs^2] The potential vanishes at infinity.
        
        
        self.trh = self._trh(self.M, self.rh, self.fbh, self.mbh, self.mst) # [Myrs] Relaxation within rh. Describes ejections.
        self.tcr = self._tcr(self.M, self.rh, k=self.tcross_index) # [Myrs] Crossing time.
        if self.GMC: self.tdis = self._tdis(self.M) # [Myrs] Disruption time scale.
        self.vesc = self._vesc(self.M, self.rh) # [km/s] Escape velocity. 
        self.b_run = self._b(self.fbh) # Exponent connecting velocity dispersion to mass.
        self.S = self.a11 * self.a12 ** (1 + self.b_run) * self.fbh ** self.b1 * (self.mbh / self.mav) ** ((1 + self.b_run) * self.b2) # Parameter indicative of equipartition.
        self.phi_0 = - (1.023 * self.vesc) ** 2 / 2 # [pc^2/Myr^2] Central potential. Needs fc to be correct.
        self.fcrun = self._fc(self.M, self.rh) # Parameter to estimate how the escape velocity evolves.
        self.beta_run = self.beta * self.beta_function(self.S) # Running beta factor.
        self.nu_run = numpy.maximum(self.nu_function(self.Z, sol.t), 0) if self.sev else 0 # Running stellar mass loss rate.
        self.sigma = 1.023 * sqrt(0.2 * self.G * self.M / self.rh) # [km/s] Velocity dispersion in the cluster.
        
        # Checks if the results need to be saved. The default option is to save the solutions of the differential equations as well as the tidal radius and average masses of the two components. Additional inclusions are possible.
        if self.output:

            # Defines table header.
            table_header = "# t[Gyrs] Mbh[msun] Mst[msun] rh[pc] rt[pc] mbh[msun] mst[msun] mbh_max[msun]"

            # Prepares data for writing.
            data = numpy.column_stack((self.t, self.Mbh, self.Mst, self.rh, self.rt, self.mbh, self.mst, self.mbh_max))

            # Writes data.
            numpy.savetxt(self.outfile, data, header=table_header, fmt="%12.5e", comments="")

#######################################################################################################################################################################################################################################################
"""
Notes:
- Another set of values that can be used is:
    zeta, beta, n, Rht, ntrh, b, Mval0, S0 = 0.0977, 0.0566, 0.7113, 0.0633, 0.1665, 2.14, 3.1156, 1.96.

- Functions _mstdot_ev, _balance and _fc are works in progress, currently deactivated or unused.
- self.virial_evolution, self.finite_escape_time, self.rotation, self.dark_clusters are future statemens that require refinement.
- Towards small values of Mbh, the expected values of mbh, mbh_max do not coincide, the difference is a few solar masses.

Use:
- An isolated cluster can be described with tidal=False. It can have stellar ejections or not.
- Simple cases where we have no stellar evolution can be described with nu=0 or sev=False.
- If we start with BHs in the balanced phase at t=0, use ntrh=0.
- If no change in the BH ejection rate is wanted, use a value for S0 quite close to 0 or disable running_bh_ejection_rate.
- A richer IMF can be specified by assigning values to additional mass breaks for the intervals, slopes and bins. Default option is 3. Masses must increase. If the upper part is similar to Kroupa, new solutions for Mst, mst are available should the user wish to, otherwise a new value for nu can be inserted. Disable sev_tune.

Limitations:
- Dissolved clusters cannot be described. Tidal effects must be considered in the stellar average mass. A numerical value for chi is needed.
- Dissolution occurs when departure from equilibrium is observed. This happens when rh/rt surpasses the value (mu rt_index(RG) / 8) ** (1 / 3) where mu is the ratio between the energy of an isolated cluster with the potential due to its motion inside a tidal field Vg.
- Description of dark clusters has not been studied extensively. 
- Rotation, if activated, needs to be more accurate since now it is a simple rigid model with constant rotation.


Orbit:
- For a given potential, in the function _odes the following part can be included.

 RG, RGdot = y[6], y[7] # [kpc], [kpc/Myrs] Motion of the cluster.
 L = y[9] # [pc^2/Myrs] Angular momentum per unit mass.
 
 dRG_dt, dRGdot_dt, dtheta_dt = 0, 0, 0 # Derivatives for the motion of the cluster.
 dL_dt = 0 # [pc^2/Myrs^2] Rate of change of angular momentum per unit mass.
 
 Mg = self.Mg # [Msun] Mass of galaxy changes when the orbit changes.
 dPhi_dr = self.dPhi_dr_func(RG, Mg) # [pc / Myrs^2] Derivative of the potential. It takes parsec as input.

 # Orbit.
 if self.eccentric and self.e > 0 and self.tidal:

    dRG_dt += RGdot # [pc / Myrs] Velocity of cluster.
    dRGdot_dt += 1e-3 * (- dPhi_dr + (L ** 2 / (RG * 1e3) ** 3)) # [pc / Myrs^2] Newton's law of motion for the center of the cluster inside a galaxy. Multiply with 1e-3 because the left hand side is kpc.
    dtheta_dt += L / (RG * 1e3) ** 2 # [rad / Myrs] Rate of change of the angle.
 .
 .
 .
 .
 .
 .
 # Condition to check whether the galactocentric distance changes due to inspiral.
 if self.tidal_spiraling and self.tidal:
     tdf = self._tdf(M, RG, Np, L) # [Myrs] Timescale for dynamical friction.
     # For eccentric orbits, we vary direcly the angular momentum.
     if self.eccentric and self.e > 0:
         dL_dt -= L / tdf * numpy.heaviside(L - 1, 0)
     # Circular orbits can be simplified and the derivative of the galactocentric distance is computed. The timescale remains the same.
     else:
         L = 1e3 * RG * self.Vc_func(RG) # [pc^2 / Myrs] Angular momentum, should change because we change the distance.
         # Equation for distance is extracted from the variation of the angular momentum. 
         dRG_dt -= 2 / (4 - self.rt_index(RG)) * RG / tdf * numpy.heaviside(RG - 0.01 , 0) # [kpc / Myrs] Prefactor appears to be connected to the tidal radius. It is due to the scaling of the velocity profile with the radius.
              
 derivs = numpy.array([Mst_dot, Mbh_dot, rh_dot, Mval_dot, r_dot, mst_dot, dRG_dt, dRGdot_dt, dtheta_dt, dL_dt], dtype=object) # Save all derivatives in a sequence.
   
This describes how a globular cluster moves, by assuming the center of its mass satisfies Newton's equation. The tidal mass loss must be changed however, because rt and tev should be different. The equations are for RG, RGdot and theta (angle in the plane of motion). 
In addition, tidal spiralling should change. Simplest approach is to introduce an equation for dL/dt.

In the function _evolve, add:
 # Initial conditions for the motion of the cluster. Angular momentum is required only for eccentric orbits with tidal spiraling.
 R = [self.rcl0] # [kpc] Initial galactocentric distance.
 dRdt = [self.Vcl0] # [pc/Myrs] Initial velocity.
 theta = [self.theta0] # [rad] Initial angle.
 L = [self.L] # [pc^2 / Myrs] Initial angular momentum.
 
 y = [Mst[0], Mbh[0], rh[0], Mval[0], r[0], mst[0], R[0], dRdt[0], theta[0], L[0]] # Combine them in a multivariable.
 .
 .
 .
 .
 self.rcl = sol.y[6] # [kpc] Galactocentric distance.
 self.vcl_r = 1e3 * sol.y[7] / 1.023 # [km/s] Radial velocity of the cluster.
 self.theta_cl = sol.y[8] # [rad] Angle of orbit.
 self.L_cl = sol.y[9] # [pc^2 / Myrs] Angular momentum of the cluster.
 
 self.vcl_theta = self.L_cl / 1.023 / (self.rcl * 1e3) # [km/s] Angular velocity of the cluster.
 self.vcl = sqrt(self.vcl_r ** 2 + self.vcl_theta ** 2) # [km/s] Total velocity of the cluster.
 
 
Finally in _init_ add,

 self.e = 0 # Eccentricity of orbit. If none, the user inserts it otherwise the pericenter and apocenter distances are expected as input.
 self.Vcl0 = 0 # [km/s] Initial radial velocity of the cluster. Used for eccentric orbits.
 self.theta0 = pi # [rad] Initial angle on the cluster. Used for eccentric orbits.
 self.inwards = True # Statement which suggests that the cluster moves towards the pericenter. Important only for eccentric orbits.

before kwargs, while after add

 if self.eccentric and self.e > 0:
     self.Eorb = self.galactic_model_dict[self.galactic_model]['E'](0) # [pc^2/Myrs^2] Orbital energy of the cluster.
     self.L = self.galactic_model_dict[self.galactic_model]['L'](0) # [pc^2/Myrs] Orbital angular momentum.
    
    # Eccentric orbits can be studied by simply inserting an initial distance. The respective velocity is computed automatically. If not, the simulation starts at the apocenter.
    # The angle is not important because of the spherical symmetry.
     if not hasattr(self, 'rcl0'):
         self.rcl0 = self.rapo # [kpc] Start from the apocenter if it is not specified.
       
     else: # If initial position is specified, compute the velocity.
         if self.rperi > self.rcl0 or self.rapo < self.rcl0:
             raise ValueError(f"Invalid initial position {self.rcl0} kpc. For semi major axis {self.rg} kpc, the initial position is expected between the pericenter {self.rperi} kpc and the apocenter {self.rapo} kpc.")
         self.Vcl0 = 1e-3 * sqrt(abs(2 * (self.Eorb - self.phi_func(self.rcl0, self.Mg)) - self.L ** 2 / (self.rcl0 * 1e3) ** 2)) # [kpc / Myrs]. Initial radial velocity of the cluster. This is because it is dR / dt.
       
         # Check if initial angle is between pi and 2pi, and adjust the sign of the radial velocity.
         if self.inwards:
             self.Vcl0 *= -1 # Reverse velocity.

 else:
     
    self.rcl0 = self.rg # [kpc]  

In the dictionary for galactic models, add orbital energy and angular momentum.
Future extension for once the tidal radius and the evaporation timescale are specified for eccentric orbits (as approximations since no analytic expression for the tidal radius is known).
One approach is to use the instantaneous tidal radius rt = (GM/(L^2/RG^4-d^2Phi_G/dr^2))^(1/3), easy implementation in clusterBH, included in the galactic_model_dict as a function of r and M and called at each time step to estimate xi. This does not mean that the cluster has enough time to adjust to such changes, to the current tev is not valid.

"""

########################################################################################################################################################################################################################################################
