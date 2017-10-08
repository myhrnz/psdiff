import numpy as np
import pylab as pl
from scipy.stats import poisson
from copy import copy

# trying something new here...

DEGTORAD = np.pi/180.
LETTERS = ('e','d','r','b','a','s','f')
NAMES = ('evolution_param','local_src_density','bin_radius_deg','sig_to_bkg','diff_events','catalog_srcs_in_fov','field_of_view')
SYMBOLS = (r"$\xi$",r"$\rho_0$",r"$\Delta \theta$",r"$N/B_{\mathrm{atm}}$",r"$N$",r"$k$",r"$f_{sky}$")
LABELS = (r"$\xi$",r"$\rho_0 $"+r"[Mpc"+r"$^{-3}$" + r"]",r"$\Delta \theta$"+r"[deg]",r"$N/B_{\mathrm{atm}}$",r"$N$",r"$k$",r"$f_{sky}$")
LOGLABELS = (r"$log_{10} \xi$",r"$\log_{10} ($" + r"$\rho_0 /$"+r"Mpc"+r"$^{-3}$" + ")",r"$\log_{10} (\Delta \theta/$"+"deg)",r"$\log_{10} (N/B_{\mathrm{atm}})$",r"$\log_{10} N$",r"$\log_{10} k$","MISSING")
INVLABELS = (r"$\xi^{-1}$","MISSING","MISSING","MISSING","MISSING","MISSING","MISSING")

class Universe:
  # constants
  HUBBLECONSTANT = 7e4
  SPEEDOFLIGHT = 3e8
  HUBBLE_DISTANCE = SPEEDOFLIGHT/HUBBLECONSTANT
  
  def __init__(self, 
               local_src_density = 1e-6, 
               evolution_param = 2.4,
               description = ""):
    self.local_src_density = local_src_density
    self.evolution_param = evolution_param
    self.description = description
    
    self.one_src_volume = 1./self.local_src_density
  
  def __str__(self):
    rep =   "\t-- Universe parameters --" + \
      "\n\tlocal_src_density:\t" + "%.1e" %self.local_src_density + \
        "\n\tevolution_param:\t" + str(self.evolution_param) + "\n"
    return rep

class Detector:
  def __init__(self, 
               field_of_view = 1., 
               catalog_srcs_in_fov = 1, 
               diff_events = 100, 
               sig_to_bkg = 1., 
               bin_radius_deg = 0.3, 
               sig_eff = 1.0,
               description = ""):
    self.field_of_view = field_of_view
    self.catalog_srcs_in_fov = catalog_srcs_in_fov
    self.diff_events = diff_events
    self.sig_to_bkg = sig_to_bkg
    self.bin_radius_deg = bin_radius_deg
    self.sig_eff = sig_eff
    self.description = description
    
    self.bkg_events = self.diff_events / self.sig_to_bkg
    self.nonsig_events = self.bkg_events + self.diff_events
    self.bin_radius_rad = self.bin_radius_deg * DEGTORAD
    self.bin_area_s = np.pi * self.bin_radius_rad**2.
    self.fov_frac_of_bin = self.bin_area_s / (self.field_of_view * 4. * np.pi)
    self.nonsig_in_bin = self.fov_frac_of_bin * self.nonsig_events
    
  def __str__(self):
    rep =   "\t-- Detector parameters --" + \
      "\n\tfield_of_view:\t\t" + str(self.field_of_view) + \
        "\n\tcatalog_srcs_in_fov:\t" + str(self.catalog_srcs_in_fov) + \
          "\n\tdiff_events:\t\t" + str(self.diff_events) + \
            "\n\tsig_to_bkg:\t\t" + str(self.sig_to_bkg) + \
              "\n\tbin_radius_deg:\t\t" + str(self.bin_radius_deg) + \
                "\n\tsig_eff:\t\t" + str(self.sig_eff) +  "\n"
    return rep

class Src_Realization:
  """ A Src_Realization takes a Parameters as parameter 
      and creates a list of sources with radii consistent with these parameters. """
  def __init__(self, params):
    self.true_shell_content = []
    self.expected_shell_content = []
    self.universe = params.universe
    self.detector = params.detector
    self.make_shell_contents(self.detector.catalog_srcs_in_fov)
    self.reverse_shell_constents()
    self.shell_calc = Shell_Frac_Calc(self.detector.field_of_view)
    self.make_partition()
    self.generate_source_radii()
    
  def make_shell_contents(self, num_srcs_to_make):
    expected_num_scrs_in_current = num_srcs_to_make
    true_num_srcs_in_current = np.random.poisson(expected_num_scrs_in_current)
    if true_num_srcs_in_current < num_srcs_to_make:
      self.make_shell_contents(num_srcs_to_make - true_num_srcs_in_current)
    self.true_shell_content.append(true_num_srcs_in_current)
    self.expected_shell_content.append(expected_num_scrs_in_current)
  
  def reverse_shell_constents(self):
    self.true_shell_content.reverse()
    self.expected_shell_content.reverse()
    
  def make_partition(self):
    self.partition = []
    min_rad = 0
    max_rad = 0
    for content in self.expected_shell_content:
      self.partition.append(max_rad)
      shell_volume = self.universe.one_src_volume*content
      max_rad = self.shell_calc.get_out_rad(min_rad,shell_volume)
      min_rad = max_rad
    self.partition.append(max_rad)
    
  def generate_source_radii(self):
    self.radii = []
    num_of_shells = len(self.true_shell_content)
    for i in np.arange(num_of_shells):
      for source in np.arange(self.true_shell_content[i]):
        radius = random_radius(self.partition[i],self.partition[i+1])
        self.radii.append(radius)
    self.radii.sort()
    self.radii = self.radii[:self.detector.catalog_srcs_in_fov]
    
class Bin_Realization:
  """ A Bin_Realization takes a Src_Realization as arguments and simulates the neutrino signal in 
      bins centered at the sources at distances given by Src_Realization.radii """
  def __init__(self, src_realization):
    self.src_realization = src_realization
    self.universe = self.src_realization.universe
    self.detector = self.src_realization.detector
    self.generate_mean_nonsig()
    self.generate_mean_sig()
    self.sim_nonsig()
    self.sim_sig()
    self.events = (np.array(self.nonsig) + np.array(self.sig)).tolist() # for consistency only
    
  def generate_mean_nonsig(self):
    # can be modified to take into account varying background
    self.mean_nonsig = [self.detector.nonsig_in_bin for i in range(0,self.detector.catalog_srcs_in_fov)]
  
  def generate_mean_sig(self):
    mean_sig = []
    for radius in self.src_realization.radii:
      curr_mean_sig = \
        (self.universe.HUBBLECONSTANT * self.detector.diff_events) / \
          (self.detector.field_of_view * 4. * np.pi * self.universe.evolution_param * self.universe.SPEEDOFLIGHT * self.universe.local_src_density * radius**2)
      mean_sig.append(curr_mean_sig)
    self.mean_sig = mean_sig
    
  def sim_nonsig(self):
    nonsig = []
    for mean in self.mean_nonsig:
      nonsig.append(np.random.poisson(mean))
    self.nonsig = nonsig
  
  def sim_sig(self):
    sig = []
    for mean in self.mean_sig:
      true = np.random.poisson(mean)
      true_in_bin = 0
      for i in np.arange(0,true):
        if np.random.uniform() < self.detector.sig_eff:
          true_in_bin += 1
      sig.append(true_in_bin)
    self.sig = sig

class Analysis:
  def __init__(self, bin_realization):
    self.bin_realization = bin_realization
    self.src_realization = bin_realization.src_realization
    self.universe = self.src_realization.universe
    self.detector = self.src_realization.detector
    self.binned_stacked_search()

  def binned_stacked_search(self):
    p_value = 1
    events_in_bins = sum(self.bin_realization.events)
    mean_nonsig_in_bins = sum(self.bin_realization.mean_nonsig)
    pdf = poisson(mean_nonsig_in_bins)
    if events_in_bins >=1:
      p_value = 1 - pdf.cdf(events_in_bins - 1)
    self.p_value = p_value

# small help classes:
class Shell_Frac_Calc:
  """ A Shell_Frac_Calc takes a fraction of a spherical shell as a parameter
  to be able to calculate other properties of such a fraction of a shell. """
  def __init__(self, fraction = 1):
    self.fraction = fraction
  def get_out_rad(self, in_rad, volume):
    out_rad = (  3. * volume/(4. * np.pi * self.fraction) + in_rad**3.  )**(1./3.)
    if out_rad > Universe.HUBBLE_DISTANCE:
      print("Warning.  Euclidean approximation no longer valid since out_rad = " + str(out_rad))
    return out_rad

class Src_Realizations:
  def __init__(self, params, num_trials):
    self.params = params
    self.srcs = []
    for trial in range(num_trials):
      srcs = Src_Realization(self.params)
      self.srcs.append(srcs)

class Bin_Realizations:
  def __init__(self, params, num_trials):
    self.params = params
    self.srcs = []
    self.bins = []
    for trial in range(num_trials):
      srcs = Src_Realization(self.params)
      bins = Bin_Realization(srcs)
      self.srcs.append(srcs)
      self.bins.append(bins)
  def signal_in_bins(self):
    signal = []
    for i in range(len(self.bins)):
      bins = self.bins[i]
      signal.append(sum(bins.sig))
    return signal
  
  def nonsig_in_bins(self):
    nonsignal = []
    for i in range(len(self.bins)):
      bins = self.bins[i]
      nonsignal.append(sum(bins.nonsig))
    return nonsignal
  
  def events_in_bins(self):
    events = []
    for i in range(len(self.bins)):
      bins = self.bins[i]
      events.append(sum(bins.events))
    return events

  def mean_nonsig_in_bins(self):
    mean_nonsig = []
    for i in range(len(self.bins)):
      bins = self.bins[i]
      mean_nonsig.append(sum(bins.mean_nonsig))
    return mean_nonsig

class Analyses:
  def __init__(self, params, num_trials):
    self.params = params
    self.srcs = []
    self.bins = []
    self.analyses = []
    for trial in range(num_trials):
      srcs = Src_Realization(self.params)
      bins = Bin_Realization(srcs)
      analysis = Analysis(bins)
      self.srcs.append(srcs)
      self.bins.append(bins)
      self.analyses.append(analysis)
  def p_values(self):
    p_values = []
    for i in range(len(self.analyses)):
      analysis = self.analyses[i]
      p_values.append(analysis.p_value)
    return p_values

class Parameters:
  """ Class for Detector and Universe to behave like a single unit. 
  Note that once a Parameters is created, the universe and the detector that went into it 
  has absolutely nothing to do with the universe and the detector that belong to the Parameters instance."""
  def __init__(self,universe,detector):
    self.universe = universe
    self.detector = detector
  
  def __str__(self):
    rep_u = self.universe.__str__()
    rep_d = self.detector.__str__()
    return rep_u + rep_d
  
  def change(self,param,value):
    if param == 'local_src_density':
      new_universe = Universe(local_src_density=value,
                              evolution_param=self.universe.evolution_param)
      self.universe = new_universe
    elif param == 'evolution_param':
      new_universe = Universe(local_src_density=self.universe.local_src_density,
                              evolution_param=value)
      self.universe = new_universe
    elif param == 'field_of_view':
      new_detector = Detector(field_of_view=value,
                              catalog_srcs_in_fov=self.detector.catalog_srcs_in_fov,
                              diff_events=self.detector.diff_events,
                              sig_to_bkg=self.detector.sig_to_bkg,
                              bin_radius_deg=self.detector.bin_radius_deg,
                              sig_eff=self.detector.sig_eff)
      self.detector = new_detector
    elif param == 'catalog_srcs_in_fov':
      new_detector = Detector(field_of_view=self.detector.field_of_view,
                              catalog_srcs_in_fov=value,
                              diff_events=self.detector.diff_events,
                              sig_to_bkg=self.detector.sig_to_bkg,
                              bin_radius_deg=self.detector.bin_radius_deg,
                              sig_eff=self.detector.sig_eff)
      self.detector = new_detector
    elif param == 'diff_events':
      new_detector = Detector(field_of_view=self.detector.field_of_view,
                              catalog_srcs_in_fov=self.detector.catalog_srcs_in_fov,
                              diff_events=value,
                              sig_to_bkg=self.detector.sig_to_bkg,
                              bin_radius_deg=self.detector.bin_radius_deg,
                              sig_eff=self.detector.sig_eff)
      self.detector = new_detector
    elif param == 'sig_to_bkg':
      new_detector = Detector(field_of_view=self.detector.field_of_view,
                              catalog_srcs_in_fov=self.detector.catalog_srcs_in_fov,
                              diff_events=self.detector.diff_events,
                              sig_to_bkg=value,
                              bin_radius_deg=self.detector.bin_radius_deg,
                              sig_eff=self.detector.sig_eff)
      self.detector = new_detector
    elif param == 'bin_radius_deg':
      new_detector = Detector(field_of_view=self.detector.field_of_view,
                              catalog_srcs_in_fov=self.detector.catalog_srcs_in_fov,
                              diff_events=self.detector.diff_events,
                              sig_to_bkg=self.detector.sig_to_bkg,
                              bin_radius_deg=value,
                              sig_eff=self.detector.sig_eff)
      self.detector = new_detector
    elif param == 'sig_eff':
      new_detector = Detector(field_of_view=self.detector.field_of_view,
                              catalog_srcs_in_fov=self.detector.catalog_srcs_in_fov,
                              diff_events=self.detector.diff_events,
                              sig_to_bkg=self.detector.sig_to_bkg,
                              bin_radius_deg=self.detector.bin_radius_deg,
                              sig_eff=value)
      self.detector = new_detector
    else:
      print("Unvalid parameter name")
  def get_universe(self):
    return self.universe
  def get_detector(self):
    return self.detector
  def get_value(self,name):
    if name == 'local_src_density':
      value = self.universe.local_src_density
    elif name == 'evolution_param':
      value = self.universe.evolution_param
    elif name == 'field_of_view':
      value = self.detector.field_of_view
    elif name == 'catalog_srcs_in_fov':
      value = self.detector.catalog_srcs_in_fov
    elif name == 'diff_events':
      value = self.detector.diff_events
    elif name == 'sig_to_bkg':
      value = self.detector.sig_to_bkg
    elif name == 'bin_radius_deg':
      value = self.detector.bin_radius_deg
    elif name == 'sig_eff':
      value = self.detector.sig_eff
    else:
      print("Unvalid parameter name")
    return value

# small help functions:
def random_radius(min_rad, max_rad):
  """ random_radius takes a minimum radius and a maximum radius
  and returns a random radius between these according to a normalized r^2 distribution."""
  normalization = max_rad**3. - min_rad**3.
  rand_uni = np.random.uniform()
  rand_rad = (normalization * rand_uni + min_rad**3.)**(1./3.)
  return rand_rad

# some theoretical calculations from parameters: 
def theo_dist_to_source(parameters,source):
  volume = source*parameters.universe.one_src_volume
  radius = ((3. * volume)/(4. * np.pi * parameters.detector.field_of_view))**(1./3.)
  return radius

def theo_sig(params,radius):
  return (params.detector.sig_eff * params.universe.HUBBLECONSTANT * params.detector.diff_events * radius)/(params.universe.SPEEDOFLIGHT * params.universe.evolution_param)

def theo_nonsig(params,radius):
  return params.detector.nonsig_in_bin * params.universe.local_src_density * params.detector.field_of_view*((4. * np.pi * radius**3)/3.)
