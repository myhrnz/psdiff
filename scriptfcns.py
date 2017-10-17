# 1) Only fractions

def option(letter,options):
  return options[LETTERS.index(letter)]

def name(letter):
  name = option(letter,NAMES)
  return name

def symbol(letter):
    symbol = option(letter,SYMBOLS)
    return symbol

def label(letter,log,inv):
  if log:
    label = option(letter,LOGLABELS)
  elif inv:
    label = option(letter,INVLABELS)
  else:
    label = option(letter,LABELS)
  return label

class Trial_Settings:
  def __init__(self,
               num_trials = 100,
               p_threshold = 1e-3,
               fraction = 0.5,
               choice = "fraction"):
    self.num_trials = num_trials
    self.p_threshold = p_threshold
    self.fraction = fraction
    self.choice = choice
    
  def __str__(self):
    rep =   "\t-- Trial settings --" + \
            "\n\tnum_trials:\t" + str(self.num_trials) + \
              "\n\tp_threshold:\t" + "%.0e" %self.p_threshold + \
                "\n\tfraction:\t" + str(self.fraction) + \
                  "\n\tchoice:" + self.choice + "\n"
    return rep

class Data(object):
  """ A Data object stores a left endpoint, right endpoint and a stepsize. 
  If the left endpoint is greater than its right endpoint, reverse is automatically set to true when the object is created. """
  def __init__(self, left, right, step):
    self.left = left
    self.right = right
    self.step = step
    if left > right:
      self.reversed = True
    else:
      self.reversed = False
    
  def bare_values(self):
    """ bare_values calculates the np.array with values that correspond directly to left/right endpoint and stepsize. 
    It returns this array together with what would be the next value in the list, as seen from the right endpoint. 
    For example, if the Data object is reversed and hence right endpoint < left endpoint, then outside is right endpoint - stepsize. """
    if self.reversed:
      values = np.arange(self.right,self.left+self.step/2.,self.step)
      values = values[::-1]
      outside = self.right - self.step
    else:
      values = np.arange(self.left,self.right+self.step/2.,self.step)
      outside = self.right + self.step
    return (values,outside) # outside can be used in contour plot generation
  
  def pow_values(self):
    """ pow_values returns the power (base 10) of the np.array and the outside value returned by bare_values(). """
    values = self.bare_values()
    return (pow(10.,values[0]),pow(10.,values[1]))
  
  def log_values(self):
    """ pow_values returns the logarithm (base 10) of the np.array and the outside value returned by bare_values(). """
    values = self.bare_values()
    return (np.log10(values[0]),np.log10(values[1]))
  
  def inv_values(self):
    """ pow_values returns the inverse of the np.array and the outside value returned by bare_values(). """
    values = self.bare_values()
    return ((1./values[0]),(1./values[1]))

  def reverse(self):
    """ Put left endpoint as right endpoint and right endpoint as left endpoint and set reversed to True. """
    self.left = self.right
    self.right = self.left
    self.reversed ^= True

class Plot_Data(Data):
  def __init__(self,letter,log,inv,*args,**kwargs):
    super(Plot_Data, self).__init__(*args,**kwargs)
    self.letter = letter
    self.log = log
    self.inv = inv
    
  def label(self):
    return label(self.letter,self.log,self.inv)
  
  def name(self):
    return name(self.letter)
  
  def symbol(self):
    return symbol(self.letter)
    
  def values(self):
    # right now can only choose logged OR inverted, not both; fix with bitwise?
    if self.log: 
      return super(Plot_Data, self).pow_values()
    elif self.inv:
      return super(Plot_Data, self).inv_values()
    else:
      return super(Plot_Data, self).bare_values()
    
  def plot_values(self):
    values,outside= super(Plot_Data, self).bare_values()
    return values
  
###################################################################################
# everthing below needs to be adjusted

# make a new version of z_req that reads loaded data instead of generating new data
def z_req(params,
          xdata,
          ydata,
          zdata,
          t_settings):
    
    xval,xout = xdata.values()
    yval,yout = ydata.values()
    zval,zout = zdata.values()
    
    z_req = np.zeros(shape=(len(yval),len(xval)))
    
    for y in np.arange(len(yval)):
        params.change(ydata.name(),yval[y])
        
        for x in np.arange(len(xval)):
            params.change(xdata.name(),xval[x])
            
            for z in np.arange(len(zval)):
                params.change(zdata.name(),zval[z])
                z_req[y][x] = zval[z]
                if threshold_crossed(t_settings=t_settings,params=params):
                  break
            else:
                z_req[y][x] = zout
    if zdata.log:
        return np.log10(z_req)
    else:
        return z_req
    if zdata.inv:
      return 1./z_req
    else: 
      return z_req

def threshold_crossed(t_settings,params):
    if t_settings.choice == "fraction":
        analyses = Analyses(params, t_settings.num_trials)
        values = np.array(analyses.p_values())
        test_var = stat_repr(values,t_settings)
        if test_var > t_settings.fraction:
          return True

def stat_repr(values,t_settings):
    if t_settings.choice == 'fraction':
        msk = (values < t_settings.p_threshold)
        return float(len(values[msk]))/float(len(values))

##############################################################################
# experimental function binned_stacked_search() and analyze_bin_realizations()
def binned_stacked_search(bin_realization, num_bins):
  """ Performs a binned_stacked_search on the num_bins closest sources in the bin_realization """
  if (bin_realization.detector.catalog_srcs_in_fov >= num_bins):
    p_value = 1
    # the summing over bins is done here
    events_in_bins = sum(bin_realization.events[:num_bins])
    mean_nonsig_in_bins = sum(bin_realization.mean_nonsig[:num_bins])
    pdf = poisson(mean_nonsig_in_bins)
    if events_in_bins >=1:
      p_value = 1 - pdf.cdf(events_in_bins - 1)
    return p_value
  else:
    print("Not enough sources in chosen bin_realization")
    
def analyze_bin_realizations(bin_realizations, num_bins):
  p_values = []
  for br in bin_realizations.bins:
    p_value = binned_stacked_search(br,num_bins)
    p_values.append(p_value)
  return p_values
###############################################################################

# this function is not used yet...
#def create_ys(params,xdata,t_settings): 
    #import copy
    #curr_params = copy.copy(params)
    #xval,xout = xdata.values()
    
    #ys = []
    #for x in xval:
        #curr_params.change(xdata.name(),x)
        #analyses = Analyses(curr_params,t.num_trials)
        #p_values = np.array(analyses.p_values())
        #ys.append(stat_repr(p_values,t_settings))
    #return ys