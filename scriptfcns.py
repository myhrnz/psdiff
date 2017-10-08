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

def create_ys(params,xdata,t_settings): 
    import copy
    curr_params = copy.copy(params)
    xval,xout = xdata.values()
    
    ys = []
    for x in xval:
        curr_params.change(xdata.name(),x)
        analyses = Analyses(curr_params,t.num_trials)
        p_values = np.array(analyses.p_values())
        ys.append(stat_repr(p_values,t_settings))
    return ys

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

def threshold_crossed(params,t_settings):
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

class Data(object):
  def __init__(self, left, right, step):
    self.left = left
    self.right = right
    self.step = step
    if left > right:
      self.reversed = True
    else:
      self.reversed = False
    
  def bare_values(self):
    if self.reversed:
      values = np.arange(self.right,self.left+self.step/2.,self.step)
      values = values[::-1]
      outside = self.right - self.step
    else:
      values = np.arange(self.left,self.right+self.step/2.,self.step)
      outside = self.right + self.step
    return (values,outside) # outside can be used in contour plot generation
  
  def pow_values(self):
    values = self.bare_values()
    return (pow(10.,values[0]),pow(10.,values[1]))
  
  def log_values(self):
    values = self.bare_values()
    return (np.log10(values[0]),np.log10(values[1]))
  
  def inv_values(self):
    values = self.bare_values()
    return ((1./values[0]),(1./values[1]))

  def reverse(self):
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

def z_req(params,
          xdata,
          ydata,
          zdata,
          t_settings,
          verbose = False):
    
    xval,xout = xdata.values()
    yval,yout = ydata.values()
    zval,zout = zdata.values()
    
    z_req = np.zeros(shape=(len(yval),len(xval)))
    
    for y in np.arange(len(yval)):
        params.change(ydata.name(),yval[y])
        if verbose:
            print(ydata.name() + ": " + str(yval[y]))
        
        for x in np.arange(len(xval)):
            params.change(xdata.name(),xval[x])
            if verbose:
                print("\t" + xdata.name() + ": " + str(xval[x]))
            
            for z in np.arange(len(zval)):
                params.change(zdata.name(),zval[z])
                if verbose:
                    print("\t\t" + zdata.name() + ": " + str(zval[z]))
                z_req[y][x] = zval[z]
                if threshold_crossed(params,t_settings):
                    if verbose:
                        print("\t\t\t" + "Threshold crossed at " + zdata.name() + ": " + str(zval[z]) )
                    break
            else:
                if verbose:
                    print("\t\t\tThrehold not crossed!")
                z_req[y][x] = zout
    if zdata.log:
        return np.log10(z_req)
    else:
        return z_req
    if zdata.inv:
      return 1./z_req
    else: 
      return z_req