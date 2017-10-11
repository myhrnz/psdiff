# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

%matplotlib inline

# <codecell>

execfile("../psdiff.py")
execfile("../scriptfcns.py")

# <codecell>

t = Trial_Settings(num_trials=10,p_threshold=0.001,fraction=0.5,choice='fraction')

u = Universe(local_src_density=1e-5,evolution_param=3.6)
d = Detector(field_of_view=0.5, catalog_srcs_in_fov=1, diff_events=500, sig_to_bkg=0.003, bin_radius_deg=0.5)
p = Parameters(u,d)

x = Plot_Data('e',True,False, *(np.log10(18.5), np.log10(0.5), 0.1))
y = Plot_Data('d',True, False, *(-8.1,-3.9,0.1))
z = Plot_Data('a',True, False, *(0.,4.,0.1))

#####################################################################
################## specific physics scenarios #######################
#u = Universe(local_src_density=1e-5,evolution_param=3.6) # AGN
#u = Universe(local_src_density=1e-4,evolution_param=18.) # STARBURST

################## specific detector scenarios ######################
#d = Detector(field_of_view=0.5, catalog_srcs_in_fov=100, diff_events=500, sig_to_bkg=0.003, bin_radius_deg=0.5) # PSSAMPLE
#####################################################################

#####################################################################
################## physics oriented parameters ######################
#x = Plot_Data('e',False,True, *(1/0.5, 1/18.5, 0.1))
#x = Plot_Data('e',True,False, *(np.log10(18.5), np.log10(0.5), 0.1))
#x = Plot_Data('e',False,False, *(18.5, 0.5, 1.0))

#y = Plot_Data('d',True, False, *(-8.,-4.,0.1))

################# detector oriented parameters #####################
#x = Plot_Data('a',True, False, *(1.,4.,0.1))
#x = Plot_Data('b',True, False, *(-3.,0.,0.1))
#x = Plot_Data('r',True, False, *(0.,-4.,0.1))

#y = Plot_Data('a',True, False, *(1.,4.,0.1))
#y = Plot_Data('b',True, False, *(-3.,0.,0.1))
#y = Plot_Data('r',True, False, *(0.,-4.,0.1))

#z = Plot_Data('a',True, False, *(0.,4.,0.1))
#z = Plot_Data('b',True, False, *(-3.,0.,0.1))
#z = Plot_Data('r',True, False, *(0.,-4.,0.1))
#####################################################################

# <codecell>

z_res = z_req(p,x,y,z,t)

# <codecell>

# title
title = z.label() + " required to get p < " + str(t.p_threshold) + " in " + str(100*t.fraction) \
                    + "% of (" + str(t.num_trials) + ")trials\n"
varying = [x.letter,y.letter,z.letter]
constant = [a for a in LETTERS if a not in varying]
conststr = ""
for c in constant:
    conststr = conststr + symbol(c) + " = " + str(p.get_value(name(c))) + ", "
title = title + conststr

# levels
MIN = min(z.plot_values())
MAX = max(z.plot_values())
STEP = z.step
levels = np.arange(MIN,MAX+STEP/2.,STEP)

# plot 
plt.rcParams.update({'font.size': 14})
plt.figure(1,figsize=(12,8))
plt.clf()
plt.contourf(x.plot_values(),
             y.plot_values(),
             z_res, #note: might need to round z_res to prec better than z.step but worse than prec of maschine
             levels,cmap=pl.cm.jet)

plt.title(title)
plt.xlabel(x.label())
plt.ylabel(y.label())

# ticks
main_ticks = [0.0, 1.0, 2.0, 3.0] # main_ticks to the left of the colorbar
livetimes = [1, 5, 10, 20]
diff_in_1yr = 25
time_ticks = [np.log10(diff_in_1yr*a) for a in livetimes]
cb = pl.colorbar(ticks = main_ticks + time_ticks)
tick_labels = [str(a) for a in main_ticks] + [str(a)+" years" for a in livetimes]
cb.ax.set_yticklabels(tick_labels)

# points (make these individually instead, no points only boxes
blazar = ['BLZR', np.log10(3.6), np.log10(1e-8)]
agn = ['AGN', np.log10(3.6), np.log10(1e-5)]
burst = ['BURST', np.log10(18.0), np.log10(1e-4)]
sources = [blazar, agn, burst]

sx = [a[1] for a in sources]
sy = [a[2] for a in sources]

plt.plot(sx,sy,'ob')

for s in sources:
    ts = plt.text(s[1]+0.01, s[2]+0.05,s[0])
    ts.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='blue'))

# sfr line (make no_evolution as well and make individually
plt.plot((np.log10(2.4),np.log10(2.4)),(min(y.plot_values()),max(y.plot_values())), 'b')

plt.show()

# <codecell>

# SANDBOX

import pickle

dict1= {'Name': 'Zara', 'Age': 7}
def fcn1(Name = 'Emelie', Age = 27):
    print(Name,Age)

pickle_out = open( "out.p", 'w')
pickle.dump( dict1,  pickle_out)
pickle_out.close()

pickle_in = open( "out.p", 'r')
dict2 = pickle.load( pickle_in )
pickle_in.close()

fcn1(**dict2)
fcn1()

# <codecell>


