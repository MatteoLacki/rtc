%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
from plotnine import *

from rta.plot.runs import plot_distances_to_reference
from rta.reference import cond_medians
from rta.filters.angry import is_angry

# from rta.dev_get_rta import data_for_clustering
# UA = pd.read_msgpack('/Users/matteo/Projects/rta/data/unlabelled_all.msg')
# AA = pd.read_msgpack('/Users/matteo/Projects/rta/data/annotated_all.msg')
# D, U = data_for_clustering(AA, 5, UA)
# D.to_msgpack('/Users/matteo/Projects/rta/data/D.msg')
# U.to_msgpack('/Users/matteo/Projects/rta/data/U.msg')
D = pd.read_msgpack('/Users/matteo/Projects/rta/data/D.msg')
U = pd.read_msgpack('/Users/matteo/Projects/rta/data/U.msg')
# denoising D, as U cannot be denoised
# this will put some entries from D to U

# rt VS rta_RollingMedian
# plot_distances_to_reference(D.rt,  D.rt_me, D.run, s=1)
# plot_distances_to_reference(D.rta, D.rt_me, D.run, s=1)

ids = D.id.values
run = D.run.values
q = D.charge.values

mass = D.mass.values
mass_me = cond_medians(mass, ids)
# plot_distances_to_reference(mass,  mass_me, run, s=1)
mass_angry = is_angry(mass_me-mass)
plt.scatter(mass, mass_me-mass, s=1, c=mass_angry)
plt.show()

# another dim: the same criterion?
rt = D.rt.values
rta = D.rta.values
rt_me = D.rt_me.values
# plot_distances_to_reference(rta, rt_me, run, s=1)
rta_angry = is_angry(rt_me - rta)
plt.scatter(rta, rt_me - rta, s=1, c=rta_angry)
plt.show()
# these are all executed not on groups! which is OK

# NOW: DRIFT TIMES  
dt = D.dt.values
dt_me = cond_medians(dt, ids)
plot_distances_to_reference(dt,  dt_me, run, s=1)

not_angry = ~np.logical_or(mass_angry, rta_angry)
dt_filtered = dt[not_angry]
ids_filtered = ids[not_angry]
dt_filtered_me, _ = cond_medians(dt_filtered, ids_filtered)

plt.scatter(dt_filtered, dt_filtered_me - dt_filtered, s=1)
plt.show()



