import time
import numpy as np
import matplotlib.pyplot as plt
from pylabdd import GB_dislocations

temp = 900.
lgb = 1.0
delta = 0.005
maxout = 1000
tfin = 0.8e9
niter = 5_000_000
ngbn = int(lgb/delta)
if ngbn % 2 == 0: ngbn += 1
dtmax = 6.25e-6/delta**4  # value 1.e4 for delta=0.005
gbdd = GB_dislocations(temp=temp, Ngbn=ngbn, len_gb_seg=lgb, tfin=tfin,
                       niter=niter, maxout=maxout, dtmax=dtmax, screenout=True)
t0 = time.time()
gbdd.run_sim()
t1 = time.time()
print(f"Simulation finisched after {t1 - t0} s.")
gbdd.plot_time_series(['timestep', 'n_gbdis_eff'])
gbdd.plot_time_series('max_bdot')
gbdd.plot_field('displacement')
gbdd.plot_field('bfield', file="T900_Lgb10_short.pdf")


