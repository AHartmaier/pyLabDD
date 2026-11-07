import time
import numpy as np
import matplotlib.pyplot as plt
from pylabdd import GB_dislocations
import pylabdd as dd

temp = 900.0
tau0 = 150.0
lgb = 0.1
grain_size = 1.0
maxout = 1000
tfin = 1.5e9
niter = 500_000_000
#ngbn = 41 #int(lgb/delta)
#if ngbn % 2 == 0: ngbn += 1
for ngbn in [21, 31, 41, 51]:
    delta = lgb / (ngbn-1)
    dtmax = 6.25e-6/delta**4  # value 1.e4 for delta=0.005
    gbdd = GB_dislocations(temp=temp, tau0=tau0,
                       Ngbn=ngbn, len_gb_seg=lgb, grain_size=grain_size,
                       tfin=tfin, niter=niter, maxout=maxout,
                       dtmax=dtmax,
                       screenout=True)
    gbdd.run_sim()
    gbdd.save_hdf5(f"pu1_Ngbn{ngbn}_tau{int(tau0)}_T{int(temp)}.h5")
