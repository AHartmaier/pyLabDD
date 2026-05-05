import time
import numpy as np
import matplotlib.pyplot as plt
from pylabdd import GB_dislocations
import pylabdd as dd

temp = 900.0
tau0 = 150.0
lgb = 0.1
grain_size = 1.0
delta = 0.005
maxout = 1000
tfin = 1.e9
niter = 500_000_000
ngbn = 41 #int(lgb/delta)
if ngbn % 2 == 0: ngbn += 1
dtmax = 5.e1 # 6.25e-6/delta**4  # value 1.e4 for delta=0.005
gbdd = GB_dislocations(temp=temp, tau0=tau0,
                       Ngbn=ngbn, len_gb_seg=lgb, grain_size=grain_size,
                       tfin=tfin, niter=niter, maxout=maxout,
                       dtmax=dtmax,
                       screenout=True)
t0 = time.time()
gbdd.run_sim()
t1 = time.time()
print(f"Simulation finished after {t1 - t0} s.")
gbdd.plot_time_series(['timestep', 'n_gbdis_eff'])
gbdd.plot_time_series("plastic_slip")
gbdd.plot_time_series("plast_slip_rate")
gbdd.plot_field('displacement')
gbdd.plot_field('bfield')  # , file="T900_Lgb10_short.pdf")
gbdd.plot_pile_up()
gbdd.save_hdf5(f"pu1_tau{int(tau0)}_T{int(temp)}.h5")


it = 20
sp_ang = 0.5*np.pi
C = gbdd.mu*gbdd.B/(2*np.pi*(1.-gbdd.nu))   # Constant for dislocation stress field
hh = gbdd.pu_dis['position'][it, :]
ind = np.nonzero(hh)[0]
yp = hh[ind]
gbf = gbdd.pu_dis['force'][it, ind]
ndis = len(yp)
xp = np.ones(ndis)*lgb*0.5

#define different dislocations in box
# initialze object of class Dislocations
dsl = dd.Dislocations(ndis, ndis, sp_ang, C, gbdd.B,
                      xpos=xp, ypos=yp, LX=lgb, LY=grain_size,
                      bc="fixed")
dsl.plot_stress(show_arrows=False)
fpk = dsl.calc_force(tau0=tau0)
print(yp)
print(fpk[1, :])
print(gbf)

"""
# add GB dislocation after first absorption
it = 140
sp_ang = 0.5*np.pi
C = gbdd.mu*gbdd.B/(2*np.pi*(1.-gbdd.nu))   # Constant for dislocation stress field
hh = gbdd.pu_dis['position'][it, :]
ind = np.nonzero(hh)[0]
ndis = len(ind) + 1
yp = np.zeros(ndis)
yp[1:] = hh[ind]
xp = np.ones(ndis)*lgb*0.5
gbf = gbdd.pu_dis['force'][it, ind]
#define different dislocations in box
# initialze object of class Dislocations
dsl = dd.Dislocations(ndis, ndis, sp_ang, C, gbdd.B,
                      xpos=xp, ypos=yp, LX=lgb, LY=grain_size,
                      bc="fixed")
dsl.plot_stress(show_arrows=False)
fpk = dsl.calc_force(tau0=tau0)
print(fpk[1, 1:])
print(gbf)"""