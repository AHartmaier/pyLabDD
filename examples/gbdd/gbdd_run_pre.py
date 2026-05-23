import pylabdd as dd
import numpy as np

# set simulation parameters
temp = 500.0
tau0 = 150.0
b0 = 0.25e-3
nu = 0.3
mu = 44.e3
drag = 500.
lgb = 0.1
gs = 10.0
delta = 0.004
ngbn = int(lgb/delta)
if ngbn % 2 == 0: ngbn += 1
dtmax = 6.25e-7/delta**4  # value 1.e4 for delta=0.005
tfin = 2.5e8
maxout = 1000
niter = 1_000_000

# Set pre-existing dislocations in pile-up
fill_fact = 0.5
hc = mu*b0/(2*np.pi*(1.-nu))   # Constant for dislocation stress field
Ntot = int(fill_fact*np.pi*(1-nu)*gs*tau0/(mu*b0)) # total number of dislocations
dmax = np.pi**2*gs / (64*fill_fact*Ntot**2)  # distance of first dislocation to GB = maximum travel distance per time step
ypos = np.zeros(Ntot)
ypos[1] = dmax
for i in range(2, Ntot):
    x0 = ypos[i-1]
    ypos[i] = x0 + np.pi*hc / (tau0*np.sqrt((gs*fill_fact-x0)/x0))

# Run gbdd simulation
gbdd = dd.GB_dislocations(temp=temp, tau0=tau0,
                   Ngbn=ngbn, len_gb_seg=lgb, grain_size=gs,
                   tfin=tfin, niter=niter, maxout=maxout,
                   dtmax=dtmax, drag=drag,
                   nu=nu, mu=mu, B=b0,
                   screenout=True)
gbdd.run_sim(pudis=ypos.copy())
gbdd.save_hdf5(f"gbpre_tau{int(tau0)}_T{int(temp)}_H{int(gs*10)}_L{int(lgb*10)}.h5")
gbdd.plot_time_series( ["n_slip_dis", "n_absorbed", "n_gbdis_eff"])
gbdd.plot_field('dGdb')
gbdd.plot_field("bfield")
gbdd.plot_time_series("plast_slip_rate")
