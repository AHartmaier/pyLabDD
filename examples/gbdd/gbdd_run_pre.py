import pylabdd as dd
import numpy as np

# set simulation parameters
temp = 900.0
tau0 = 150.0
b0 = 0.25e-3
nu = 0.3
mu = 44.e3
drag = 500.
C = mu*b0/(2*np.pi*(1.-nu))   # Constant for dislocation stress field
f0 = 0.1             # initial slip resistance
lgb = 0.1
gs = 1.0
delta = 0.004
ngbn = int(lgb/delta)
if ngbn % 2 == 0: ngbn += 1
dtmax = 6.25e-7/delta**4  # value 1.e4 for delta=0.005
tfin = 2.5e9
maxout = 1000
niter = 50_000_000

# Set pre-existing dislocations in pile-up an relax under applied stress
# define numerical parameters for pile-up relaxation
dt0 = 0.02           # sim_time step
bc = 'fixed'            # set boundary conditions: 'fixed' or 'pbc'

# define dislocations
Ntot = int(np.pi*(1-nu)*gs*tau0/(mu*b0)) # total number of dislocations according to Hall-Petch model
dsl = dd.Dislocations(Ntot, Ntot, 0.5*np.pi, C, b0, LX=lgb, LY=gs, bc=bc, f0=f0) 
for i in range(Ntot):
    dsl.xpos[i] = lgb*0.5
    dsl.ypos[i] = gs * (i+0.2) / Ntot

# relax dislocation configuration, i.e. move dislocations to force equilibrium
for i in range(5000):
    fsp, dt = dsl.move_disl(-tau0, Ntot, "viscous", dt0)

# Run gbdd simulation
gbdd = dd.GB_dislocations(temp=temp, tau0=tau0,
                   Ngbn=ngbn, len_gb_seg=lgb, grain_size=gs,
                   tfin=tfin, niter=niter, maxout=maxout,
                   dtmax=dtmax,
                   screenout=True)
gbdd.run_sim(pudis=dsl.ypos.copy())
gbdd.save_hdf5(f"gbpre_tau{int(tau0)}_T{int(temp)}_H{int(gs*10)}_L{int(lgb*10)}.h5")
gbdd.plot_time_series( ["n_slip_dis", "n_absorbed", "n_gbdis_eff"])
gbdd.plot_field('dGdb')
#gbdd.plot_field('flux')
gbdd.plot_field("bfield")
gbdd.plot_time_series("psr_av")
print(np.mean(gbdd.glob_data['psr_av'][500:-1]))
