import time
import numpy as np
import matplotlib.pyplot as plt
from pylabdd import GB_dislocations

delta = 0.005
temp = 900.0
maxout = 1000
tfin = 0.8e9
niter = 5_000_000
dtmax = 6.25e-6/delta**4  # value 1.e4 for delta=0.005
#ngbn = 21  # std. value for L_GB=0.1 and delta=0.005
lgb_list = np.array([0.1, 0.3, 0.5])
tfin_list = np.array([0.8e9, 16.e9, 100.e9])
Np = len(lgb_list)
bdmax = np.zeros((Np, maxout))
btot = np.zeros((Np, maxout))
sim_time = np.zeros((Np, maxout))
dur = np.zeros(Np)
nit = np.zeros(Np, dtype=int)
ngbn_list = np.zeros(Np, dtype=int)
for i,val in enumerate(lgb_list):
    ngbn = int(val / delta)
    if ngbn % 2 == 0:
        ngbn += 1
    ngbn_list[i] = ngbn
    print(f'Number of GB nodes: {ngbn}, max. timestep: {dtmax}, len GB segment: {val}')
    t0 = time.time()
    gbdd = GB_dislocations(temp=temp, Ngbn=ngbn, len_gb_seg=val, tfin=tfin_list[i],
                           niter=niter, maxout=maxout, dtmax=dtmax, screenout=True)
    gbdd.run_sim()
    t1 = time.time()
    dur[i] = t1 - t0
    gbdd.plot_time_series(['timestep', 'n_gbdis_eff'])
    gbdd.plot_time_series('max_bdot')
    gbdd.plot_field('displacement')
    nit[i] = gbdd.nout
    sim_time[i, 0:gbdd.nout] = gbdd.sim_time
    bdmax[i, 0:gbdd.nout] = gbdd.glob_data['max_bdot']
    btot[i, 0:gbdd.nout] = gbdd.glob_data['n_gbdis_eff']

for i in range(Np):
    plt.plot(sim_time[i, :nit[i]]*1.e-6,
             btot[i, :nit[i]],
             marker="none",
             linestyle="-",
             color=plt.cm.viridis(i / max(1, Np)),
             label=r"$L_{GB}$" +f"={lgb_list[i]}, N={ngbn_list[i]}, sim={dur[i]:.2f}s"
             )
plt.legend()
plt.xlabel("time (s)")
plt.ylabel(r"$\sum b_i \,/ \,N$")
plt.savefig(f'bres_time_t{int(temp)}_var_lgb.pdf', dpi=300, format='pdf')
plt.show()

for i in range(Np):
    ts = sim_time[i, :nit[i]]*1.e-9 / lgb_list[i]**3
    plt.plot(ts,
             btot[i, :nit[i]],
             marker="none",
             linestyle="-",
             color=plt.cm.viridis(i / max(1, Np)),
             label=r"$L_{GB}$" +f"={lgb_list[i]}, N={ngbn_list[i]}, sim={dur[i]:.2f}s"
             )
plt.legend()
plt.xlabel(r"$t\,/\,L_{GB}^3$ (1000 s)")
plt.ylabel(r"$\sum b_i \,/ \,N$")
plt.show()

