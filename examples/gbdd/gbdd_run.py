import time
import numpy as np
import matplotlib.pyplot as plt
from pylabdd import GB_dislocations

delta = 0.005
maxout = 1000
tfin = 0.8e9
niter = 5_000_000
ngbn = 31
dt_list = np.array([100., 1.e3, 1.e4, 1.e5], dtype=float)
Np = len(dt_list)
bdmax = np.zeros((Np, maxout))
btot = np.zeros((Np, maxout))
sim_time = np.zeros((Np, maxout))
dur = np.zeros(Np)
nit = np.zeros(Np, dtype=int)
for i,n in enumerate(dt_list):
    # dtmax = 4.e6/ngbn**4
    print(f'Number of GB nodes: {n}, max. timestep: {n}')
    t0 = time.time()
    gbdd = GB_dislocations(temp=900.0, Ngbn=ngbn, len_gb_seg=0.1, tfin=tfin,
                           niter=niter, maxout=maxout, dtmax=n, screenout=True)
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
             label=f"N={ngbn}, dt_max={dt_list[i]:1.1e}, sim={dur[i]:.2f}s"
             )
plt.legend()
plt.xlabel("time (s)")
plt.ylabel(r"$\sum b_i \,/ \,N$")
plt.savefig(f'conv_dtmax_n{ngbn}.pdf', dpi=300, format='pdf')
plt.show()

