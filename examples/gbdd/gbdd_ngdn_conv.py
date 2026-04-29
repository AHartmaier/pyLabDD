import time
import numpy as np
import matplotlib.pyplot as plt
from pylabdd import GB_dislocations

delta = 0.005
maxout = 1000
tfin = 1.e8
niter = 50_000_000
Ngb_list = np.arange(11, 50, 10, dtype=int)
Np = len(Ngb_list)
bdmax = np.zeros((Np, maxout))
btot = np.zeros((Np, maxout))
sim_time = np.zeros((Np, maxout))
dur = np.zeros(Np)
nit = np.zeros(Np, dtype=int)
for i,n in enumerate(Ngb_list):
    dtmax = 4.e6/n**4
    print(f'Number of GB nodes: {n}, max. timestep: {dtmax}')
    t0 = time.time()
    gbdd = GB_dislocations(temp=900.0, Ngbn=n, len_gb_seg=0.1, tfin=tfin,
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
             label=f"N={Ngb_list[i]}, sim={dur[i]:.2f}s"
             )
plt.legend()
plt.xlabel("time (s)")
plt.ylabel(r"$\sum b_i \,/ \,N$")
plt.savefig('conv_ngbn.pdf', dpi=300, format='pdf')
plt.show()

