from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pylabdd import GB_dislocations as gbd

#files = ["gb1_L1_N21_T900_dt5.h5", "gb1_L3_N61_T900_dt5.h5",
#         "gb1_L10_N201_T900_dt5.h5", "gb1_L30_N601_T900_dt5.h5"]
files = ["gb1_L1_N21_T600_dt6.h5", "gb1_L3_N61_T600_dt6.h5",
         "gb1_L10_N201_T600_dt6.h5"]
maxout = 1000
Nf = len(files)
bdmax = np.zeros((Nf, maxout))
btot = np.zeros((Nf, maxout))
sim_time = np.zeros((Nf, maxout))
nit = np.zeros(Nf, dtype=int)
lgb_list = np.zeros(Nf, dtype=float)
ngbn_list = np.zeros(Nf, dtype=int)
tfin_list = np.zeros(Nf, dtype=float)
temp_list = np.zeros(Nf, dtype=int)
for i, f in enumerate(files):
    print(f'Analyzing file: {f}')
    gbdd = gbd(from_file=f)
    gbdd.plot_time_series(['timestep', 'n_gbdis_eff'])
    gbdd.plot_time_series('max_bdot', file=f"bdot_T{int(gbdd.temp)}_Lgb{int(gbdd.Dgp*10)}.pdf")
    gbdd.plot_field('displacement', file=f"T{int(gbdd.temp)}_Lgb{int(gbdd.Dgp*10)}.pdf")
    nit[i] = gbdd.nout
    sim_time[i, 0:gbdd.nout] = gbdd.sim_time
    bdmax[i, 0:gbdd.nout] = gbdd.glob_data['max_bdot']
    btot[i, 0:gbdd.nout] = gbdd.glob_data['n_gbdis_eff']
    lgb_list[i] = gbdd.Dgp
    ngbn_list[i] = gbdd.Ngbn
    temp_list[i] = int(gbdd.temp)
print(f'Done reading {Nf} files.')

for i in range(Nf):
    plt.plot(sim_time[i, :nit[i]]*1.e-9,
             btot[i, :nit[i]],
             marker="none",
             linestyle="-",
             color=plt.cm.viridis(i / max(1, Nf)),
             label=r"$L_{GB}$" + f"={lgb_list[i]}" + r" $\mu$m," + f" N={ngbn_list[i]}"
             )
plt.legend()
plt.xlabel(r"$t$ (10$^3$ s)")
plt.ylabel(r"$\sum b_i \,/ \,B$")
plt.savefig(f'bres_time_T{temp_list[i]}_var_lgb.pdf', dpi=300, format='pdf')
plt.show()

for i in range(Nf):
    ts = sim_time[i, :nit[i]]*1.e-9 / lgb_list[i]**3
    plt.plot(ts,
             btot[i, :nit[i]],
             marker="none",
             linestyle="-",
             color=plt.cm.viridis(i / max(1, Nf)),
             label=r"$L_{GB}$" + f"={lgb_list[i]}" + r" $\mu$m," + f" N={ngbn_list[i]}"
             )
plt.legend()
plt.xlabel(r"$t\,/\,L_{GB}^3$ (10$^3$ s/$\mu$m$^3$)")
plt.ylabel(r"$\sum b_i \,/ \,B$")
plt.savefig(f'bres_time_scaled_T{temp_list[i]}_var_lgb.pdf', dpi=300, format='pdf')
plt.show()