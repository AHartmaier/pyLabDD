#!/home/users/hartmajg/miniconda3/envs/pylabdd/bin/python
import pylabdd as dd
import numpy as np

path = "./"
fname = "gb_final_tau100_T700_H600_L1.h5"
gbr = dd.GB_dislocations(from_file=path+fname)
gbdd_init = {
    "temp": gbr.temp,
    "tau0": gbr.tau0,
    "Ngbn": gbr.Ngbn,
    "len_gb_seg": gbr.Dgp,
    "grain_size": gbr.D2,
    "maxdis": gbr.maxdis,
    "maxout": gbr.maxout,
    "niter": gbr.niter,
    "dtmax": gbr.dtmax,
    "screenout": gbr.screenout,
    "mu": gbr.mu,
    "nu": gbr.nu,
    "B": gbr.B,
    "delta": gbr.delta,
    "Qact": gbr.Qact,
    "drag": gbr.drag,
    "Dif_gb": gbr.Dif_gb,
}

pudis=gbr.pu_dis['position'][gbdd.nout-1, :].copy()
vout=gbr.vout[gbdd.nout-1, :, :].copy()
r_time=float(gbdd.sim_time[gbdd.nout-1])
nabs=gbr.nabs
gbdd_init.update({
    "maxout": 800,
    "niter": 80_000_000,
    "tfin": r_time + 2.0e9,
    "screenout": True,
})
print(gbdd_init)
gbdd = dd.GB_dislocations(**gbdd_init)
print(f"r_time={r_time}, nout={gbr.nout}, tfin={gbdd.tfin}")
gbdd.run_sim(pudis=pudis, vout=vout, r_time=r_time, nabs=nabs)
fname = (
            f"gb_final2_tau{int(gbdd.tau0)}_T{int(gbdd.temp)}_"
            f"H{int(gbdd.D2*10)}_L{int(gbdd.Dgp*10)}.h5"
        )
gbdd.save_hdf5(path+fname)