import pylabdd as dd
import numpy as np

def run_iter(
    fill_fact=0.75,
    *,
    tau0=150.0,
    temp=500.0,
    len_gb_seg=0.1,
    grain_size=10.0,
    Ngbn=None,
    maxdis=1000,
    maxout=200,
    niter=20_000_000,
    tfin=0.0,
    dtmax=None,
    screenout=True,
    mu=44.e3,
    nu=0.3,
    B=0.25e-3,
    delta=0.004,
    Qact=95.e3,
    drag=500.,
    Dif_gb=10.,
    time_inc=0.5e9,
    max_iter=9,
    stability_window=100,
    stability_tol=2,
    variation_fraction=0.07,
    plot=True,
    save=True,
    output_prefix="gbpre",
    **gbdd_kwargs,
):
    """Run iterative GBDD simulations until the pile-up is stable.

    Parameters not used directly by this driver are forwarded to
    ``dd.GB_dislocations`` through ``gbdd_kwargs``.
    """
    if Ngbn is None:
        Ngbn = int(len_gb_seg / delta)
        if Ngbn % 2 == 0:
            Ngbn += 1
    if dtmax is None:
        dtmax = 6.25e-7 / delta**4

    gbdd_init = {
        "temp": temp,
        "tau0": tau0,
        "Ngbn": Ngbn,
        "len_gb_seg": len_gb_seg,
        "grain_size": grain_size,
        "maxdis": maxdis,
        "maxout": maxout,
        "niter": niter,
        "dtmax": dtmax,
        "screenout": screenout,
        "mu": mu,
        "nu": nu,
        "B": B,
        "delta": delta,
        "Qact": Qact,
        "drag": drag,
        "Dif_gb": Dif_gb,
    }
    gbdd_init.update(gbdd_kwargs)

    hc = mu * B / (2 * np.pi * (1. - nu))

    def initial_pileup_positions(fill_fact):
        ndis = int(fill_fact * np.pi * (1 - nu) * grain_size * tau0 / (mu * B))
        if ndis < 2:
            raise ValueError(
                f"Initial pile-up has {ndis} dislocations; increase fill_fact, grain_size, or tau0."
            )
        dmax = np.pi**2 * grain_size / (64 * fill_fact * ndis**2)
        ypos = np.zeros(ndis)
        ypos[1] = dmax
        for i in range(2, ndis):
            x0 = ypos[i - 1]
            ypos[i] = x0 + np.pi * hc / (tau0 * np.sqrt((grain_size * fill_fact - x0) / x0))
        return ypos.copy()

    # Set pre-existing dislocations in pile-up
    pudis = initial_pileup_positions(fill_fact)
    Ntot = pudis.size
    vout = None
    r_time = None
    nabs = None
    res_ = []
    it = 1
    ii = 1
    npu_const = False
    while not npu_const and it < max_iter:
        # Run gbdd simulation
        tfin += time_inc
        print(f"### it={it}, ii={ii}, tfin={tfin:.2e}, Ntot={Ntot}")
        gbdd = dd.GB_dislocations(tfin=tfin, **gbdd_init)
        gbdd.run_sim(pudis=pudis, vout=vout, r_time=r_time, nabs=nabs)
        if plot:
            gbdd.plot_time_series(["n_slip_dis", "n_absorbed", "n_gbdis_eff"])
            gbdd.plot_time_series("plast_slip_rate")
        if save:
            fname = (
                f"{output_prefix}_{ii}_{Ntot}_tau{int(tau0)}_T{int(temp)}_"
                f"H{int(grain_size*10)}_L{int(len_gb_seg*10)}.h5"
            )
            gbdd.save_hdf5(fname)
        res_.append(gbdd)

        if gbdd.nout < stability_window:
            raise ValueError(
                "Not enough output data to check pile-up stability. "
                "Increase max_iter or restart from an HDF5 file."
            )
        hh = gbdd.glob_data["n_slip_dis"][gbdd.nout-stability_window:gbdd.nout]
        hmax = max(hh)
        hmin = min(hh)
        hvar = hmax - hmin
        if hvar < stability_tol:
            npu_const = True
            print(f"Pile-up is stable at it={it}, ii={ii}, tfin={tfin:.2e}, Nslip={gbdd.glob_data['n_slip_dis'][-1]}")
        elif hvar > variation_fraction*hmax:
            sample_width = max(1, stability_window // 5)
            h1 = np.mean(gbdd.glob_data["n_slip_dis"][gbdd.nout-stability_window:gbdd.nout-stability_window+sample_width])
            h2 = np.mean(gbdd.glob_data["n_slip_dis"][gbdd.nout-sample_width:gbdd.nout])
            if h2-h1 > 0.:
                # number of slip dislocation increasing, pile-up is still growing, increase initial number of dislocations
                fill_fact *= 1.2
                print(f"Pile-up is growing, re-starting simulation with fill_fact={fill_fact}")
            else:
                # number of slip dislocation decreasing, pile-up is shrinking, decrease initial number of dislocations
                fill_fact *= 0.8
                print(f"Pile-up is shrinking, re-starting simulation with fill_fact={fill_fact}")
            pudis = initial_pileup_positions(fill_fact)
            Ntot = pudis.size
            vout = None
            r_time = None
            nabs = None
            res_ = []
            tfin = 0.
            ii = 0
        else:
            pudis=gbdd.pu_dis['position'][gbdd.nout-1, :].copy()
            vout=gbdd.vout[gbdd.nout-1, :, :].copy()
            r_time=float(gbdd.sim_time[gbdd.nout-1])
            nabs=gbdd.nabs
            print(f"Pile-up is not yet stable at it={it}, ii={ii}, r_time={r_time}, nout={gbdd.nout}, hvar={hvar}")
        it += 1
        ii += 1
    return res_


# set simulation parameters
temp = 800.0
tau0 = 150.0
b0 = 0.25e-3
nu = 0.3
mu = 44.e3
drag = 500.
Qact = 95.e3
Dif_gb = 10.
lgb = 0.1
gs = 10.0
delta = 0.004
ngbn = int(lgb/delta)
if ngbn % 2 == 0: ngbn += 1
dtmax = 6.25e-7/delta**4  # value 1.e4 for delta=0.005
time_inc = 0.5e9
maxout = 200
niter = 20_000_000

D2_list = [0.3, 1.0, 3.0, 10.0, 30.0]

for i, gs in enumerate(D2_list):
    print(f"=====  Simulation with N_gbn={ngbn}, grain_size={gs}, dtmax={dtmax}")  
    res_ = run_iter(
        fill_fact=0.75,
        temp=temp, tau0=tau0,
        Ngbn=ngbn, len_gb_seg=lgb, grain_size=gs,
        time_inc=time_inc, niter=niter, maxout=maxout,
        dtmax=dtmax, drag=drag,
        nu=nu, mu=mu, B=b0,
        Qact=Qact, Dif_gb=Dif_gb,
        screenout=True,
        plot=False,
        save=False,
        )
    gbo = res_[-1]
    gbo.save_hdf5(f"gb_final_tau{int(tau0)}_T{int(temp)}_"
                f"H{int(gs*10)}_L{int(lgb*10)}.h5", overwrite=True)
