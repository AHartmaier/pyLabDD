from pylabdd import GB_dislocations

temp = 900.0
tau0 = 0.0
lgb = 0.1
grain_size = 1.0
maxout = 1000
tfin = 8.e8
niter = 50_000_000
#ngbn = 41 #int(lgb/delta)
#if ngbn % 2 == 0: ngbn += 1
for ngbn in [21, 31, 41]:
    delta = lgb/(ngbn - 1)  # 0.005
    dtmax = 6.25e-6/delta**4  # value 1.e4 for delta=0.005, 50. for PU
    gbdd = GB_dislocations(temp=temp, tau0=tau0,
                       Ngbn=ngbn, len_gb_seg=lgb, grain_size=grain_size,
                       tfin=tfin, niter=niter, maxout=maxout,
                       dtmax=dtmax,
                       screenout=True)
    gbdd.run_sim()
    gbdd.save_hdf5(f"gb1_Ngbn{ngbn}_L{int(lgb*10)}_T{int(temp)}.h5")
