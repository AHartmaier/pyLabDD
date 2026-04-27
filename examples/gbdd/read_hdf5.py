import h5py
import json
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("test_T900_lgb01_Ngbn31_dt0.h5", "r") as f:  # opjbd
    time = f["time"][:]
    xval = f["xval"][:]
    values = f["values"][:]
    val_names = f["field_names"][:]
    glob_val = f["global"][:]
    glob_names = f["global_names"][:]
    meta = f["metadata_json"][0]
    try:
        dislocations = f["dislocations"][:]
        dis_names = f["dis_names"][:]
        dis_avail = True
    except:
        dis_avail = False
        print('No data on pilu-up dislocations available')

# Bytes -> Python-Strings
field_names = [
    name.decode("utf-8").strip() if isinstance(name, (bytes, np.bytes_)) else str(name).strip()
    for name in val_names
]
glob_names = [
    name.decode("utf-8").strip() if isinstance(name, (bytes, np.bytes_)) else str(name).strip()
    for name in glob_names
]

nt = len(time)
nout = 10
dt = max(1, int(nt / nout))
print("time.shape   =", time.shape)
print("xval.shape   =", xval.shape)
print("values.shape =", values.shape)
print("field_names  =", field_names)
print("glob_val.shape =", glob_val.shape)
print("glob_names  =", glob_names)

print("metadata.type =", type(meta))
#print(meta.decode("utf-8"))
mdict = json.loads(meta.decode("utf-8"))
print(mdict)

# Felder per Name in Dict ablegen
field_data = {name: values[:, i, :] for i, name in enumerate(field_names)}
glob_data = {name: glob_val[i, :] for i, name in enumerate(glob_names)}

#print(field_data['flux'][1:3, :])

# store parameters
ts = time*1.e-6
gs = mdict['parameters']['grain_size']
Ngbn = mdict['parameters']['number_GB_cells']

# process and plot dislocation data if available
if dis_avail:
    Npu_max = dislocations.shape[0]
    if Npu_max > 0:
        dis_names = [
            name.decode("utf-8").strip() if isinstance(name, (bytes, np.bytes_)) else str(name).strip()
            for name in dis_names
        ]
        print("dislocations.shape =", dislocations.shape)
        print("dis_names  =", dis_names)
        dis_data = {name: dislocations[:, i, :] for i, name in enumerate(dis_names)}

        # plot dislocation config
        for i in range(0, nt, dt):
            ind = np.nonzero(dislocations[:, 0, i])[0]
            for j in ind:
                plt.plot(dislocations[j, 0, i], ts[i],
                        marker='o',
                        linestyle='none',
                        color=plt.cm.viridis(j / Npu_max),
                        label=f'pile-up@t={ts[i]:.2f}s')
        plt.xlabel(r'dislocation position ($\mu$m)')
        plt.ylabel('time (s)')
        plt.xlim((0, gs*1.05))
        plt.show()

        plt.plot(ts, dislocations[0,1,:], '-k', label='force #0')
        plt.legend()
        plt.show()

# plot times series of global values
gv = glob_data['timestep'][1:]
plt.plot(ts[1:], gv/max(gv), '-b', label='Time step')
plt.plot(ts[1:], glob_data['n_gbdis_eff'][1:], '-r', label='total gb bv')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('normalized global data')
plt.show()

gv = glob_data['max_bdot'][1:]*1.e6
plt.semilogy(ts[1:], gv, '-k', label='max. bdot')
#plt.legend()
plt.xlabel('time (s)')
plt.ylabel(r'$\dot{b}_{max}$ (1/s)')
plt.show()

# plot field data over GB
for name in field_names:
    for i in range(0, nt, dt):
        #print(f"Plotting {name} at time {time[i]:.2e} s")
        yv = field_data[name][:, i]
        if name=='displacement':
            yv -= np.sum(yv)/Ngbn
        plt.plot(xval, yv, 
                 marker=None,  # '.',  'none', 
                 linestyle='-',  # 'none', '-', 
                 color=plt.cm.viridis(i / nt),
                 label=f'{name} t={ts[i]:.2f}s')
    plt.legend()
    plt.ylabel(name)
    plt.xlabel(r'x ($\mu$m)')
    plt.show()
