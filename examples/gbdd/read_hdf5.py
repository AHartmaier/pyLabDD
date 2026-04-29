from __future__ import annotations

from typing import Any
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt

def _read_attrs(group: h5py.Group) -> dict[str, Any]:
    """Read HDF5 attributes and decode byte strings."""
    out: dict[str, Any] = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            out[key] = value.decode()
        elif isinstance(value, np.bytes_):
            out[key] = value.decode()
        else:
            out[key] = value
    return out
        
with h5py.File("gb1_L3_N61_T900_dt5.h5", "r") as f:
    time = f["time"][()]
    xval = f["xval"][()]
    gbfield = f["gb/fields"][()]
    glob_val = f["global/fields"][()]
    dislocations = f["pileup/fields"][()] if "pileup" in f else None
    mdict = _read_attrs(f["metadata"]) if "metadata" in f else {}
    attrs = _read_attrs(f)
    names = _read_attrs(f["gb"])["field_names"]
    field_names = [
        name.decode("utf-8").strip() if isinstance(name, (bytes, np.bytes_)) else str(name).strip()
        for name in names
    ]
    names = _read_attrs(f["global"])["field_names"]
    glob_names = [
        name.decode("utf-8").strip() if isinstance(name, (bytes, np.bytes_)) else str(name).strip()
        for name in names
    ]
    names = _read_attrs(f["pileup"])["field_names"]
    dis_names = [
        name.decode("utf-8").strip() if isinstance(name, (bytes, np.bytes_)) else str(name).strip()
        for name in names
    ]

nt = len(time)
nout = 10
dt = max(1, int(nt / nout))
print("time.shape   =", time.shape)
print("xval.shape   =", xval.shape)
print("gbfield.shape =", gbfield.shape)
print("field_names  =", field_names)
print("glob_val.shape =", glob_val.shape)
print("glob_names  =", glob_names)
print(attrs)
print(mdict.keys())

# Felder per Name in Dict ablegen
field_data = {name: gbfield[:, i, :] for i, name in enumerate(field_names)}
glob_data = {name: glob_val[:, i] for i, name in enumerate(glob_names)}

# store parameters
ts = time*1.e-6
gs = mdict['grain_size']
Ngbn = mdict['Ngbn']

# process and plot dislocation data if available
Npu_max = mdict["npu_max"]
if Npu_max > 0:
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
print(len(gv), len(ts))
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
        yv = field_data[name][i, :]
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
