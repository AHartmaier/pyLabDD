import numpy as np
import pylabdd as dd

def test_material():
    #check if the PK force values are consistent
    assert np.linalg.norm(ffp-ffa) < 1E-7
    assert np.linalg.norm(fff-ffa) < 1E-7
    
def calc_fpk_py(tau0, dsl):
    sigdxx = np.zeros(dsl.Ntot)
    sigdyy = np.zeros(dsl.Ntot)
    sigdxy = np.zeros(dsl.Ntot)

    for i in range(dsl.Ntot):
        jl = list(range(0,i)) + list(range(i+1,dsl.Ntot))
        xpi = dsl.xpos[i]
        ypi = dsl.ypos[i]
        h11 = 0.
        h22 = 0.
        h12 = 0.
        for j in jl:
            x = xpi-dsl.xpos[j]
            y = ypi-dsl.ypos[j]
            h11 += dsl.bx[j]*dsl.sig_xx(x, y)
            h11 += dsl.by[j]*dsl.sig_yy(y, x)
            h22 += dsl.bx[j]*dsl.sig_yy(x, y)
            h22 += dsl.by[j]*dsl.sig_xx(y, x)
            h12 += dsl.bx[j]*dsl.sig_xy(x, y)
            h12 += dsl.by[j]*dsl.sig_xy(y, x)
        sigdxx[i] = h11
        sigdyy[i] = h22
        sigdxy[i] = h12

    sigdxy += tau0   # add applied shear stress to internal stress
    hh1 = np.multiply(sigdxy, dsl.bx[0:dsl.Nmob]) + np.multiply(sigdyy,dsl.by[0:dsl.Nmob])
    hh2 = np.multiply(sigdxx, dsl.bx[0:dsl.Nmob]) + np.multiply(sigdxy,dsl.by[0:dsl.Nmob])
    return np.array([hh1, -hh2])

#define material parameters
#units: stress: MPa; length: micron; time: microseconds
mu = 80.0e3          # shear modulus
nu = 0.3             # Poisson ratio
b0 = 0.2e-3          # Burgers vector norm
C = mu*b0/(2*np.pi*(1.-nu))   # Constant for dislocation stress field
f0 = 10.             # initial slip resistance

#box geometry
LX = 100.             # box dimension in x-direction
LY = 100.             # box dimension in y-direction
bc = 'pbc'            # set boundary conditions: 'fixed' or 'pbc'

#define numerical parameters
dt0 = 0.02           # time step
np.random.seed(110)  # seed RNG

#Validation of different methods for calculation of PK force
d1 = dd.Dislocations(5,5,0.,C,b0, LX=LX, LY=LY, bc=bc, dt0=dt0)
d1.positions()
tau0=0.

# Fortran subroutine for periodic BC
ffp = d1.calc_force(tau0=tau0, lx=1.e6*LX, ly=1.e6*LY)
# fortran subroutine for fixed BC
d1.bc = 'fixed'
fff = d1.calc_force(tau0=tau0)
# Python expression (reference)
ffa = calc_fpk_py(tau0, d1)



