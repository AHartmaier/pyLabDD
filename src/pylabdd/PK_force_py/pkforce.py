# Module pylabdd.pkforce
'''Module pylabdd.pkforce introduces subroutine to calculate 
Peach-Koehler force either with periodic boundary conditions Calc_fpk_pbc() 
or in infinite medium calc_fpk() as pure python code. Used as fallback option
to faster F90 subroutines in case of issues during compilation. 

Kudos to ChatPGT for translation of F90 code.

uses NumPy

Author: Alexander Hartmaier, ICAMS/Ruhr-University Bochum, December 2023
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)
August 2025
'''


import numpy as np

def calc_fpk_pbc(xpos, ypos, bx, by, tau0, len_x, len_y, Nmob, N):
    """ Solution based on Eqs (2.1.25a) and (2.1.25b) from Linyong Pang "A new O(N) method for
    modeling and simulating the behavior of a large number of dislocations in
    anisotropic linear elastic media", PhD thesis, Stanford University, USA. 2001 
    """
    FPK = np.zeros((2, Nmob), dtype=np.float64)
    pih = np.pi / len_x
    pih2 = pih * pih
    imunit = 1.0j

    for j in range(Nmob):
        h11 = 0.0
        h22 = 0.0
        h12 = tau0
        px = xpos[j]
        py = ypos[j]

        for i in range(N):
            if j == i:
                continue
            hbx = bx[i]
            hby = by[i]
            hx = xpos[i]
            hy = ypos[i]

            for m in range(3):
                z = px - hx + (py - (hy + float(m) * len_y)) * imunit
                z = z * pih
                hcre = 1.0 / np.sin(z)
                hcot = np.cos(z) * hcre * pih
                hcre = hcre * hcre
                pxx1 = 2.0 * (hby - 2.0 * hbx * imunit)
                pxx2 = hbx + hby * imunit
                pxx3 = hbx * imunit
                pyy1 = pxx2 * 2.0 * (py - (hy + float(m) * len_y)) * pih2 * hcre
                h11 = h11 + np.real(pxx1 * hcot)
                h11 = h11 - np.real(pyy1)
                h22 = h22 + np.real(2.0 * hby * hcot)
                h22 = h22 + np.real(pyy1)
                h12 = h12 + np.imag(pxx3 * 2.0 * hcot)
                h12 = h12 + np.imag(pyy1)

                if m > 0:
                    z = px - hx + (py - hy + float(m) * len_y) * imunit
                    z = z * pih
                    hcre = 1.0 / np.sin(z)
                    hcot = np.cos(z) * hcre * pih
                    hcre = hcre * hcre
                    pxx1 = 2.0 * (hby - 2.0 * hbx * imunit)
                    pxx2 = hbx + hby * imunit
                    pxx3 = hbx * imunit
                    pyy1 = pxx2 * 2.0 * (py - hy + float(m) * len_y) * pih2 * hcre
                    h11 = h11 + np.real(pxx1 * hcot)
                    h11 = h11 - np.real(pyy1)
                    h22 = h22 + np.real(2.0 * hby * hcot)
                    h22 = h22 + np.real(pyy1)
                    h12 = h12 + np.imag(pxx3 * 2.0 * hcot)
                    h12 = h12 + np.imag(pyy1)

        FPK[0, j] = h12 * bx[j] + h22 * by[j]
        FPK[1, j] = -(h11 * bx[j] + h12 * by[j])

    return 0.5*FPK

def calc_fpk(xpos, ypos, bx, by, tau0, Nmob, N):
    FPK = np.zeros((2, Nmob), dtype=np.float64)

    for i in range(Nmob):
        xpi = xpos[i]
        ypi = ypos[i]
        h11 = 0.0
        h22 = 0.0
        h12 = tau0

        for j in range(N):
            if i == j:
                continue
            x = xpi - xpos[j]
            y = ypi - ypos[j]
            hx = x * x
            hy = y * y
            hh = hx + hy
            hh = hh * hh
            hbx = bx[j]
            hby = by[j]
            h11 = h11 - hbx * y * (3.0 * hx + hy) / hh
            h11 = h11 + hby * x * (hy - hx) / hh
            h22 = h22 + hbx * y * (hx - hy) / hh
            h22 = h22 - hby * x * (3.0 * hy + hx) / hh
            h12 = h12 + hbx * x * (hx - hy) / hh
            h12 = h12 + hby * y * (hy - hx) / hh

        FPK[0, i] = h12 * bx[i] + h22 * by[i]
        FPK[1, i] = -(h11 * bx[i] + h12 * by[i])

    return FPK
