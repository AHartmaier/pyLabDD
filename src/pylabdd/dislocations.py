# Module pylabdd.dislocations
'''Module pylabdd.dislocations introduces class ``Dislocations`` that contains attributes 
and methods needed to handle a dislocation configuration. 

uses NumPy and MatPlotLib.pyplot

Author: Alexander Hartmaier, ICAMS/Ruhr-University Bochum, December 2023
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)
August 2025
'''

import logging
import os
import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#define class for dislocations
class Dislocations:
    '''Define class for Dislocations

    Parameters
    ----------
    Nd : int
        Total number of dislocations
    Nm : int
        Number of mobile dislocations

    Attributes
    ----------
    xpos  : Nd-array
        x-positions of dislocations
        
    '''
    def __init__(self, Nd, Nm, spi1, C, b0, \
                dmob=1., f0=0.8, m=7, dmax=0.002, \
                xpos=None, ypos=None,\
                LX=10., LY=10., bc='pbc',\
                dt0=0.02
                ):
        # import locally from installed package, as either F90 subroutines from PK_force
        # might be used or slower Python subroutines from PK_force_py as fallback option 
        # in case of compilation issues
        from pylabdd import calc_fpk_pbc, calc_fpk
        self.cfpk = calc_fpk
        self.cfpk_pbc = calc_fpk_pbc
        
        self.Ntot = Nd   # total number of dislocation
        self.Nmob = Nm   # number of mobile dislocations

        #dislocation positions
        if xpos is None:
            self.xpos = np.zeros(Nd)
        else:
            if 'float' in str(type(xpos)):
                xpos = [xpos]
            self.xpos = np.array(xpos)
        if ypos is None:
            self.ypos = np.zeros(Nd)
        else:
            if 'float' in str(type(ypos)):
                ypos = [ypos]
            self.ypos = ypos
        self.dx   = np.zeros(Nd)
        self.dy   = np.zeros(Nd)
        self.xpeq = None # equilibrium positions, will be defined in Dislocation.relax_disl
        self.ypeq = None
    
        #slip plane inclination angles
        self.sp_inc = np.ones(Nd)*spi1
        
        #Burgers vectors
        self.bx = np.cos(self.sp_inc)
        self.by = np.sin(self.sp_inc)

        #calculate dislocation densities
        self.rho = Nd/(LX*LY)
        self.rho_m = Nm/(LX*LY)
        
        #dislocation mobility parameters
        self.b0   = b0     # norm of Burgers vector
        self.C    = C      # elastic parameter C=mu*b0/(2*pi*(1.-nu))
        self.dmob = dmob   # dislocation mobility
        self.f0   = f0     # lattice friction stress
        self.m    = m      # stress exponent
        self.dmax = dmax   # max. distance a dislocation can move in one time step
        
        #geometry of the domain
        self.lx = LX      # x-dimension of domain
        self.ly = LY      # y-dimension of domain
        self.bc = bc      # chose between 'pbc' and 'fixed'
        if bc!='pbc' and bc!='fixed':
            raise ValueError('BC not defined: '+bc)

        #numerical parameters
        self.dt0 = dt0

    #define functions for stress field evaluation
    def sig_xx(self, X, Y):
        hx = np.multiply(X, X)
        hy = np.multiply(Y, Y)
        hh = hx + hy
        return -self.C*Y*(3.*hx + hy)/(hh*hh)
    
    def sig_yy(self, X, Y):
        hx = np.multiply(X, X)
        hy = np.multiply(Y, Y)
        hh = hx + hy
        return self.C*Y*(hx - hy)/(hh*hh)
    
    def sig_xy(self, X, Y):
        hx = np.multiply(X, X)
        hy = np.multiply(Y, Y)
        hh = hx + hy
        return self.C*X*(hx - hy)/(hh*hh)
    
    '''#create single new dislocation
    def create(self, x, y, bx, by):
        self.xpos = np.append(self.xpos, x)
        self.ypos = np.append(self.ypos, y)
        self.bx = np.append(self.bx, bx)
        self.by = np.append(self.by, by)'''
        
    def calc_force(self, xp=None, yp=None, Nm=None, tau0=None,
                   lx=None, ly=None):
        if xp is None:
            xp = self.xpos
        if yp is None:
            yp = self.ypos
        if Nm is None:
            Nm = self.Nmob
        if tau0 is None:
            tau0 = 0.
        if lx is None:
            lx = self.lx
        if ly is None:
            ly = self.ly
        if self.bc=='pbc':
            FPK = self.C* self.cfpk_pbc(xp, yp, self.bx, self.by, tau0, 
                                      lx, ly, Nm, self.Ntot)
        else:
            FPK = self.C* self.cfpk(xp, yp, self.bx, self.by, tau0, Nm, self.Ntot)
        return FPK
        
    #initialize random dislocation positions
    def positions(self, stol=0.25):
        #select slip planes first by random sequential algorithm
        #make sure that slip planes are at least a distance of stol apart
        self.ypos[0] = self.ly*np.random.rand(1)[0]
        isl = 1
        while isl<self.Ntot:
            hy = self.ly*np.random.rand(1)[0]
            flag = np.logical_and(self.ypos[0:isl]<hy+stol, self.ypos[0:isl]>hy-stol)
            if not np.any(flag):
                self.ypos[isl] = hy
                isl += 1
    
        #place dislocations randomnly on slip planes
        hh = np.random.rand(self.Ntot)
        self.xpos = self.lx*hh
        self.ypos += np.sin(self.sp_inc)*hh*self.ly
        ih = np.nonzero(self.ypos<0.)[0]
        self.ypos[ih] += self.ly
        ih = np.nonzero(self.ypos>self.ly)[0]
        self.ypos[ih] -= self.ly
        
        #bx = np.multiply(bx, np.sign(np.random.rand(N)-0.5))  # random positive and negative Burgers vectors
        self.bx[0:self.Ntot:2] *= -1. # change sign of every second dislocation
        self.by[0:self.Ntot:2] *= -1.
    
    #define force norm for relaxation with L-BFGS-B method
    def fnorm(self, dr, tau0, Nm):
        xp = self.xpos
        yp = self.ypos
        dx = np.multiply(dr, np.abs(self.bx[0:Nm]))
        dy = np.multiply(dr, np.abs(self.by[0:Nm]))
        xp[0:Nm] += dx
        yp[0:Nm] += dy
        if self.bc=='pbc':
            FPK = self.C* self.cfpk_pbc(xp, yp, self.bx, self.by, tau0, 
                                          self.lx, self.ly, Nm, self.Ntot)
        else:
            FPK = self.C* self.cfpk(xp, yp, self.bx, self.by, tau0, Nm, self.Ntot)
        fsp = np.sum(np.multiply(FPK,np.absolute(np.array([self.bx[0:Nm],\
                                                self.by[0:Nm]]))), axis=0)               
        return np.sum(np.absolute(fsp))/Nm

    #calculate dislocation velocity
    def dvel(self, fsp, ml):
        if ml=='viscous':
            hh = fsp
        elif ml=='powerlaw':
            hh = np.multiply(np.abs(fsp/self.f0)**self.m, np.sign(fsp))
        else:
            raise ValueError('Dislocation mobility ""'+ml+'" not supported.')
        return hh*self.dmob
    
    #update dislocation positions
    def move_disl(self, tau0, Nm, ml, dt, bc=None):
        if bc is None:
            bc = self.bc
        if bc=='pbc':
            FPK = self.C* self.cfpk_pbc(self.xpos, self.ypos, self.bx, self.by, \
                                         tau0, self.lx, self.ly, Nm, self.Ntot)
            FPK[:][1] *= -1. 
            #define maximum dislocation displacement
            lb = -self.dmax
            ub = self.dmax
        elif bc=='fixed':
            FPK = self.C* self.cfpk(self.xpos, self.ypos, self.bx, self.by, tau0, Nm, self.Ntot)
            #define possible range to move a dislocation within box
            lb = -np.minimum(np.abs(self.xpos[0:Nm]/self.bx[0:Nm]), np.ones(Nm)*self.dmax)
            ub =  np.minimum(np.abs((self.lx-self.xpos[0:Nm])/self.bx[0:Nm]), np.ones(Nm)*self.dmax)
        else:
            raise ValueError('BC not defined: '+bc)
        fsp = np.sum(np.multiply(FPK,np.abs(np.array([self.bx[0:Nm],self.by[0:Nm]]))), axis=0)
        drp = self.dvel(fsp, ml)*dt  # predictor for simple forward Euler integration dr = v.dt
        drp = np.clip(drp, lb, ub)  # enforce speed limit for dislocations and make sure they stay in box
        dr  = np.zeros(self.Ntot)
        dr[0:Nm] += drp  # only Nm dislocations are moved, the rest is fixed
        #do some analysis for time step control
        hh = np.abs(drp)
        dr_max = np.amax(hh)
        nmax = np.nonzero(hh>=self.dmax)[0]
        self.dx = np.multiply(dr, np.abs(self.bx))  # projection on slip plane (defined by B-vector)
        self.dy = np.multiply(dr, np.abs(self.by))
        xp = self.xpos + self.dx
        yp = self.ypos + self.dy
        #verify if force after predictor step has same sign as before
        #if not dislocation passes a minimum position and needs a reduced time step
        ih = np.array([1, 1])  # initialize ih such that while is performed at least once
        jc = 0
        while len(ih)>0 and jc<5:
            if bc=='pbc':
                FPK = self.C* self.cfpk_pbc(xp, yp, self.bx, self.by, \
                                             tau0, self.lx, self.ly, Nm, self.Ntot)
            elif bc=='fixed':
                FPK = self.C* self.cfpk(xp, yp, self.bx, self.by, tau0, Nm, self.Ntot)
            fsp2 = np.sum(np.multiply(FPK,np.absolute(np.array([self.bx[0:Nm],\
                   self.by[0:Nm]]))), axis=0)               
            hh = fsp*fsp2
            ih = np.nonzero(hh<0.)[0]
            #dislocation with indices ih traversed a minimum and need special treatment
            #reduce speed of dislocation to prevent them from crossing zero force position
            if jc==4:
                self.dx[ih] = 0.
                self.dy[ih] = 0.
                fsp[ih] = 0.
            self.dx[ih] *= 0.5
            self.dy[ih] *= 0.5
            xp[ih] = self.xpos[ih] + self.dx[ih]
            yp[ih] = self.ypos[ih] + self.dy[ih]
            jc += 1
        #update positions according to boundary conditions
        if bc=='fixed':
            self.xpos = np.clip(xp, 0, self.lx)
            self.ypos = np.clip(yp, 0, self.ly)
            bc1 = np.logical_or(self.xpos==0, self.ypos==0)
            bc2 = np.logical_or(self.xpos==self.lx, self.ypos==self.ly)
            ih = np.nonzero(np.logical_or(bc1, bc2))
            fsp[ih] = 0.
            self.dx[ih] = 0.
            self.dy[ih] = 0.
        elif bc=='pbc':
            self.xpos = xp
            self.ypos = yp
            ih = np.nonzero(self.xpos<0.)
            self.xpos[ih] += self.lx
            ih = np.nonzero(self.ypos<0.)
            self.ypos[ih] += self.ly
            ih = np.nonzero(self.xpos>self.lx)
            self.xpos[ih] -= self.lx
            ih = np.nonzero(self.ypos>self.ly)
            self.ypos[ih] -= self.ly
        #time step control
        if len(nmax)>2:
            dt = np.maximum(self.dt0*0.02, dt*0.9)    # reduce time step im more than 3 dislocation are fast
        elif dr_max<self.dmax*0.9:
            dt = np.minimum(self.dt0*50, dt*1.1)     # increase time step if all dislocations are slow
        return fsp, dt

    # relax all dislocation if True, otherwise only mobile dislocations are relaxed
    def relax_disl(self, relax_all=False, ftol=5.e-2, dt=0.02, plot_conf=False, 
                   plot_relax=True):
        # ftot acceptable residual error in force relaxation
        if relax_all:
            Nm = self.Ntot
        else:
            Nm = self.Nmob
        # initialze parameters for relaxation    
        fn = 2.*ftol
        nl = 0
        nout = 1000
        fout= int(50000/nout)
        fd = []
        dt = 0.03
        while fn>ftol and nl<50000:
            fsp, dt = self.move_disl(0., Nm, 'viscous', dt)  # move dislocations w/o ext. stress, motion is viscous for relaxation
            fn = np.sum(np.absolute(fsp))/Nm
            nl += 1
            if plot_relax and np.mod(nl,fout)==0:
                fd.append(fn)
            if plot_conf and np.mod(nl,5000)==0:
                self.plot_stress()
                print('Iteration:', nl, ', residual force:',fn)
        self.xpeq = self.xpos  # store equilibrium positions
        self.ypeq = self.ypos
        if plot_conf:
            self.plot_stress()
            print('Final configuration', nl, fn)
        if plot_relax:
            fd.append(fn)
            fd = np.array(fd)
            plt.semilogy(fd)
            plt.title('Dislocation structure relaxation')
            plt.xlabel('iteration')
            plt.ylabel('PK force norm')
            plt.show()
        return

    #calculate and plot stress field on grid
    def plot_stress(self):
        ngp = 150  # number of grid points
        #dx = self.lx/ngp
        #dy = self.ly/ngp
        xp = np.linspace(0, self.lx, ngp)
        yp = np.linspace(0, self.ly, ngp)
        XP, YP = np.meshgrid(xp, yp)
        s11 = np.zeros((ngp,ngp))
        s22 = np.zeros((ngp,ngp))
        s12 = np.zeros((ngp,ngp))
        for i in range(self.Ntot):
            s11 += self.bx[i]*self.sig_xx(XP-self.xpos[i], YP-self.ypos[i])
            s11 += self.by[i]*self.sig_yy(YP-self.ypos[i], XP-self.xpos[i])
            s22 += self.bx[i]*self.sig_yy(XP-self.xpos[i], YP-self.ypos[i])
            s22 += self.by[i]*self.sig_xx(YP-self.ypos[i], XP-self.xpos[i])
            s12 += self.bx[i]*self.sig_xy(XP-self.xpos[i], YP-self.ypos[i])
            s12 += self.by[i]*self.sig_xy(YP-self.ypos[i], XP-self.xpos[i])
    
        extent = (0, self.lx, 0, self.ly)
        fig, axs  = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
        fig.subplots_adjust(hspace=0.2)

        [axs[i].set_xlabel(r'x ($\mu$m)') for i in range(3)]
        [axs[i].set_ylabel(r'y ($\mu$m)') for i in range(3)]
        axs[0].set_title(r'$\sigma_{xx}$ (MPa)')
        axs[1].set_title(r'$\sigma_{yy}$ (MPa)')
        axs[2].set_title(r'$\sigma_{xy}$ (MPa)')
        im = axs[0].imshow(s11, origin='lower', extent=extent, vmin=-8., vmax=8., cmap=cm.RdBu)
        #fig.colorbar(im, ax=axs[0])
        im = axs[1].imshow(s22, origin='lower', extent=extent, vmin=-8., vmax=8., cmap=cm.RdBu)
        #fig.colorbar(im, ax=axs[1])
        im = axs[2].imshow(s12, origin='lower', extent=extent, vmin=-8., vmax=8., cmap=cm.RdBu)
        fig.colorbar(im, ax=axs[2])
    
        # plot markers for dislocations if not too many
        if self.Ntot<10:
            [axs[i].scatter(self.xpos, self.ypos, s=30, c='yellow', marker='o') for i in range(3)]
        
        #plot arrows for mobile dislocations
        for i in range(self.Nmob):
            dx = self.dx[i]
            dy = self.dy[i]
            hh = dx*dx + dy*dy
            if hh<self.b0:
                dx = self.bx[i]
                dy = self.by[i]
            axs[0].arrow(self.xpos[i], self.ypos[i], 4*dx, 4*dy, head_width=1.5,
                     width=0.5, head_length=2, color='#20ff00')
            axs[1].arrow(self.xpos[i], self.ypos[i], 4*dx, 4*dy, head_width=1.5,
                     width=0.5, head_length=2, color='#20ff00')
            axs[2].arrow(self.xpos[i], self.ypos[i], 4*dx, 4*dy, head_width=1.5,
                     width=0.5, head_length=2, color='#20ff00')
        fig.tight_layout()
        plt.show()

    #create line plot with Peach Koehler force
    def calc_PKforce(self, hy, ngp=150, x1=0.01, x2=None):
        '''Calculate Peach-Koehler (PK) force along a given plane within the 
        current dislocation configuration
        
        Parameters:
        hy  : float
            y-offset for line of which PK-force is calculate
        npg (optional) : int
            number of grid points
        x1 (optional)  : int
            start point of line plot
        x2 (optional)  : int 
            end point of line plot
        
        #Returns:
        fpk : (npg,)-array
            PK force (units: mN/m)
        xp  : (npg,)-array
            x-positions at which PK force is evaluted (units: micron)
        '''
        
        if x2 is None:
            x2=self.LX
        Nd = len(self.xpos)  # number of dislocation in group
        xp = np.linspace(x1, x2, num=ngp)
        yp = np.ones(ngp)*hy
        fpk = np.zeros(ngp)
        for i in range(Nd):
            fpk += self.b0*self.bx[i]*self.sig_xy(xp-self.xpos[i], yp-self.ypos[i])
            fpk -= self.b0*self.by[i]*self.sig_xy(yp-self.ypos[i], xp-self.xpos[i])
        return fpk*1000, xp
