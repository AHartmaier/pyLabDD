! Grain Boundary Dislocation Dynamics
! version 2.1.2
! based on GB-dislo, version 2013-11-18
! 2026-04-22: v1.0.0: Initial version, added HDF5 output
! 2026-04-22: v1.1.0: Corrected mathematical expressions for Peach-Kohler force and GB elastic field
! 2026-04-24: v1.1.1: Added output for pile-up dislocations
! 2026-04-24: v2.0.0: Split into subroutine and lean main for Python integration
! 2026-04-24: v2.1.0: Strip HDF5 output for Python integration
! 2026-04-27: v2.1.1: Introduced params-vector to interface to pass material parameters
! 2026-04-29: v2.1.2: Updated equations for dGdb with damping factor df, inactivate PRT file
! 2026-05-04: v2.1.3: tau0 controls initial conditions:
!                     tau0=0: start with one absobed dis, no pileup; tau0>0: no absobed dis, pileup nucleation active
!                     added Nabs to return values; fixed bug in tau_GB wrt tau0
!
! Author: Alexander Hartmaier
! Institution: Ruhr-Universitaet Bochum, ICAMS
! Copyright (c) 2013-2026 by the Author. All rights reserved.
! This code can be used under the terms of the GNU General Public License version 3 (GNU GPL-3.0)

subroutine calc_gbdd(params, nparams, tau0, temp, Dgp, D2, Ngbn, maxdis, tfin, niter, dtmax, &
    it, Npu_max, ih5, Nabs, time_out, xout, vout, pu_out, globout, screen_out)

    implicit none

    integer, parameter :: name_len = 16
    integer, parameter :: nsav = 200
    integer, parameter :: IDX_MU     = 1
    integer, parameter :: IDX_NU     = 2
    integer, parameter :: IDX_B      = 3
    integer, parameter :: IDX_DELTA  = 4
    integer, parameter :: IDX_QACT   = 5
    integer, parameter :: IDX_DRAG   = 6
    integer, parameter :: IDX_DIFGB  = 7
    integer, parameter :: IDX_FCRIT  = 8
    integer, parameter :: N_GBDD_PARAMS = 8

    logical, intent(in) :: screen_out  ! print output on screen
    integer, intent(in) :: nparams  ! number of constitutive parameters
    integer, intent(in) :: Ngbn    ! number of gain boundary nodes, should be odd to have a center node
    integer, intent(in) :: niter   ! maximum number of iteration steps
    integer, intent(in) :: maxdis  ! maximum number of dislocations in pile-up
    integer, intent(out) :: Npu_max ! maximum number of dislocation in pile-up reached
    integer, intent(out) :: it !number of iterations
    integer, intent(out) :: ih5  ! number of outputs
    integer, intent(out) :: Nabs ! number of absorbed dislocations
    real(8), intent(in) :: params(nparams)  ! Vector for constitutive parameters
    real(8), intent(in) :: tfin  ! final sim_time for simulation (microseconds), std: 25d6
    real(8), intent(in) :: dtmax ! 1.d3 w/o pu; 60 with pile up
    real(8), intent(in) :: D2    ! Grain size D/2, distance FR source-GB (micron)
    real(8), intent(in) :: Dgp   ! Length of GB segment considered (micron); size 0.005
    real(8), intent(in) :: tau0  ! applied shear stress (MPa) 
    real(8), intent(in) :: temp  ! temperature
    real(8), intent(out) :: vout(:, :, :), time_out(:), xout(:)
    real(8), intent(out) :: pu_out(:, :, :), globout(:, :)
    real(8) :: bfield(Ngbn), ypu(maxdis), fdis(maxdis), vdis(maxdis)
    real(8) :: M, C, DC, B, mu, nu, Omega, pi, gbdx, drag
    real(8) :: R, delta, Dif_gb, D0, Qact, df, fcrit
    real(8) :: twr0  ! sim_time interval (microseconds) for sim_time series output, std: 20d6
    real(8) :: frk(maxdis), yrk(maxdis), Jf(Ngbn), bdot(Ngbn)
    real(8) :: dGdb(Ngbn), Upot(Ngbn, Ngbn)
    real(8) :: ts(nsav), gps(nsav)
    real(8) :: dt, ttot, eps
    real(8) :: gdpl, gplast, vmax, bdmax
    real(8) :: epsmax, edtot, cdist, dsrc
    real(8) :: twr
    real(8) :: hh, hh1, hh2, hx2, hy2, gbdxD2, hc, omdx
    integer :: Nc, Npu
    integer :: nfields, nglob, maxout, nwr
    integer :: i, k, ih, iwr, j

    !character (len=24) pname

    maxout = size(time_out)
    nfields = size(vout, 2)
    nglob = size(globout, 2)

    if (nparams < N_GBDD_PARAMS) then
        error stop "GBDD parameter vector is too short."
    end if
    if (nparams > N_GBDD_PARAMS) then
        error stop "GBDD parameter vector is too long."
    end if
    call init()
    print*, "START", temp, Ngbn, Dgp

    ih = int(D2*10)
    !pname = 'gbdd-tau'// char(int(tau0/100)+48)//char(mod(int(tau0),100)/10+48) // &
    !pname = 'gb1pt_L001'// &
    !    '-T' // char(int(temp)/100+48) // char(mod(int(temp),100)/10+48) // &
    !    '-d' // char(ih/1000+48) // char(mod(ih,1000)/100+48) // char(mod(ih,100)/10+48) // char(mod(ih,10)+48) //'.prt'

    !start condition if no applied stress: one absorbed dislocation
    if (abs(tau0) < 1.d-6) then
        bfield(Nc) = B
        Nabs = 1
    end if
    ! store initial values for GB nodes in output array
    hh = 0.d0
    do i=1,Ngbn
        vout(1,1,i) = Jf(i)
        vout(1,2,i) = dGdb(i)
        vout(1,3,i) = bdot(i)/B
        vout(1,4,i) = bfield(i)/B
        vout(1,5,i) = hh
        hh = hh + bfield(i)/B
    end do
    iwr = 2
    it = 1

    ! ---------------------------
    ! write protocol file for sim_time series output
    ! ---------------------------
    !open(40,file=pname, status='unknown', position='rewind')
    !write(40,30) 'GB dislocation dynamics - version 2013-11-19'
    !write(40,30) 'fully boundary conditions for flux, B and elastic field'
    !write(40,10) 'shear modulus (MPa)', mu
    !write(40,10) 'Poisson ratio', nu
    !write(40,10) 'Temperature (K)', temp
    !write(40,10) 'Diffusion coeff D0 (micron^2/micro sec)', D0
    !write(40,10) 'Diffusion constant DC', DC
    !write(40,10) 'dislocation mobility B/(MPa micro sec)', M
    !write(40,10) 'strain rate (1/micro sec)', edtot
    !write(40,10) 'grain size (micron)', D2
    !write(40,10) 'distance of glide planes (micron)', Dgp
    !write(40,10) 'Burgers vector norm (micron)', B
    !write(40,10) 'GB cell size (micron)', gbdx
    !write(40,20) 'number GB cells', Ngbn
    !write(40,30) 'it, ttot (s), dt, gdpl (1/s), dgpl_av (1/s), gplast, tau0, bdmax, vmax, v_av, Npu, Nabs, sum(bfield)'

    !10 format('# ',A40,G14.5)
    !20 format('# ',A40,I8)
    !30 format('# ',A100)

    !iteration loop 
    !do while (((bdmax>1.e-13).or.(gdpl>1.e-12)).and.(it<=niter)) 
    !do while ((gplast<epsmax).and.(it<=niter))
    do while ((it<=niter).and.(ttot < tfin))
    !   ! calculate current stress!
    !   hh   = ttot*edtot
    !   tau0 = mu*(hh-gplast)
        
        !dislocation nucleation criterion 
        !hh = 0.d0
        !do i=1,Npu
        !    !print*, "START1", it, i, ypu(i), Npu
        !    if (ypu(i) > hh) hh=ypu(i)
        !end do
        if (abs(tau0) < 1.d-6) then
            hh = D2  ! avoid dislocation nucleation on slip plane
        else
            hh = maxval(ypu(:Npu))
        end if
        if (hh < dsrc) then
            Npu = Npu + 1
            ypu(Npu) = D2
            call tau_GB() ! calculate force on dislocations
            if (fdis(Npu) > -fcrit) Npu = Npu - 1 !nucleation failed if test dislocation is pushed out
        end if 
        !print*, "START2", it, Npu
        if (Npu > maxdis) then
            write(*,*) 'Npu greater maxdis'
            stop
        end if
        Npu_max = max(Npu_max, Npu)

        ! move dislocations 
        if (Npu > 0) then
            call tau_GB() ! calculate force on dislocations
            vmax = 0.d0
            yrk = ypu
            frk = fdis
            !print*, 'MOVE1', it, Npu, ypu(:Npu), dt, M, fdis(:Npu)
            ypu = ypu + 0.8d0*M*dt*fdis
            !print*, 'MOVE2', it, Npu, ypu(:Npu), dt, M, fdis(:Npu)
            call tau_GB()
            fdis = 0.5d0*(fdis + frk)
            ypu = yrk
            do i=1,Npu
                hh = M*fdis(i)
                vdis(i) = hh
                if (abs(hh).gt.vmax) vmax = abs(hh)
                !print*, "VDIS1", it, i, fdis(i), vdis(i), Npu
            end do
                
            ! sim_time step control
            hh = dtmax*min(cdist/(dt*abs(vmax)), 1.e-12/bdmax, 1.d0)
            hh = max(1.d-5, hh)
            if (hh < 0.5d0*dt) then
                dt = hh
            else
                dt = 0.2d0*hh + 0.8d0*dt
            end if
            hh = 0.d0
            yrk = ypu
            do i=1,Npu
                !print *, "VDIS", it, i, ypu(i), vdis(i)
                ypu(i) = ypu(i) + vdis(i)*dt
                hh = hh + ypu(i)
            end do
            hh = (Npu*D2 - hh + D2*Nabs)*B/(D2*Dgp)
            gdpl = (hh-gplast)/dt
            gplast = hh
        else
            hh = dtmax*min(1.e-11/bdmax, 1.d0)
            if (hh < 0.5d0*dt) then
                dt = hh
            else
                dt = 0.2d0*hh + 0.8d0*dt
            end if
            gdpl = 0.
        end if 
        ttot = ttot + dt

        ! dislocation absorption in GB if first dislocation is close 
        hh = D2
        do i=1,Npu
            if (ypu(i) < hh) then
                hh = ypu(i)
                ih = i
            end if
        end do
        if (hh < cdist) then
            do i=ih,Npu
                ypu(i) = ypu(i+1)
            end do
            Npu = Npu-1
            Nabs = Nabs+1
            bfield(Nc) = bfield(Nc) + B
        end if 

        ! calculate diffusion flux in GB 
        do i=1,Ngbn 
            hh1 = 0.
            hh2 = 0.
            hx2 = (i-Nc)*gbdx
            hx2 = hx2*hx2
            ! contribution of GB elastic field: U(x_i, x_j)
            do k=1,Ngbn
               hh1 = hh1 + bfield(k)*Upot(i,k)  ! Upot(i,i) = 0
!                hh1 = hh1 + bfield(k)*log(sin(pi*(k-i)/Ngbn))
            end do
            ! contribution of pile-up/GB dis interaction: V(x_i, y_j)
            do k=1,Npu
                hy2 = ypu(k)
                hy2 = hy2*hy2 
                !print *, "HERE",it, k, ypu(k), hy2
                !hh2 = hh2 + hx2/(hx2 + hy2) + log(sqrt(hx2+hy2)/B); ! old version, v1.0.0
                hh = sqrt(hx2+hy2)/D2
                hc = hy2 / (hx2 + hy2)
                hh2 = hh2 + log(hh) * sqrt(1.d0 + 4.d0*hc - 4.d0*hc*hc)  ! new version, v1.1.0
            end do
            dGdb(i) = mu*bfield(i) - df*C*hh1 - C*B*hh2  ! mu*bfield(i) ! new in v2.1.2
        end do
        if ((mod(it,1000)==0).and.(dt>dtmax*0.9)) then
            df = min(df*(1.d0+dtmax*1.d-7), 1.d0)  ! gradually increase damping factor to 1 (no damping)
            !print*,mu*bfield(Nc), df*C*hh1, dt, df
        end if
        do i=2,Ngbn-1
            Jf(i) = - DC*( 2.d0*dGdb(i) - dGdb(i-1) - dGdb(i+1) )
        end do ! loop i
        
        ! ** periodic flux BC **
        ! Jf(1)    = -DC*( 2.d0*dGdb(1)     - dGdb(2)       - dGdb(Ngbn) )
        ! Jf(Ngbn) = -DC*( 2.d0*dGdb(Ngbn) - dGdb(Ngbn-1) - dGdb(1) )

        ! ** flux BC for open boundary **
        ! Jf(1) = -DC*(dGdb(1) - dGdb(2))
        ! Jf(Ngbn) = -DC*(dGdb(Ngbn) - dGdb(Ngbn-1))

        ! ** no flux BC, anti-symmetric bfield **
        Jf(1) = 0.d0
        Jf(Ngbn) = 0.d0

        ! calculate GB Burgers vectors from flux rate 
        bdmax = 0.d0
        do i=2,Ngbn-1
            hh = omdx*(2.d0*Jf(i) - Jf(i-1) - Jf(i+1))  ! new in v2.1.2
            bfield(i) = bfield(i) + hh*dt
            bdot(i) = hh
            if (bdmax < abs(hh)) bdmax=abs(hh)
        end do
        ! ** periodic b-field **
        !hh = (2.d0*Jf(1) - Jf(2) - Jf(Ngbn))*omdx
        !bfield(1) = bfield(1) + hh*dt
        !bdot(1) = hh;
        !if (bdmax < abs(hh)) bdmax=abs(hh)
        !hh = (2.d0*Jf(Ngbn) - Jf(Ngbn-1) - Jf(1))*omdx
        !bfield(Ngbn) = bfield(Ngbn) + hh*dt
        !bdot(Ngbn) = hh;
        !if (bdmax < abs(hh)) bdmax=abs(hh)

        ! ** open or anti-periodic GB **
        bfield(1) = 0.d0
        bfield(Ngbn) = 0.d0
        bdot(1) = 0.d0
        bdot(Ngbn) = 0.d0

        ! Write protocol output for sim_time series
        if ((screen_out).and.(mod(it,nwr)==0)) then
            ih = mod(it/nwr,nsav) + 1
            ts(ih)  = ttot
            gps(ih) = gplast
            if (ih==nsav) ih = 0
            hh1 = (gplast-gps(ih+1))/(1.d-6*(ttot-ts(ih+1)))
            if (Npu > 0) then
                hh2 = sum(vdis)/dble(Npu)
            else
                hh2 = 0.d0
            end if
            hh = sum(bfield)/B
            !write(40,500) it, ttot*1.d-6, dt, gdpl*1.d6, hh1, gplast, tau0, &
            !    bdmax, vmax, hh2, Npu, Nabs, hh
            write(*,*) "Iteration, dt, ttot(s), max(bdot), sum(bfield), Npu, Nabs, max(y)", &
                    it, dt, ttot*1.d-6, bdmax, sum(bfield), Npu, Nabs, maxval(ypu(:Npu))
            !do i=1,ih5
            !    write(*,*) pu_out(i, 1, :Npu)
            !end do
        end if
        
        ! collect sim_time series output for GB nodes
        if (ttot>=twr) then
            ! collect gb data
            if (iwr>maxout) then
                write(*,*) 'iwr greater maxout, last sim_time step will be overwritten'
                ih5 = maxout
            else
                ih5 = iwr
            end if
            time_out(ih5) = ttot

            hh = 0.d0
            do i=1,Ngbn
                vout(ih5,1,i) = Jf(i)
                vout(ih5,2,i) = dGdb(i)
                vout(ih5,3,i) = bdot(i)/B
                vout(ih5,4,i) = bfield(i)/B
                vout(ih5,5,i) = hh
                hh = hh + bfield(i)/B
            end do
            iwr = iwr + 1
            twr = twr + twr0 !*iwr*iwr
            ! collect dislocation positions
            pu_out(ih5, 1, :Npu) = ypu(:Npu)
            pu_out(ih5, 2, :Npu) = fdis(:Npu)
            pu_out(ih5, 3, :Npu) = vdis(:Npu)
            ! collect global sim_time series data
            ih = mod(it/nwr,nsav) + 1
            ts(ih)  = ttot
            gps(ih) = gplast
            if (ih==nsav) ih = 0
            hh1 = (gplast-gps(ih+1))/((ttot-ts(ih+1)))
            if (Npu > 0) then
                hh2 = sum(vdis)/dble(Npu)
            else
                hh2 = 0.d0
            end if
            hh = sum(bfield)/B
            globout(ih5, 1) = it
            globout(ih5, 2) = dt
            globout(ih5, 3) = gdpl
            globout(ih5, 4) = hh1
            globout(ih5, 5) = gplast
            globout(ih5, 6) = tau0
            globout(ih5, 7) = bdmax
            globout(ih5, 8) = vmax
            globout(ih5, 9) = hh2
            globout(ih5, 10) = Npu
            globout(ih5, 11) = Nabs
            globout(ih5, 12) = hh
        end if 

        it = it + 1
    end do  ! iteration loop

    !write final output 
    if (iwr<maxout) then
        ih5 = iwr
        time_out(ih5) = ttot
        hh = 0.d0
        do i=1,Ngbn 
            vout(ih5,1,i) = Jf(i)
            vout(ih5,2,i) = dGdb(i)
            vout(ih5,3,i) = bdot(i)/B
            vout(ih5,4,i) = bfield(i)/B
            vout(ih5,5,i) = hh
            hh = hh + bfield(i)/B
        end do
        ! collect global sim_time series data
        ih = mod(it/nwr,nsav) + 1
        ts(ih)  = ttot
        gps(ih) = gplast
        if (ih==nsav) ih = 0
        hh1 = (gplast-gps(ih+1))/((ttot-ts(ih+1)))
        if (Npu > 0) then
            hh2 = sum(vdis)/dble(Npu)
        else
            hh2 = 0.d0
        end if
        hh = sum(bfield)/B
        globout(ih5, 1) = it
        globout(ih5, 2) = dt
        globout(ih5, 3) = gdpl
        globout(ih5, 4) = hh1
        globout(ih5, 5) = gplast
        globout(ih5, 6) = tau0
        globout(ih5, 7) = bdmax
        globout(ih5, 8) = vmax
        globout(ih5, 9) = hh2
        globout(ih5, 10) = Npu
        globout(ih5, 11) = Nabs
        globout(ih5, 12) = hh
    end if

    !close(40)  ! close protocol file

    500 format(I8, 4G14.5, G18.9,4G14.5, 2I5, G15.5)


contains
    subroutine init()
        implicit none
        integer :: i, k

        ! initialize fields
        time_out = 0.d0
        pu_out = 0.d0
        globout = 0.d0
        vout = 0.d0
        ts = 0.d0
        gps = 0.d0
        vdis = 0.d0
        fdis = 0.d0
        bdmax = 0.d0
        vmax = 0.d0
        dgdb = 0.d0
        gdpl = 1.d0
        gplast = 0.d0
        ttot = 0.d0
        Jf = 0.d0
        bdot = 0.d0
        ypu = 0.d0
        bfield = 0.d0

        ! material parameters
        mu    = params(IDX_MU)  !mu = 44.d3 ! (MPa)
        nu    = params(IDX_NU)  !nu = 0.3
        B     = params(IDX_B)   !B = 0.25d-3 !bulk Burgers vector norm (micron)
        delta = params(IDX_DELTA)  !delta = 5.d-4 ! GB thickness (micron)
        Qact  = params(IDX_QACT)  !Qact = 57.d3 ! activation energy for GB diffusion (J/mol)
        drag  = params(IDX_DRAG)  ! 500.d0
        Dif_gb = params(IDX_DIFGB)  !Dif_gb = 1.d1 ! GB diffusion coeff (micron^2/micro s)
        fcrit = params(IDX_FCRIT)  ! critical force for dislocation nucleation
        pi = 4.d0*datan(1.d0)
        C = mu/(2*pi*(1-nu)) ! is A in paper
        M = B/drag ! dislocation mobility B/(microsecond.MPa)
        R = 8.31446d0 ! gas constant (J/molK)
        D0 = Dif_gb*exp(-Qact/(R*temp)) ! GB diffusion coefficient
        Omega = B*B !atomic volume
        dsrc = D2 - 100*B ! 0.9*D2  ! distance require for dislocation source

        ! set numerical parameters
        twr0 = tfin / maxout ! for cubic time scale: apply maxout**2
        dt = dtmax*1.d-4  ! 0.05 ! initial time step (microseconds)
        df = 0.5  ! damping factor for contribution of GB to dgdb; df=1: no damping
        nwr = niter / 100
        eps = 1.d-20
        epsmax = 1.5d-2
        edtot  = 1.d-9 ! (1/micro sec)
        cdist = 10.d0*B
        twr = twr0    ! first write sim_time for sim_time series output
        Nc = (Ngbn+1)/2 !center of GB
        gbdx = Dgp/(Ngbn-1) ! size of GB elements
        gbdxD2 = gbdx/D2
        omdx = Omega/gbdx
        DC = D0*delta/(R*temp*gbdx*gbdx) !D delta/ (RT gbdx**2)
        Upot = 0.d0
        do i=1,Ngbn
            xout(i) = (i-Nc)*gbdx
            do k=1, i-1
                Upot(i, k) = log((i-k)*gbdxD2)
            end do
            do k=i+1, Ngbn
                Upot(i, k) = log((k-i)*gbdxD2)
            end do
        end do
        Nabs = 0 ! number of dislocations absorbed in GB
        Npu = 0 ! number of dislocations in pile up (on slip plane)
        !ypu(1) = 0.9*D2
        Npu_max = Npu

    end subroutine init
    !===================================

    !===================================
    subroutine tau_GB()
        implicit none 

        real(8) :: hh, hx, hx2, hy, hy2, hc, hr2, hsc, hfr
        integer :: i, j
        ! 
        ! calculate Peach-Kohler force on dislocations on slip plane 
        ! input: bfield and ypu, output: fdis

        do j=1,Npu 
            hh = 0.d0
            hy = ypu(j)
            hy2 = hy*hy
            do i=1,Npu
                if (j==i) cycle
                hh = hh + B/(hy-ypu(i))
            end do
            do i=1,Ngbn 
                hx = abs(i-Nc)*gbdx
                !hx2 = hx*hx
                !hc = hy2/(hx2 + hy2)   ! new formulation in v1.1.0
                !hr2 = hx2 + hy2
                !!print*, 'TAU', j, i, hx, hy, hc, hr2
                !hsc = 1.d0 + 4.d0*hc - 4.d0*hc*hc
                !hfr = 4.d0*hx2*hy*(1.d0-2.d0*hc) / (hr2*hr2*hsc)
                !hh = hh + bfield(i)*sqrt(hsc)*(hfr*log(sqrt(hr2)/D2) + hy/hr2)
                hh = hh + bfield(i)/(hy + hx)
            end do 
            fdis(j) = (C*hh - tau0)*B
        end do
    end subroutine tau_GB

end subroutine calc_gbdd
