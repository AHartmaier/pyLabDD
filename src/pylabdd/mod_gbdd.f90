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
! 2026-05-07: v2.2.0: Modified pile-up GB interaction terms
! 2026-05-09: v2.2.1: Automatic selection and truncation of output times
! 2026-05-12: v2.2.2: Introduced constant Z for PU-GB interaction normalization (instead of D2)
!
! Author: Alexander Hartmaier
! Institution: Ruhr-Universitaet Bochum, ICAMS
! Copyright (c) 2013-2026 by the Author. All rights reserved.
! This code can be used under the terms of the GNU General Public License version 3 (GNU GPL-3.0)

subroutine calc_gbdd(params, nparams, tau0, temp, Dgp, D2, Ngbn, maxdis, tfin, niter, dtmax, &
    it, Npu_max, nout, Nabs, time, xpos, vout, pu_out, globout, screen_out)

    implicit none

    integer, parameter :: name_len = 16
    integer, parameter :: IDX_MU     = 1
    integer, parameter :: IDX_NU     = 2
    integer, parameter :: IDX_B      = 3
    integer, parameter :: IDX_DELTA  = 4
    integer, parameter :: IDX_QACT   = 5
    integer, parameter :: IDX_DRAG   = 6
    integer, parameter :: IDX_DIFGB  = 7
    integer, parameter :: IDX_FCRIT  = 8
    integer, parameter :: N_GBDD_PARAMS = 8

    logical, intent(in) :: screen_out  ! write output on screen
    integer, intent(in) :: nparams  ! number of constitutive parameters
    integer, intent(in) :: Ngbn    ! number of gain boundary nodes, should be odd to have a center node
    integer, intent(in) :: niter   ! maximum number of iteration steps
    integer, intent(in) :: maxdis  ! maximum number of dislocations in pile-up
    integer, intent(out) :: Npu_max ! maximum number of dislocation in pile-up reached
    integer, intent(out) :: it !number of iterations
    integer, intent(out) :: nout  ! number of outputs
    integer, intent(out) :: Nabs ! number of absorbed dislocations
    real(8), intent(in) :: params(nparams)  ! Vector for constitutive parameters
    real(8), intent(in) :: tfin  ! final sim_time for simulation (microseconds), std: 25d6
    real(8), intent(in) :: dtmax ! 1.d3 w/o pu; 60 with pile up
    real(8), intent(in) :: D2    ! Grain size D/2, distance FR source-GB (micron)
    real(8), intent(in) :: Dgp   ! Length of GB segment considered (micron); size 0.005
    real(8), intent(in) :: tau0  ! applied shear stress (MPa) 
    real(8), intent(in) :: temp  ! temperature
    real(8), intent(out) :: vout(:, :, :), time(:), xpos(:)
    real(8), intent(out) :: pu_out(:, :, :), globout(:, :)
    real(8) :: bfield(Ngbn), ypu(maxdis), fdis(maxdis), vdis(maxdis)
    real(8) :: M, C, DC, B, mu, nu, Omega, pi, gbdx, drag
    real(8) :: R, delta, Dif_gb, D0, Qact, df, fcrit
    real(8) :: frk(maxdis), yrk(maxdis), Jf(Ngbn), bdot(Ngbn)
    real(8) :: dGdb(Ngbn), Upot(Ngbn, Ngbn)
    real(8) :: dt, ttot, eps
    real(8) :: gdpl, gplast, vmax, bdmax
    real(8) :: epsmax, edtot, cdist, dsrc
    real(8) :: twr(1000), dtwr  ! sim_time interval (microseconds) for sim_time series output, std: 20d6
    real(8) :: hh, hh1, hh2, hx2, hy2, hr2, hxl2, omdx
    integer :: Nc, Npu
    integer :: nfields, nglob, maxout, nwr
    integer :: i, k, ih, j

    maxout = size(time)
    nfields = size(vout, 2)
    nglob = size(globout, 2)
    
    if (maxout > 1000) then
        error stop "maxout should be smaller than 1000, decrease value or increase twr-array"
    end if
    if (nparams < N_GBDD_PARAMS) then
        error stop "GBDD parameter vector is too short."
    end if
    if (nparams > N_GBDD_PARAMS) then
        error stop "GBDD parameter vector is too long."
    end if
    call init()

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
    nout = 2  ! points to next output position
    it = 1  ! iteration counter
    if (screen_out) write(*,*) "Starting iteration for T, tau0, D2, Lgb, Ngbn", temp, tau0, D2, Dgp, Ngbn

    !  === iteration loop ===
    !do while (((bdmax>1.e-13).or.(gdpl>1.e-12)).and.(it<=niter)) 
    !do while ((gplast<epsmax).and.(it<=niter))
    do while ((it<=niter).and.(ttot < tfin))
        !! calculate current stress
        !hh   = ttot*edtot
        !tau0 = mu*(hh-gplast)
        
        ! dislocation nucleation criterion
        if (abs(tau0) < 1.d-6) then
            hh = D2  ! avoid dislocation nucleation on slip plane
        else
            hh = maxval(ypu(:Npu))
        end if
        if (hh < dsrc) then
            Npu = Npu + 1
            ypu(Npu) = D2
            call tau_GB()  ! calculate force on dislocations
            if (fdis(Npu) > -fcrit) Npu = Npu - 1  ! nucleation failed if test dislocation is pushed out
        end if 
        if (Npu > maxdis) then
            write(*,*) 'Npu greater maxdis'
            stop
        end if
        Npu_max = max(Npu_max, Npu)

        ! move dislocations 
        if (Npu > 0) then
            vmax = 0.d0
            yrk = ypu
            frk = fdis
            ypu = ypu + 0.8d0*M*dt*fdis
            call tau_GB()  ! calculates fdis
            fdis = 0.5d0*(fdis + frk)
            ypu = yrk
            do i=1,Npu
                hh = M*fdis(i)
                vdis(i) = hh
                if (abs(hh) > vmax) vmax = abs(hh)
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
                ypu(i) = ypu(i) + vdis(i)*dt
                hh = hh + ypu(i)
            end do
            call tau_GB()  ! update fdis with new positions
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
            hx2 = xpos(i)*xpos(i)
            ! contribution of GB elastic field: U(x_i, x_j)
            do k=1,Ngbn
               hh1 = hh1 + bfield(k)*Upot(i,k)  ! Upot(i,i) = 0
            end do
            ! contribution of pile-up/GB dis interaction: V(x_i, y_j)
            do k=1,Npu
                hy2 = ypu(k)
                hy2 = hy2*hy2
                hr2 = hx2 + hy2
                hxl2 = hx2 + 4.d0 ! D2*D2
                hh2 = hh2 + 0.5*log(hr2/hxl2) + hx2/hr2 - hx2/hxl2  ! new version, v2.2.0
            end do
            dGdb(i) = mu*bfield(i) - df*C*hh1 - C*B*hh2  ! new in v2.1.2
        end do
        if ((mod(it,1000)==0).and.(dt>dtmax*0.9)) then
            df = min(df*(1.d0+dtmax*1.d-7), 1.d0)  ! gradually increase damping factor to 1 (no damping)
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

        ! write protocol output to standard device if requested
        if ((screen_out).and.(mod(it,nwr)==0)) call print_out()
        
        ! collect sim_time series output for GB nodes
        if (ttot >= twr(nout)) then
            call store_data()
            nout = nout + 1
            if (nout > maxout) then
                write(*,*) 'nout > maxout, last sim_time step will be overwritten'
                nout = maxout
                stop
            end if
        end if 

        it = it + 1
    end do  ! iteration loop

    !write final output 
    if (nout <= maxout) call store_data()

contains
    subroutine init()
        implicit none
        real(8) :: hh, hf
        integer :: i, k

        ! initialize fields
        time = 0.d0
        pu_out = 0.d0
        globout = 0.d0
        vout = 0.d0
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
        Nabs = 0 ! number of dislocations absorbed in GB
        Npu = 0 ! number of dislocations in pile up (on slip plane)
        Npu_max = Npu

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
        dsrc = D2 - 100*B  ! distance required for dislocation source

        ! write truncated output times into array; new in v2.2.1
        dtwr = tfin / (maxout-2)
        if (abs(tau0) < 1.d-6) dtwr = dtwr / maxout  ! decressive output frequency for GB diffusion w/o PU
        twr(1) = 0.d0
        do i=2, maxout
            hh = (i-1) * dtwr ! constant output frequency
            if (abs(tau0) < 1.d-6) hh = hh*i  ! decressive output frequency
            k = int(log(hh) / log(10.d0))
            if (k > 8) then
                k = k - 4  ! round to 5 leading digits if > 1e9
            elseif (k > 5) then
                k = k - 3  ! round to 4 leading digits if > 1e6
            elseif (k > 3) then
                k = k - 2  ! round to 3 leading digits if > 1e4
            elseif (k > 2) then
                k = k - 1  ! round to 2 leading digits if > 1e3
            end if
            hf = 10.d0**k
            twr(i) = int(hh / hf) * hf
        end do

        ! set numerical parameters
        dt = dtmax*1.d-4  ! initial time step (microseconds)
        df = 0.5  ! damping factor for contribution of GB to dgdb; df=1: no damping
        nwr = niter / 100
        eps = 1.d-20
        epsmax = 1.5d-2
        edtot  = 1.d-9 ! (1/micro sec)
        cdist = 10.d0*B
        Nc = (Ngbn+1)/2 !center of GB
        gbdx = Dgp/(Ngbn-1) ! size of GB elements
        omdx = Omega/gbdx
        DC = D0*delta/(R*temp*gbdx*gbdx) !D delta/ (RT gbdx**2)
        Upot = 0.d0
        do i=1,Ngbn
            xpos(i) = (i-Nc)*gbdx
            do k=1, i-1
                Upot(i, k) = log((i-k)*gbdx/2.d0) ! 2.d0) !/D2)
            end do
            do k=i+1, Ngbn
                Upot(i, k) = log((k-i)*gbdx/2.d0) !2.d0) !/D2)
            end do
        end do

    end subroutine init
    !===================================

    !===================================
    subroutine tau_GB()
        implicit none 

        real(8) :: hh, hx2, hy, hy2, hr2
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
                hx2 = xpos(i)
                hx2 = hx2*hx2
                hr2 = hx2 + hy2
                hh = hh + bfield(i)*hy*(hy2 - hx2)/(hr2*hr2)  ! new in v2.2.0
            end do 
            fdis(j) = (C*hh - tau0)*B
        end do
    end subroutine tau_GB
    !===================================

    !===================================
    subroutine store_data()
        ! collect gb data
        if (ttot-time(nout-1) < 1.d-6) return
        time(nout) = ttot
        hh = 0.d0
        do i=1,Ngbn
            vout(nout,1,i) = Jf(i)
            vout(nout,2,i) = dGdb(i)
            vout(nout,3,i) = bdot(i)/B
            vout(nout,4,i) = bfield(i)/B
            vout(nout,5,i) = hh
            hh = hh + bfield(i)/B
        end do
        ! collect dislocation positions
        pu_out(nout, 1, :Npu) = ypu(:Npu)
        pu_out(nout, 2, :Npu) = fdis(:Npu)
        pu_out(nout, 3, :Npu) = vdis(:Npu)
        ! collect global sim_time series data
        hh1 = (gplast-globout(nout-1, 5)) / (ttot-time(nout-1))
        if (Npu > 0) then
            hh2 = sum(vdis)/dble(Npu)
        else
            hh2 = 0.d0
        end if
        globout(nout, 1) = it
        globout(nout, 2) = dt
        globout(nout, 3) = gdpl
        globout(nout, 4) = hh1
        globout(nout, 5) = gplast
        globout(nout, 6) = tau0
        globout(nout, 7) = bdmax
        globout(nout, 8) = vmax
        globout(nout, 9) = hh2
        globout(nout, 10) = Npu
        globout(nout, 11) = Nabs
        globout(nout, 12) = sum(bfield)/B
    end subroutine store_data
    !===================================

    subroutine print_out()
        write(*,*) "Iteration, dt, ttot(s), max(bdot), sum(bfield), Npu, Nabs, max(y)", &
                it, dt, ttot*1.d-6, bdmax, sum(bfield), Npu, Nabs, maxval(ypu(:Npu))
    end subroutine print_out

end subroutine calc_gbdd
