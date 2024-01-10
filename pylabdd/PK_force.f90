! Fortran90 subroutine to be used in Python
! calculate Peach-Koehler force on dislocation configuration
! will be embedded via the fmodpy wrapper

subroutine calc_fpk_pbc(xpos, ypos, bx, by, tau0, len_x, len_y, FPK, Nmob, N)
! Solution based on Eqs (2.1.25a) and (2.1.25b) from Linyong Pang "A new O(N) method for
! modeling and simulating the behavior of a large number of dislocations in
! anisotropic linear elastic media", PhD thesis, Stanford University, USA. 2001 
implicit none
integer, intent(in) :: N
integer, intent(in) :: Nmob
double precision, intent(in), dimension(N) :: xpos, ypos, bx, by
double precision, intent(out), dimension(2,Nmob) :: FPK
double precision, intent(in) :: tau0, len_x, len_y

integer :: i,j, m
double precision ::pih,pih2,pi
double complex ::z,hcot,hcre, imunit
double complex ::pxx1,pxx2,pxx3,pyy1
double precision ::h11,h22,h12, px, py
double precision :: hbx, hby, hx, hy

pi = 4.d0*datan(1.d0)
pih = pi/len_x
pih2=pih*pih
imunit = (0.d0, 1.d0)
do j=1, Nmob
   h11=0.d0
   h22=0.d0
   h12=tau0
   px = xpos(j)
   py = ypos(j)
   do i=1, N
     if (j==i) cycle
     hbx = bx(i)
     hby = by(i)
     hx = xpos(i)
     hy = ypos(i)
     do m=0,3
        z = px-hx + (py-(hy+dble(m)*len_y))*imunit
        z = z*pih
        hcre=1.d0/sin(z)
        hcot = cos(z)*hcre*pih
        hcre=hcre*hcre
        pxx1=2.d0*(hby - 2.d0*hbx*imunit)
        pxx2=hbx + hby*imunit
        pxx3=hbx*imunit
        pyy1=pxx2*2.d0*(py-(hy+dble(m)*len_y))*pih2*hcre
        h11=h11+dble(pxx1*hcot)
        h11=h11-dble(pyy1)
        h22=h22+dble(2.d0*hby*hcot)
        h22=h22+dble(pyy1)
        h12=h12+dimag(pxx3*2.d0*hcot)
        h12=h12+dimag(pyy1)
        if(m>0)then
           z = px-hx + (py-hy+dble(m)*len_y)*imunit
           z = z*pih
           hcre=1.d0/sin(z)
           hcot = cos(z)*hcre*pih
           hcre=hcre*hcre
           pxx1=2.d0*(hby - 2.d0*hbx*imunit)
           pxx2=hbx + hby*imunit
           pxx3=hbx*imunit
           pyy1=pxx2*2.d0*(py-hy+dble(m)*len_y)*pih2*hcre
           h11=h11+dble(pxx1*hcot)
           h11=h11-dble(pyy1)
           h22=h22+dble(2.d0*hby*hcot)
           h22=h22+dble(pyy1)
           h12=h12+dimag(pxx3*2.d0*hcot)
           h12=h12+dimag(pyy1)
        end if
     end do   !loop over m
   end do  ! loop over i
   FPK(1,j) = 0.5*(h12*bx(j) + h22*by(j))
   FPK(2,j) = -0.5*(h11*bx(j) + h12*by(j))
end do   ! loop over j
end subroutine calc_fpk_pbc

subroutine calc_fpk(xpos, ypos, bx, by, tau0, FPK, Nmob, N)
implicit none
integer, intent(in) :: N
integer, intent(in) :: Nmob
double precision, intent(in), dimension(N) :: xpos, ypos, bx, by
double precision, intent(out), dimension(2,Nmob) :: FPK
double precision, intent(in) :: tau0

integer :: i,j
double precision :: h11, h22, h12
double precision :: xpi, ypi, x, y
double precision :: hx, hy, hh, hbx, hby

do i=1, Nmob
    xpi = xpos(i)
    ypi = ypos(i)
    h11 = 0.d0
    h22 = 0.d0
    h12 = tau0
    do j=1, N
        if (j==i) cycle
        x = xpi-xpos(j)
        y = ypi-ypos(j)
        hx = x*x
        hy = y*y
        hh = hx + hy
        hh = hh*hh
        hbx = bx(j)
        hby = by(j)
        h11 = h11 - hbx*y*(3.d0*hx + hy)/hh
        h11 = h11 + hby*x*(hy - hx)/hh
        h22 = h22 + hbx*y*(hx - hy)/hh
        h22 = h22 - hby*x*(3.d0*hy + hx)/hh
        h12 = h12 + hbx*x*(hx - hy)/hh
        h12 = h12 + hby*y*(hy - hx)/hh
    end do
    FPK(1,i) = h12*bx(i) + h22*by(i)
    FPK(2,i) = -(h11*bx(i) + h12*by(i))
end do
end subroutine calc_fpk
