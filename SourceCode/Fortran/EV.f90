subroutine ev(cash,nrow,ncol,v,nco,prob,n,fy,eyp,grid,secondd,ret,reg_coef,ev_out)
INTEGER, INTENT(IN) :: nrow, ncol, nco, n
REAL, INTENT(IN) :: ret
REAL, INTENT(IN) :: reg_coef
REAL, INTENT(IN) :: cash(nrow,ncol),v(nco,1),prob(n,1)
REAL, INTENT(IN) ::  fy(n,1),eyp(n,1),grid(nco,1),secondd(nco,1)
REAL, INTENT(OUT) :: ev_out(nrow,ncol)
INTEGER :: ind1, ind2
REAL :: inc
REAL, allocatable :: ones(:,:), inc2(:,:), v1(:,:), aux(:,:)
ALLOCATE(ones(nrow,ncol),inc2(nrow,ncol),v1(nrow,ncol),aux(nrow,ncol))
ev_out = 0.0
ones= 1.0
!$$$$$$ print *, 'The first argument in EV.f90 is', cash
!$$$$$$ print *, 'The second argument in EV.f90 is', nrow
!$$$$$$ print *, 'The third argument in EV.f90 is', ncol
!$$$$$$ print *, 'The fourth argument in EV.f90 is', v
!$$$$$$ print *, 'The fifth argument in EV.f90 is', nco
!$$$$$$ print *, 'The sixth argument in EV.f90 is', prob
!$$$$$$ print *, 'The seventh argument in EV.f90 is', n
!$$$$$$ print *, 'The eighth argument in EV.f90 is', fy
!$$$$$$ print *, 'The ninth argument in EV.f90 is', eyp
!$$$$$$ print *, 'The tenth argument in EV.f90 is',grid
!$$$$$$ print *, 'The eleventh argument in EV.f90 is',secondd
!$$$$$$ print *, 'The twelth argument in EV.f90 is',ret
!$$$$$$ print *, 'The thirteenth argument in EV.f90 is',reg_coef
!$$$$$$ print *, 'The fourteenth argument in EV.f90 is',ev_out
do ind1=1,n
  do ind2=1,n
     inc=(fy(ind1,1)*(eyp(ind2,1)+reg_coef*ret))
     inc2 = inc*ones+cash
     inc2=MIN(inc2,grid(nco,1))
     inc2=MAX(inc2,grid(1,1))
     call splint(grid,v(:,1),secondd(:,1),nco,inc2,nrow,ncol,aux)
     v1=prob(ind1,1)*prob(ind2,1)*aux
     ev_out=ev_out+v1
  end do
end do
DEALLOCATE(ones,inc2,v1,aux)
end subroutine

