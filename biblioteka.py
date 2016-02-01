# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:01:43 2016

@author: ricevind
"""

import numpy as np
import numba as nb

#==============================================================================
# Stałe modelu D2Q9
q = 9 #liczba kierunków prędkości dyskretnych

c = np.array([[0,0], [0,1],[-1,0],[0,-1], [1,0], [-1,1], [-1,-1], [1,-1], [1, 1]]) # kierunki prędkości dyskretnych

w = 1./36. * np.ones(q)  # wagi poszczególnych prędkości
w[np.asarray([np.linalg.norm(ci)<1.1 for ci in c])] = 1./9.; w[0] = 4./9. # wagi poszczególnych prędkości

noslip = np.array([c.tolist().index((-c[i]).tolist()) for i in range(q)]) 
i1 = np.where(c.T[1]<0) # Niewiadome na prawej ścianie
i2 = np.where(c.T[1]==0) # rozkłady środkowe poziome
i3 = np.where(c.T[1]>0) # Niewiadome na lewej ścianie
#==============================================================================

#==============================================================================
# Funkcja obliczająca rokładrównowagowy dla danych wielkości makroskopowyh
@nb.jit(nopython=True)
def rownowaga(rho, u, obstacle):
    feq = np.zeros((9,ni,nj))   
    for i in range(ni):
        for j in range(nj):
            if not obstacle[i,j]:
                for k in range(q):
                    sk1 = u[0,i,j]*c[k,0] + u[1,i,j]*c[k,1]
                    sk2 = sk1**2
                    sk3 = u[0,i,j]**2 + u[1,i,j]**2
                    feq[k,i,j] = rho[i,j]*w[k]*(1 + 3*sk1 + 9/2*sk2 -3/2*sk3)
            else:
                feq[:,i,j] = w
    return feq
    
#==============================================================================
#==============================================================================
# Funkcja obliczająca kolizje
@nb.jit(nopython=True)
def kolizja(fin, feq, omega, obstacle):
    fout = fin.copy()
    for i in range(ni):
        for j in range(nj):
            if not obstacle[i,j]:
                for k in range(q):
                    fout[k,i,j] = fin[k,i,j] + omega[i,j]*(feq[k,i,j] - fin[k,i,j])
            else:
                fout[:,i,j] = fin[:,i,j]
    return fout
    
#==============================================================================
#==============================================================================
# Funkcja stream
@nb.jit(nopython=True)
def stream(fin):
    fout = fin.copy()
    for i in range(ni):
        for j in range(nj):
            for k in range(q):
                ic = i - c[k,0]
                jc = j - c[k,1]
                if (ic == -1 or jc == -1)  or (ic == ni or jc == nj):
                    if ic == -1:
                        ic = ni - 1
                    if jc == -1:
                        jc == nj - 1
                    if ic == ni:
                        ic = 0
                    if jc == nj:
                        jc = 0
                fout[k,i,j] = fin[k,ic,jc]
    return fout
    
#==============================================================================
#==============================================================================
# Funkcja makro

@nb.jit( nopython=True)
def makro(fin, obstacle):
    ul = np.zeros((2,ni,nj))
    rho = np.zeros((ni,nj))
    for i in range(ni):
        for j in range(nj):
            for k in range(q):
                rho[i,j] += fin[k,i,j]
                if not obstacle[i,j]:
                    ul[0,i,j] += fin[k,i,j]*c[k,0]
                    ul[1,i,j] += fin[k,i,j]*c[k,1]
            ul[0,i,j] /= rho[i,j] 
            ul[1,i,j] /= rho[i,j]
    return rho, ul
#==============================================================================
#==============================================================================
#Funkca bounceback
@nb.jit(nopython=True)
def bounceback(fin, obstacle):
    fout = fin.copy()
    for i in range(ni):
        for j in range(nj):
            if obstacle[i,j]:
                for k in range(q):
                    fout[k,i,j] = fin[noslip[k],i,j]
    return fout
    

#==============================================================================
#==============================================================================
# Klasa parametrow symulacji
class symulacja():
    cs = 1/np.sqrt(3)
    cs2 = 1/3
    
    def __init__(self, H, LdH, u, ni):
        # wielkosci makroskowpowe
        self.H = H
        self.LdH = LdH
        self.u = u
        self.ni = ni
        # bezwymiarowe wielkosci charakterystyczne
        self.Re = self.u*self.H/self.ni
        self.T0 = self.H/self.u
        self.ud = self.u*self.T0/self.H
        
    def nx_based(self, nx, uLB):
        # wielkosci siatki
        self.nx = np.int(nx)
        self.ny = np.int(self.nx * self.LdH)
        self.uLB = uLB# predkosc siatki odpowaiadajaca 1 ud
        #wielkosci charakterystyczne siatki
        self.dx = 1/self.nx
        self.dy = self.dx # siatka kwadratowa
        self.dt = self.uLB*self.dx/self.ud
        self.niLB = (self.dt/self.dx**2)/self.Re
        self.omega_nx = 1/(3*self.niLB + 0.5)
        self.tau_nx = 1/self.omega_nx
        self.omega = np.ones((self.nx, self.ny))*self.omega_nx
        return self.nx, self.ny, self.omega, self.niLB, self.dt
        
    def physMakro(self, ul, nIter, ny, nx):
        pass
        
    def tau_based(self, tau, csph):
        self.csph = csph # predkosc dzwieku 
        self.niLBtau = (tau - 1/2)*symulacja.cs2
        pass
        
#==============================================================================
if __name__ == '__main__':
    
    #==============================================================================
    # Testy
    ni = 1
    nj = 2
    u = np.ones((2,1,2), dtype='f8')
    u[:,0,0] = np.array([2,4])
    rho = np.ones((ni,nj), dtype='f8')
    obstacle = np.zeros((ni,nj), dtype='i4')
    obstacle[0,1] = 1
    
    feq = rownowaga(rho,u, obstacle)
    feq_values = np.ones((9,1,2), dtype='f8')
    feq_values[:,0,0] = [-29,55,-17,31,-5,-5,115,-17,151]
    
    feq_test = feq_values*rho*w[:,np.newaxis,np.newaxis]
    
    print('test 1, rownowaga:{}'.format(np.array_equal(feq,feq_test)))
    
    ###########################################################################
    ni = 1
    nj = 2
    
    fin = np.ones((2,1,2), dtype='f8')
    fin[0,0,0] = 3; fin[1,0,0] = 20;fin[0,0,1] = 4; fin[1,0,1] = 5;
    feq = np.ones((2,1,2), dtype='f8')
    feq[0,0,0] = 4; feq[1,0,0] = 5;feq[0,0,1] = 10; feq[1,0,1] = 8;
    omega = np.ones((ni,nj), dtype='f8')
    omega[0,0] = 1.5
    omega[0,1] = 2
    obstacle = np.zeros((ni,nj), dtype='i4')
    obstacle[0,1] = 1
    q = 2
    fout_test= np.ones((2,1,2), dtype='f8')
    fout_test[0,0,0] = 4.5; fout_test[1,0,0] = -2.5;fout_test[0,0,1] = 4; fout_test[1,0,1] = 5;
    
    fout = kolizja(fin, feq, omega, obstacle)
    print('test 2, kolizja:{}'.format(np.array_equal(fout,fout_test)))
    
    ###########################################################################
    
    ni =5
    nj =5
    q = 9
    
    fin = np.zeros((9,ni,nj), dtype='f8')
    fin[:,2,2] = 1; fin[:,0,4] = 2; fin[:,2,4] = 3; fin[:,4,4] = 4; fin[:,4,2] = 5;
    fin[:,4,0] = 6; fin[:,2,0] = 7; fin[:,0,0] = 8; fin[:,0,2] = 9; 
    fout_test = fin.copy()
    for k in range(9):
        fout_test[k] = np.roll(np.roll(fin[k],c[k,0],axis=0), c[k,1], axis=1)
    fout = stream(fin)
    print('test 3, stream:{}'.format(np.array_equal(fout,fout_test)))
    
    ############################################################################
    fin = np.random.rand(9,ni,nj)
    fin.dtype ='f8'
    fin[fin==0]=1
    obstacle = np.zeros((ni,nj), dtype='i4')
    obstacle[2,2] = 1
    sumpop = lambda fin: np.sum(fin,axis=0)
    rho_test = sumpop(fin)           # Calculate macroscopic density and velocity.
    u_test = np.dot(c.transpose(), fin.transpose((1,0,2)))/rho_test
    u_test[:,2,2] = 0
    
    rho, u = makro(fin,obstacle)
    print('test 4, makro:{}'.format((np.array_equal(rho,rho_test) and np.array_equal(u,u_test)  )))
    
    ############################################################################
    ni =5
    nj =6
    q = 9
    
    obstacle = np.zeros((ni,nj), dtype='i4')
    obstacle[[0,-1],:] = 1
    obstacle[:,[0,-1]] = 1
    fin = np.zeros((9,ni,nj), dtype='f8')
    fin[[1,2,5,6],0,:] = 1
    fin[[1,2,5,6],:,-1] = 1
    fin[[1,2,5,6],-1,:] = 1
    fin[[1,2,5,6],:,0] = 2
    
    fout_test = np.zeros((9,ni,nj), dtype='f8')
    fout_test[[3,4,7,8],0,:] = 1
    fout_test[[3,4,7,8],:,-1] = 1
    fout_test[[3,4,7,8],-1,:] = 1
    fout_test[[3,4,7,8],:,0] = 2
    
    fout = bounceback(fin,obstacle)
    print('test 5, bounceback:{}'.format(np.array_equal(fout,fout_test)))
    
    ############################################################################
    H = 0.01
    LdH = 10
    u = 0.01
    ni = 0.035*1e-4
    
    ny=40
    uLB = 0.1


     
    tau_test = 0.92
    dx_test = 0.025
    dt_test = 0.0025000000000000005
    
    pousile = symulacja(H,LdH,u,ni)
    pousile.nx_based(ny,uLB)
    tau = pousile.tau_nx
    dx = pousile.dx
    dt = pousile.dt
    
    print('test 6, tau, dx, dt:{} {} {}'.format(tau_test==tau, dx == dx_test, dt == dt_test))
    #==============================================================================
    