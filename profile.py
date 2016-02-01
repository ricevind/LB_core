## -*- coding: utf-8 -*-
#"""
#Created on Wed Jan 13 21:48:51 2016
#
#@author: ricevind
##"""
##
import os
from numpy import linspace, sqrt
import matplotlib.pylab as plt
def save_profile(path, name, ny, nx,  u, uLB, rho):
    plt.clf()
    x = linspace(0,ny,len(u[1,:,0]))
    y = sqrt(u[0]**2+u[1]**2)[:,-1]/uLB
    ff = lambda r: 1.5*(1-(r/(ny/2))**2)
    xx = linspace(0,ny/2,25)
    yy = ff(xx)
    plt.subplot(2,1,1)
    plt.plot(x,y)
    plt.plot(xx+ny/2, yy,'o')
    plt.plot(xx, yy[::-1],'o')
    
    plt.subplot(2,1,2)
    xxx = linspace(0,nx,len(rho[1,1:]))
    rhol = rho[int(ny/2),1:]
    plt.plot(xxx, rhol)
    plt.savefig(os.path.join(path, name+'.png'))
    
    
    
#y = u[0,:,-1]/uLB
#y0 = ud[:,200]

#w = np.ones(len(x))
#w[[0,-1]] = 1000
#z = np.polyfit(x,y,2)
#f = np.poly1d(z)



#
#
##

#plt.plot(x,y0)

#plt.savefig('rura_tau_5525ny90_Re497_poussile_n1_d00038'+".png")
#
####
#xxx = linspace(0,nx,len(rho[1,1:]))
#rhol = rho[int(ny/2),1:]
#
#plt.plot(xxx, rhol)

#po = 400
#xxx = linspace(0,nx,len(rhol[po:]))
#w = np.ones(len(rhol[po:]))
#w[-1] = 1000
#z = np.polyfit(xxx,rhol[po:],1,w=w)
#f = np.poly1d(z)
#
#xx = linspace(po,nx,100)
#yy = f(xx)
#plt.plot(xx, yy,)
