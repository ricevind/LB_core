# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:05:29 2016

@author: ricevind
"""
import matplotlib.pylab as plt
from numpy import sqrt, linspace
from os import path

def plot(rho, u, uLB, tau, rho_history, zdjecia, image, nx, maxIter ):
#    plt.figure(figsize=(15,15))
#    plt.subplot(4, 1, 1)
#    plt.imshow(u[1,:,0:50],vmin=-uLB*.15, vmax=uLB*.15, interpolation='none')#,cmap=cm.seismic
#    plt.colorbar()
    plt.rcParams["figure.figsize"] = (15,15)
    plt.subplot(5, 1, 1)
    plt.imshow(sqrt(u[0]**2+u[1]**2),vmin=0, vmax=uLB*1.6)#,cmap=cm.seismic
    plt.colorbar()
    plt.title('tau = {:f}'.format(tau))      
    
    plt.subplot(5, 1, 2)
    plt.imshow(u[0,:,:30],  interpolation='none')#,cmap=cm.seismicvmax=uLB*1.6,
    plt.colorbar()
    plt.title('tau = {:f}'.format(tau))  
    
    plt.subplot(5, 1, 3)
    plt.imshow(rho, interpolation='none' )#,cmap=cm.seismic
    plt.title('rho')   
    
    plt.subplot(5, 1,4)
    plt.title(' history rho')
    plt.plot(linspace(0,len(rho_history),len(rho_history)),rho_history)
    plt.xlim([0,maxIter])   
    
    plt.subplot(5, 1,5)
    plt.title(' u0 middle develop')
    plt.plot(linspace(0,nx,len(u[0,20,:])), u[1,20,:])
    plt.tight_layout()                  
    
    plt.savefig(path.join(zdjecia,'f{0:06d}.png'.format(image)))
    plt.clf();
        