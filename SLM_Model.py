# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:11:23 2024

@author: paul rouquette

The aim of this code file is to generate the figures of the paper: 
    'Full comprehensive model of a SLM' by Paul Rouquette and Edoardo Bellone 
    de Grecis

"""

import os
os.chdir("C:\\Users\\prouquette\\Documents\\SLM\\Codes_python\\article")

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})

#%% def sinus cardinal sin(x)/x
def sin_card(x):
    Nx = np.size(x)
    if Nx == 1:
        if x == 0:
            y = 1
        else:
            y = np.sin(x)/x
    else:
        y = np.zeros(Nx)
        for i in range(Nx):
            if x[i] ==0:
                y[i] = 1
            else:
                y[i] = np.sin(x[i])/x[i]
    return y

#%% read mesures - Fig 2

pathData = "data\\"
fname = pathData + 'SLMData.npz'
data = np.load(fname)
Nims = data['Nims']

# 1) dark: measures obtained whithout light
dark = data['dark']
(Ny,Nx) = np.shape(dark)

# 2) results when the SLM is used as a plain miror (no tilt is applied). It 
# enables to define the x,y coordinates with (x,y)=(0,0) the optical axis
Im02D = data['tilt000']
# remove dark
Im02D = Im02D - dark
# mean over the y direction
Im0 = np.sum(Im02D,0)/Ny
# x,y axis definition
x = np.arange(0,len(Im0))
x = x - x[np.argmax(Im0)]
y = np.arange(Ny)
y = y - y[np.argmax(np.sum(Im02D,1)/Nx)]

# 3) results when a tilt of 100 lambda is applied on the SLM (lambda is the 
# wavelenght of the beam illuminating the SLM)
Im1002D = data['tilt0100']
Im1002D = Im1002D - dark
Im100 = np.sum(Im1002D,0)/Ny
Im1002D = abs((Im1002D>0)*Im1002D)

plt.figure(1)
plt.clf()
plt.semilogy(x,Im100)
plt.xlim([min(x),max(x)])
plt.xlabel('x (pixels)')
plt.ylabel('ADU')
plt.grid()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(num=2)
cax = ax.imshow(np.log10(Im1002D), cmap='turbo',
                extent=[min(x),max(x),max(y),min(y)])
cbar = fig.colorbar(cax, ticks=[-2,-1,0,1,2,3])
cbar.ax.set_yticklabels([r'$10^{-2}$',r'$10^{-1}$',r'$10^0$',r'$10^1$',
                         r'$10^2$',r'$10^3$'])
cbar.ax.set_ylabel('ADU')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.grid()
plt.tight_layout()
plt.show()


#%% Model - Fig 3 a) and b)

# SLM parameters:
# size of the beam on the SLM defined in SLM pixels ie D=n
Nxbeam = 1100
D = Nxbeam
n = Nxbeam
# tilt definition
delta_lambda = 100
phi = delta_lambda*2*np.pi

# wrapping value determination phiw
# central spatial frequency associated to the tilt
nu0 = phi/2/np.pi/D
# find the two maxima - conversion factor between detector pixels and spatial 
# frequency
indCentralSpot = np.argmax(Im100)
indPixDiffraction = np.argmax(Im100*(x<-100))
nu2pix = x[indCentralSpot] - x[indPixDiffraction]
phiw = nu2pix*nu0*2*np.pi/x[indCentralSpot]
print('phiw = '+str(np.round(phiw/2/np.pi,4))+'*2pi')

# spatial frequency
nu = x/nu2pix

# spots considered 
k = np.arange(-5,10) # index k (grating due to incrorrect wrapping - see paper)
m = np.arange(-1,1)  # index m (grating due to SLM pixelization    - see paper)
Nm = len(m)
Nk = len(k)
nup = [] # spatial frequency of the spots
for i in range(Nk):
    for j in range(Nm):
        nup.append(k[i]*nu0*2*np.pi/phiw + m[j]*n/D)
nup.sort()
nu = np.concatenate((nu,nup))
nu.sort()
Nnu = len(nu)

# Model sgn = sum_{k,m} Akm*gkm(nu)
A = np.zeros((Nk,Nm),dtype=complex)
g = np.zeros((Nk,Nm,Nnu),dtype=complex)
sgn = np.zeros(Nnu,dtype=complex)
I2 = np.zeros(Nnu)
for i in range(Nk):
    for j in range(Nm):
        a = np.pi*D/n*(k[i]*nu0*2*np.pi/phiw + m[j]*n/D)
        b = np.pi*(k[i] - 2*np.pi/phiw)
        Akm = sin_card(a)*np.exp(1j*a)*sin_card(b)*np.exp(1j*b)
        A[i,j] = Akm
        c = np.pi*D*(nu - k[i]*nu0*2*np.pi/phiw -m[j]*n/D)
        gkm = sin_card(c)*np.exp(1j*c)
        g[i,j,:] = gkm
        sgn = sgn + Akm*gkm
        I2 = I2 + abs(Akm*gkm)**2

plt.figure(3)
plt.clf()
plt.semilogy(nu*nu2pix,abs(sgn)**2/max(abs(sgn)**2),label=r'$|\sum A_{km}g_{km}|^2$')
plt.semilogy(x,Im100/max(Im100)/200,'r',label='measures')
plt.legend()
plt.xlim(min(x),max(x))
plt.ylim(1e-8,1e-2)
plt.xlabel('x (pixels)')
plt.ylabel('normalized units')
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(4)
plt.clf()
plt.subplot(211)
plt.semilogy(nu*nu2pix,abs(sgn)**2/max(abs(sgn)**2),label=r'$|\sum A_{km}g_{km}|^2$')
plt.legend()
plt.xlim(min(x),max(x))
plt.ylim(1e-7,1e-3)
plt.ylabel('normalized units')
plt.grid()
plt.tight_layout()
plt.subplot(212)
plt.semilogy(nu*nu2pix,I2/max(I2),label=r'$\sum |A_{km}g_{km}|^2$')
plt.legend()
plt.xlim(min(x),max(x))
plt.ylim(1e-7,1e-3)
plt.xlabel('x (pixels)')
plt.ylabel('normalized units')
plt.grid()
plt.tight_layout()
plt.show()


#%% Model Fig 3 c)
# some spots are considered 

kplot0 = 1
kplot = [-1,2,3,4,5,6,7,8,9]
mplot = [-1]
mplot0 = -1
plt.figure(5,figsize=[6.4*1.34, 4.8])
plt.clf()
ink = np.argmin(abs(k-kplot0))
spotcentral = True
if k[ink] != kplot0:
    print('erreur: k = '+str(kplot0))
    spotcentral = False
inm = np.argmin(abs(m-mplot0))
if m[inm] != mplot0:
    print('erreur: m = '+str(mplot0))
    spotcentral = False
if spotcentral:
    Akm = A[ink,inm]
    gkm = g[ink,inm,:]
    kstr = str(k[ink])
    mstr = str(m[inm])
    lab = r'$|A_{'+kstr+','+mstr+'}g_{'+kstr+','+mstr+'}|^2$'
    plt.semilogy(nu*nu2pix,abs(Akm*gkm)**2,label = lab)
for i in range(len(kplot)):
    ink = np.argmin(abs(k-kplot[i]))
    if k[ink] != kplot[i]:
        print('erreur: k = '+str(kplot[i]))
        break
    else:
        for j in range(len(mplot)):
            inm = np.argmin(abs(m-mplot[j]))
            if m[inm] != mplot[j]:
                print('erreur: m = '+str(mplot[j]))
                break
            else:
                Akm = A[ink,inm]
                gkm = g[ink,inm,:]
                kstr = str(k[ink])
                mstr = str(m[inm])
                lab = r'$|A_{'+kstr+','+mstr+'}g_{'+kstr+','+mstr+'}|^2$'
                plt.semilogy(nu*nu2pix,abs(Akm*gkm)**2,label = lab)
plt.semilogy(x,Im100/max(Im100)/200,'r')
plt.xlim(min(x),max(x))
plt.ylim(1e-7,1e-3)
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.xlabel('x (pixels)')
plt.ylabel('normalized units')
plt.grid()
plt.tight_layout()
plt.show()

#%% Model Fig 3 d)
# some spots are considered 

kplot0 = 1
kplot = [  -5,  -4,  -3,  -2,  -1,   0,   2,   3, 4]
mplot = [0]
mplot0 = 0
plt.figure(6,figsize=[6.4*1.34, 4.8])
plt.clf()
ink = np.argmin(abs(k-kplot0))
spotcentral = True
if k[ink] != kplot0:
    print('erreur: k = '+str(kplot0))
    spotcentral = False
inm = np.argmin(abs(m-mplot0))
if m[inm] != mplot0:
    print('erreur: m = '+str(mplot0))
    spotcentral = False
if spotcentral:
    Akm = A[ink,inm]
    gkm = g[ink,inm,:]
    kstr = str(k[ink])
    mstr = str(m[inm])
    lab = r'$|A_{'+kstr+','+mstr+'}g_{'+kstr+','+mstr+'}|^2$'
    plt.semilogy(nu*nu2pix,abs(Akm*gkm)**2,label = lab)
for i in range(len(kplot)):
    ink = np.argmin(abs(k-kplot[i]))
    if k[ink] != kplot[i]:
        print('erreur: k = '+str(kplot[i]))
        break
    else:
        for j in range(len(mplot)):
            inm = np.argmin(abs(m-mplot[j]))
            if m[inm] != mplot[j]:
                print('erreur: m = '+str(mplot[j]))
                break
            else:
                Akm = A[ink,inm]
                gkm = g[ink,inm,:]
                kstr = str(k[ink])
                mstr = str(m[inm])
                lab = r'$|A_{'+kstr+','+mstr+'}g_{'+kstr+','+mstr+'}|^2$'
                plt.semilogy(nu*nu2pix,abs(Akm*gkm)**2,label = lab)
plt.semilogy(x,Im100/max(Im100)/200,'r')
plt.xlim(min(x),max(x))
plt.ylim(1e-7,1e-3)
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.xlabel('x (pixels)')
plt.ylabel('normalized units')
plt.grid()
plt.tight_layout()
plt.show()

#%% Efficiency - Fig 5

phi = 3*np.pi
phi_lambda = phi/np.pi/2
phiw = [1*2*np.pi,0.99*2*np.pi,0.98*2*np.pi,0.97*2*np.pi]
n = np.arange(3, 28, 0.1)

eff = np.zeros((len(n),len(phiw)))
for j in range(len(phiw)):
    eff[:,j] = sin_card(phi/2/n*2*np.pi/phiw[j])**2*sin_card(np.pi*(1-2*np.pi/phiw[j]))**2

plt.figure(7)
plt.clf()
indphiwplot = [0,3]
for j in range(len(indphiwplot)):
    plt.plot(n/phi_lambda,eff[:,indphiwplot[j]]*100, label = str(np.round(phiw[indphiwplot[j]]/2/np.pi,2)))
plt.xlabel(r'$n/\phi (pixels/\lambda)$')
plt.ylabel('Efficiency (%)')
plt.xlim((min(n/phi_lambda),max(n/phi_lambda)))
plt.ylim([40,100])
plt.xticks(np.arange(2,20,2))
plt.legend(title=r"$\phi_w/2/\pi$")
plt.tight_layout()
plt.grid()
plt.show()

plt.figure(8)
plt.clf()
for j in range(len(phiw)-1):
    plt.plot(n/phi_lambda,(eff[:,0]-eff[:,j+1])/eff[:,0]*100,label=str(np.round(phiw[j+1]/2/np.pi,2)))
plt.xlabel(r'$n/\phi (pixels/\lambda)$')
plt.ylabel('Efficiency loss (%)')
plt.xlim((min(n/phi_lambda),max(n/phi_lambda)))
plt.legend(title=r"$\phi_w/2/\pi$")
plt.ylim([0,7])
plt.xticks(np.arange(2,20,2))
plt.grid()
plt.tight_layout()
plt.show()


