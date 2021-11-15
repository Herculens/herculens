#################################################################################################
#Loading necesarry packages
#################################################################################################

import math
import numpy as np
from numpy.fft import fft, fft2, fftn, fftfreq, ifft, fftshift

#################################################################################################
# Power spectrum
#################################################################################################

def powspec(L, variance, Npix, Psum, power):
    if L <= 0:
        P = 0.0 # variance*(Npix**2.)/(2.*Psum)*L**(0.0)
    else:
        A = variance*(Npix**2.)/(2.*Psum)
        P = A*L**(power)
    return P

#################################################################################################
# Sum over the power spectrum
#################################################################################################

def Psum_calculator(nx, ny, Lx, Ly, power):
    #lxaxis = np.append (np.arange (0.,(nx/2.)/Lx, 1./Lx), np.arange ((-nx/2.)/Lx, 0.,1./Lx))
    #lyaxis = np.append (np.arange (0.,(ny/2.)/Ly, 1./Ly), np.arange ((-ny/2.)/Ly, 0.,1./Ly))
    resolution = float(Lx) / float(nx)
    lxaxis = np.fft.fftfreq (nx, resolution)
    lyaxis = np.fft.fftfreq (ny, resolution)

    lx = list (np.zeros([nx, 1]))
    ly = list (np.zeros([ny, 1]))

    for x in range (len(lx)):
        lx [x] = lxaxis
    
    for y in range (len(ly)):
        ly [y] = lyaxis
    
    lx = np.array (lx)
    ly = np.transpose(np.array (ly))
    l  = np.sqrt (lx**2. + ly**2.)
    
    summ = 0.
    for y in range(np.shape(l)[0]):
        for x in range(np.shape(l)[1]):
            if l[y][x]==0.:
                summ += 0.
            else:
                summ += l[y][x]**(power)
    return summ
    
################################################################################################

def gauss_rand_2d (par):
    nx = par[0].astype(int)
    ny = par[1].astype(int)
    Lx = par[2]
    Ly = par[3]
    var = par[4]
    power = par[5]

    j= 0 + 1j # Defining the complex number
    plane = np.zeros ([nx, ny], dtype='cfloat') # Empty matrix to be filled in for the Fourier plane

    #lxaxis = np.append (np.arange (0.,(nx/2.)/Lx, 1./Lx), np.arange ((-nx/2.)/Lx, 0.,1./Lx))
    #lyaxis = np.append (np.arange (0.,(ny/2.)/Ly, 1./Ly), np.arange ((-ny/2.)/Ly, 0.,1./Ly))

    resolution = float(Lx) / float(nx)
    lxaxis = np.fft.fftfreq (nx, resolution)
    lyaxis = np.fft.fftfreq (ny, resolution)

    Psum = Psum_calculator (nx, ny, Lx, Ly, power)

    for y in range(np.shape(plane)[0]):
        for x in range(np.shape(plane)[1]):
            # The coordinates centered at x = n/2, y = n/2
            i1 = x - nx/2
            j1 = y - ny/2 

            #The coordinates in the Fourier plane
            lx = lxaxis [x]
            ly = lyaxis [y]
            # The magnitude of the l-vector
            l = np.sqrt (lx**2. + ly**2.) 

            # Polar Box-Muller transform
            sigma = math.sqrt(powspec(l, var, nx*ny, Psum, power))
            s = 1.1
            while s > 1. :
                u = np.random.uniform (-1.,1.) 
                v = np.random.uniform (-1.,1.)
                s = u**2. + v**2.
            fac = math.sqrt(-2.*math.log(s)/s)
            z1 = u*fac*sigma
            z2 = v*fac*sigma

            # Filling in the grid
            if x==0 and y==0: # average of the field
                plane[y][x] = 0.0

            # three points that need to be real valued to get a real image after FFT:
            elif x== 0 and y==ny/2:
                plane[y][x] = z1
            elif x==nx/2 and y==0:
                plane[y][x] = z1
            elif x==nx/2 and y==ny/2:
                plane[y][x] = z1
            else :
                plane[y][x] = z1+j*z2

            # Creating symmetry f(k) = f*(-k)
            y2 = int( -(j1 + ny/2) )
            x2 = int( -(i1 + nx/2) )
        
            plane[y2][x2] = plane[y][x].conjugate()

        if y>np.shape(plane)[0]/2.:
            break
    return np.fft.ifftshift(np.fft.ifft2(plane)).real
