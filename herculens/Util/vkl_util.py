# This file is part of Herculens, a python code for lens
# modelling and it is based on the VKL C++ code described in
# Vernardos & Koopmans 2022, which can be found here:
# https://github.com/gvernard/verykool
#
# Copyright (C) 2022 Georgios Vernardos <gvernard@astro.rug.nl>
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

__author__ = 'gvernard', 'aymgal'


import math
import numpy as np
from scipy import sparse


def vkl_operator(data_x,data_y,dpsi_xmin,dpsi_dx,dpsi_Nx,dpsi_x,dpsi_ymax,dpsi_dy,dpsi_Ny,dpsi_y,source0_dx,source0_dy):
    # Arguments:
    #  data_x: An array containing the X coordinates of the data pixels - must be the length of the data
    #  data_y: An array containing the Y coordinates of the data pixels - must be the length of the data
    #  dpsi_xmin: The minimum X coordinate of the dpsi grid
    #  dpsi_dx: The separation between two dpsi grid pixels in the X direction
    #  dpsi_Nx: The number of dpsi pixels in the X direction
    #  dpsi_x: An array containing the X coordinates of the dpsi pixels - must be the length of the data
    #  dpsi_ymax: The maximum Y coordinate of the dpsi grid
    #  dpsi_dy: The separation between two dpsi grid pixels in the Y direction
    #  dpsi_Ny: The number of dpsi pixels in the Y direction
    #  source0_x: An array of the source derivatives in the X direction at the location of the deflected data pixels (size=number of data pixels)
    #  source0_y: An array of the source derivatives in the Y direction at the location of the deflected data pixels (size=number of data pixels)
    crosses = createCrosses(data_x,data_y,dpsi_xmin,dpsi_dx,dpsi_Nx,dpsi_x,dpsi_ymax,dpsi_dy,dpsi_Ny,dpsi_y)
    operator = constructDsDpsi(crosses,source0_dx,source0_dy,dpsi_Nx,dpsi_Ny)
    return operator


def derivativeDirection(q,qmax,den):
    # Description:
    #  Helper function (called only from within this code) that calculates the finite difference coefficients and their indices.
    #  This is done for the 2nd order accuracy 1st derivatives, for both X and Y derivatives.
    # Arguments:
    #  q: the index (in X or Y direction) of the given dpsi grid pixel where the derivative needs to be calculated via finite differences
    #  qmax: the maximum index (in X or Y direction) of the dpsi grid
    #  den: the denominator to use in the finite difference coefficients, i.e. either dx or dy of the dpsi grid
    # Returns:
    #  rel_ind: the relative indices of the finite difference coefficients with respect to q
    #  coeff: the finite difference coefficients

    if q == 0:
        # FORWARD finite difference, 1st derivative, 2nd order
        rel_ind = [0,1,2]
        coeff = [-3.0/2.0, 2.0, -1.0/2.0]/den
    elif q == qmax:
        # BACKWARD finite difference, 1st derivative, 2nd order
        rel_ind = [0,-1,-2]
        coeff = [3.0/2.0, -2.0, 1.0/2.0]/den
    else:
        # CENTRAL finite difference, 1st derivative, 2nd order
        rel_ind = [-1,0,1]
        coeff = [-1.0/2.0, 0.0, 1.0/2.0]/den
    return rel_ind,coeff


def createCrosses(data_x,data_y,dpsi_xmin,dpsi_dx,dpsi_Nx,dpsi_x,dpsi_ymax,dpsi_dy,dpsi_Ny,dpsi_y):
    # Description:
    #  This function creates a 'cross' structure for each data pixel.
    # Arguments:
    #  data_x: An array containing the X coordinates of the data pixels - must be the length of the data
    #  data_y: An array containing the Y coordinates of the data pixels - must be the length of the data
    #  dpsi_xmin: The minimum X coordinate of the dpsi grid
    #  dpsi_dx: The separation between two dpsi grid pixels in the X direction
    #  dpsi_Nx: The number of dpsi pixels in the X direction
    #  dpsi_x: An array containing the X coordinates of the dpsi pixels - must be the length of the data
    #  dpsi_ymax: The maximum Y coordinate of the dpsi grid
    #  dpsi_dy: The separation between two dpsi grid pixels in the Y direction
    #  dpsi_Ny: The number of dpsi pixels in the Y direction
    # Returns:
    #  crosses: An array of 'cross' structures

    # Loop over the data pixels and assign a 'cross' structure to each.
    crosses = []
    for h in range(0,len(data_x)):
        # Step 1: find where the data pixel (ray) lies within the dpsi grid.
        #         This ray will lie between 4 pixels in the dpsi grid.
        #         Find the indices (row,col) of the top left dpsi pixel of these four.
        j0 = math.floor( (data_x[h] - dpsi_xmin)/dpsi_dx )
        i0 = math.floor( (dpsi_ymax - data_y[h])/dpsi_dy ) # y-axis is reflected (indices i start from the top)
        if j0 >= (dpsi_Nx - 1):
            j0 = dpsi_Nx - 2
        if i0 >= (dpsi_Ny - 1):
            i0 = dpsi_Ny - 2
        four_i = [i0,i0,i0+1,i0+1]
        four_j = [j0,j0+1,j0,j0+1]

        
        # Step 2: get the interpolation weights for the ray (data) with respect to the four dpsi grid pixels that contain it
        ya = data_y[h]               - dpsi_y[(i0+1)*dpsi_Nx]
        yb = dpsi_y[i0*dpsi_Nx]      - data_y[h]
        xa = data_x[h]               - dpsi_x[i0*dpsi_Nx+j0]
        xb = dpsi_x[i0*dpsi_Nx+j0+1] - data_x[h]
        den = dpsi_dx*dpsi_dy
        four_w = [xb*ya,xa*ya,xb*yb,xa*yb]/den

        # Step 3: loop over the dpsi pixel vertices and store the X and Y coefficients in the cross.
        #         The way the coefficients are ordered is defined ad-hoc, left-to-right and top-to-bottom (see notes).
        cross = {
            'i0': i0,
            'j0': j0,
            'coeff_x': np.zeros(8),
            'coeff_y': np.zeros(8)
        }
        for k in range(0,len(four_i)):
            # X derivatives
            rel_ind,coeffs = derivativeDirection(four_j[k],dpsi_Nx-1,dpsi_dx)
            for qq in range(0,len(rel_ind)):
                cross_index = (four_i[k]-i0)*4 + (four_j[k]-j0) + rel_ind[qq] + 1
                cross['coeff_x'][cross_index] += coeffs[qq]*four_w[k]

            # Y derivatives
            rel_ind,coeffs = derivativeDirection(four_i[k],dpsi_Ny-1,dpsi_dy)
            for qq in range(0,len(rel_ind)):
                cross_index = (four_i[k]-i0+rel_ind[qq]+1)*2 + (four_j[k]-j0)
                cross['coeff_y'][cross_index] += coeffs[qq]*four_w[k]
        crosses.append(cross)

    return crosses


def constructDsDpsi(crosses,source0_x,source0_y,dpsi_Nx,dpsi_Ny):
    # Description:
    #  Finds the non-zero coefficients for the term given by equation 7 in Vernardos & Koopmans 2022 for each data pixel.
    # Arguments:
    #  crosses: An array of 'cross' structures, the output of 'createCrosses' (size=number of data pixels)
    #  source0_x: An array of the source derivatives in the X direction at the location of the deflected data pixels (size=number of data pixels)
    #  source0_y: An array of the source derivatives in the Y direction at the location of the deflected data pixels (size=number of data pixels)
    # Returns:
    #  sparse_mat: An array of tuples. Each tuple contains the row index, column index, and value of the non-zero elements of the sparse matrix
    rel_i = [-1, -1,  0, 0, 0, 0,  1, 1, 1, 1, 2, 2]
    rel_j = [ 0,  1, -1, 0, 1, 2, -1, 0, 1, 2, 0, 1]
    vals = np.zeros(12)
    
    # sparse_mat = []
    sparse_values = []
    sparse_rows, sparse_cols = [], []
    for h in range(0,len(crosses)):
        src_Dx = source0_x[h]
        src_Dy = source0_y[h]

        vals[0]  = crosses[h]['coeff_y'][0]*src_Dy
        vals[1]  = crosses[h]['coeff_y'][1]*src_Dy
        vals[2]  = crosses[h]['coeff_x'][0]*src_Dx
        vals[3]  = crosses[h]['coeff_x'][1]*src_Dx + crosses[h]['coeff_y'][2]*src_Dy
        vals[4]  = crosses[h]['coeff_x'][2]*src_Dx + crosses[h]['coeff_y'][3]*src_Dy
        vals[5]  = crosses[h]['coeff_x'][3]*src_Dx
        vals[6]  = crosses[h]['coeff_x'][4]*src_Dx
        vals[7]  = crosses[h]['coeff_x'][5]*src_Dx + crosses[h]['coeff_y'][4]*src_Dy
        vals[8]  = crosses[h]['coeff_x'][6]*src_Dx + crosses[h]['coeff_y'][5]*src_Dy
        vals[9]  = crosses[h]['coeff_x'][7]*src_Dx
        vals[10] = crosses[h]['coeff_y'][6]*src_Dy
        vals[11] = crosses[h]['coeff_y'][7]*src_Dy
        
        for q in range(0,12):
            if vals[q] != 0:
                col_index = (crosses[h]['i0']+rel_i[q])*dpsi_Nx + (crosses[h]['j0']+rel_j[q])
                
                # sparse_mat.append( (h,col_index,vals[q]) )
                sparse_values.append(vals[q])
                sparse_rows.append(h)
                sparse_cols.append(col_index)
            
            vals[q] = 0 # Need to set the values of the 12 coefficients back to zero

    # populate the sparse matrix
    Ndata = len(crosses)
    Ndpsi = dpsi_Nx*dpsi_Ny
    dense_shape = (Ndata, Ndpsi)
    DsDpsi_matrix = sparse.csr_matrix((sparse_values, (sparse_rows, sparse_cols)), 
                                      shape=dense_shape)
    return DsDpsi_matrix
