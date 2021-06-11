# =========================================================================
# 
# GPTO
# 
# A Python/Numpy adaptation of the MATLAB code for topology optimization 
# with bars using the geometry projection method.
# Version 0.5 -- April 2021
#
# Original Authors: Hollis Smith and Julian Norato
# Department of Mechanical Engineering
# University of Connecticut

# Translator: Andres Ortegon
# Departamento de Matemáticas
# Universidad Nacional de Colombia
#
# ORIGINAL COMENTS:
#
# Disclaimer
# ==========
# This software is provided by the contributors "as-is" with no explicit or
# implied warranty of any kind. In no event shall the University of
# Connecticut or the contributors be held liable for damages incurred by
# the use of this software.
#
# License
# =======
# This software is released under the Creative Commons CC BY-NC 4.0
# license. As such, you are allowed to copy and redistribute the material 
# in any medium or format, and to remix, transform, and build upon the 
# material, as long as you: 
# a) give appropriate credit, provide a link to the license, and indicate 
# if changes were made. You may do so in any reasonable manner, but not in 
# any way that suggests the licensor endorses you or your use.
# b) do not use it for commercial purposes.
#
# To fulfill part a) above, we kindly ask that you please cite the paper
# that introduces this code:
#
# Smith, H. and Norato, J.A. "A MATLAB code for topology optimization
# using the geometry projection method."
# Structural and Multidisciplinary Optimization, 2020,
# https://doi.org/10.1007/s00158-020-02552-0
#
# =========================================================================

## source folders containing scripts not in this folder
import time
import os
from typing import Iterable

import numpy as np
import scipy.sparse as sp 
import matplotlib.collections
import matplotlib.pyplot as plt

from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy.sparse import linalg
from scipy.io import savemat
from MMA import mmasub, kktcheck


OPT  = {}
GEOM = {}
FE   = {}

FE['FEA_t']    = 0 
FE['FEA_n']    = 0
GEOM['proj_t'] = 0
OPT['Fun_t']   = 0

def fd_check_cost():
    # This function performs a finite difference check of the sensitivities of
    # the COST function with respect to the bar design variables.
    global OPT 

    # Finitite diference sensitivities
    n_dv = OPT['n_dv']
    grad_theta_i = np.zeros( (n_dv,1) )

    fd_step = OPT['fd_step_size']

    max_error           = 0.0 
    max_rel_error       = 0.0 
    max_error_bar       = 0 
    max_rel_error_bar   = 0 
    dv_0 = OPT['dv'].copy() 
    dv_i = OPT['dv'].copy()

    theta_0, grad_theta_0 = obj(dv_0) 
    
    # Finite differences
    print('Computing finite difference sensitivities...') 

    # Do this for all design variables or only a few
    up_to_dv = n_dv 

    for i in range(0,up_to_dv):
        # Preturb dv
        dv_i[i] = dv_0[i] + fd_step 
        theta_i , __ = obj(dv_i) 

        grad_theta_i[i] = (theta_i - theta_0)/fd_step 

        error = grad_theta_0[i] - grad_theta_i[i] 

        if np.abs(error) > np.abs(max_error):
            max_error = error 
            max_error_dv = i 

        rel_error = error / theta_0 
        if np.abs(rel_error) > np.abs(max_rel_error):
            max_rel_error = rel_error 
            max_rel_error_dv = i    

        dv_i = dv_0.copy() 

    theta_0, grad_theta_0 = obj(dv_0)  # to reset the design

    print('Max. ABSOLUTE error is:' )
    print(max_error)
    print('It occurs at:') 
    print('\tvariable:' + str(max_error_dv) )

    print('Max. RELATIVE error is:' )
    print(max_rel_error)
    print('It occurs at:') 
    print('\tvariable:' + str(max_rel_error_dv) )

    plt.figure(2)
    plt.plot(grad_theta_i,'+')
    plt.plot(grad_theta_0,'.')
    # plt.legend('fd','analytical')
    plt.title('cost function')
    plt.xlabel('design variable z') 
    plt.ylabel('dc/dz') 
    plt.show()


def fd_check_constraint():
    # This function performs a finite difference check of the sensitivities of
    # the CONSTRAINT function with respect to the bar design variables.
    # It is currently setup for one constraint, but it can be easily modified
    # for other/more constraints.
    global OPT

    # ===============================
    # FINITE DIFFERENCE SENSITIVITIES
    # ===============================
    n_dv = OPT['n_dv']
    grad_theta_i = np.zeros( (n_dv, 1) )

    fd_step = OPT['fd_step_size']

    max_error           = 0.0 
    max_rel_error       = 0.0 
    max_error_bar       = 0 
    max_rel_error_bar   = 0 
    
    dv_0 = OPT['dv'].copy()
    dv_i = OPT['dv'].copy()

    theta_0 = nonlcon(dv_0) 
    grad_theta_0 = nonlcongrad(dv_0)
    # Finite differences
    print('Computing finite difference sensitivities...') 

    # Do this for all design variables or only a few
    # up_to_dv = n_dv 
    up_to_dv = n_dv 

    for i in range(0,up_to_dv):
        #perturb dv
        dv_i[i] = dv_0[i] + fd_step 
        theta_i = nonlcon(dv_i) 

        grad_theta_i[i] = (theta_i - theta_0)/fd_step 

        error = grad_theta_0[i] - grad_theta_i[i] 

        if np.abs(error) > np.abs(max_error):
            max_error = error 
            max_error_dv = i 
        
        rel_error = error / theta_0 
        if np.abs(rel_error) > np.abs(max_rel_error):
            max_rel_error = rel_error 
            max_rel_error_dv = i    

        dv_i = dv_0.copy()

    theta_0 = nonlcon(dv_0)  # to reset the design
    grad_theta_0 = nonlcongrad(dv_0)  # to reset the design

    print('Max. ABSOLUTE error is:') 
    print( max_error )
    print('It occurs at:') 
    print('\tvariable:' + str(max_error_dv) )

    print('Max. RELATIVE error is:')
    print( max_rel_error )
    print('It occurs at:') 
    print('\tvariable:' + str(max_rel_error_dv) )

    plt.figure(3)
    plt.plot(grad_theta_i,'+')
    plt.plot(grad_theta_0,'.')
    # plt.legend('fd','analytical')
    plt.title('constraint function')
    plt.xlabel('design variable z') 
    plt.ylabel('dv/dz') 
    plt.show()


def run_finite_difference_check():
    # This function performs a finite difference check of the analytical
    # sensitivities of the cost and/or constraint functions by invoking the
    # corresponding routines.
    global OPT

    if OPT['check_cost_sens']:
        fd_check_cost()
    if OPT['check_cons_sens']:
        fd_check_constraint()


def plot_density(fig):
    if FE['mesh_input']['type'] =='read-gmsh':
        # mesh was made by gmsh
        plot_density_cells(fig)
    else:
        # mesh was generated and comforms to meshgrid format. 
        # we then default to plotting level-sets of the density as
        # linearly interpolated between the centroids of the mesh.
        plot_density_levelsets(fig)


def plot_density_levelsets(fig):
    global FE, OPT, GEOM

    if 2 == FE['dim']:
        img = 1 - OPT['elem_dens'].reshape( (
            FE['mesh_input']['elements_per_side'][1],
            FE['mesh_input']['elements_per_side'][0]) , order='F')

        img = np.flip(img,0)

        plt.ion()
        plt.figure(fig)
        plt.imshow(img)
        plt.gray()
        plt.pause(0.0001)
        plt.draw()


def plot_density_cells(fig):
    # Plot the density field into the specified figure
    global FE, OPT

    ## Change here whether you want to plot the penalized (i.e., effective) or 
    ## the unpenalized (i.e., projected) densities.  By default, we plot the 
    ## effective densities.
    #
    # For penalized, use OPT.penalized_elem_dens;
    # For unpenalized, use OPT.elem_dens;
    #
    # plot_dens = OPT['penalized_elem_dens'];
    plot_dens = OPT['elem_dens']

    ## 2D
    if FE['dim'] == 2:
        F = FE['elem_node'].T # matrix of faces to be sent to patch function
        V = FE['coords'].T # vertex list to be sent to patch function

    ## 3D
    if FE['dim'] == 3:
        element_face_nodes = np.array( (
            (1,2,3,4) ,
            (1,2,6,5) ,
            (2,3,7,6) , 
            (3,4,8,7) , 
            (4,1,5,8) , 
            (5,6,7,8) ) ).T
        F = FE['elem_node'][element_face_nodes,:].reshape((-1,4))
        V = FE['coords'].T # vertex lest to be sent to patch function
    
    plt.ion()
    plt.figure(fig)
    ax = plt.gca()
    ax.cla()

    verts = V[F]
    pc  = matplotlib.collections.PolyCollection(verts,  cmap='gray' )
    pc.set_array(1-plot_dens)
    ax.add_collection(pc)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim( (FE['coord_min'][0], FE['coord_max'][0]) )
    plt.ylim( (FE['coord_min'][1], FE['coord_max'][1]) )

    plt.pause(0.0001)
    plt.draw()

    # for n levels of opacity color
    # n = 64
    # level = np.linspace(0,1,n+1)
    # ax  = plt.gca()

    # for i in range(n,1,-1): #1:n
    #     low     = level[i-1]
    #     high    = level[i]
    #     alpha   = low

    #     if FE['dim'] == 3:
    #         C = np.amin( plot_dens , axis=0 ).repeat(6,axis=0)
    #     else:
    #         C = np.amin( plot_dens , axis=0 )
        
    #     f = np.logical_and( low < C , C <= high )

    #     verts = V[F[f,:]]
    #     pc  = matplotlib.collections.PolyCollection(verts,  cmap='gray' )
    
    #     pc.set_array(alpha)
    #     ax.add_collection(pc)


def plot_design(*args):
    # Plot_design(fig,point_mat,bar_mat) plots the bars into the figure fig
    # fig is the number (or handle) of the figure to use
    global GEOM, FE

    nargs = len(args)
    if nargs == 0:
        fig = 1
        point_mat = GEOM['current_design']['point_matrix']
        bar_mat = GEOM['current_design']['bar_matrix']
    elif nargs == 1:
        fig = args[0]
        point_mat = GEOM['current_design']['point_matrix']
        bar_mat = GEOM['current_design']['bar_matrix']   
    elif nargs == 3:
        fig = args[0]
        point_mat = args[1]
        bar_mat = args[2] 
    else:
        print('plot_design received an invalid number of arguments.')

    ## user specified parameters

    # set the color of the bars
    bar_color = (1,0,0)    # red 
    # set size variable threshold to plot bars
    size_tol = 0.05
    # set the resolution of the bar-mesh (>=8 and even)
    N = 16

    ## bar points,vectors and length
    bar_tol = 1e-12; # threshold below which bar is just a circle
    n_bar = bar_mat.shape[0]

    x_1b = np.zeros( (3,n_bar) )
    x_2b = np.zeros( (3,n_bar) ) # these are always in 3D 

    pt1_IDs = bar_mat[:,1].astype(int)
    pt2_IDs = bar_mat[:,2].astype(int)

    x_1b[0:FE['dim'],:] = point_mat[GEOM['point_mat_row'][pt1_IDs].toarray()[:,0],1:].T 
    x_2b[0:FE['dim'],:] = point_mat[GEOM['point_mat_row'][pt2_IDs].toarray()[:,0],1:].T
        
    n_b = x_2b - x_1b
    l_b = np.sqrt(np.sum(n_b*n_b,0))[None,:] # length of the bars
    
    ## principle bar direction
    e_hat_1b = n_b/l_b
    short = l_b < bar_tol
    if short.any():
        e_hat_1b[:,short[0,:]] = np.tile( np.array([[1],[0],[0]]) , (1,sum(short)) )

    # determine coordinate direction most orthogonal to bar
    case_1 = ( abs(n_b[0,:]) < abs(n_b[1,:]) ) & ( abs(n_b[0,:]) < abs(n_b[2,:]) )
    case_2 = ( abs(n_b[1,:]) < abs(n_b[0,:]) ) & ( abs(n_b[1,:]) < abs(n_b[2,:]) )
    case_3 = np.logical_not( case_1 | case_2 )
    
    ## secondary bar direction
    e_alpha = np.zeros(n_b.shape)
    e_alpha[0,case_1] = 1
    e_alpha[1,case_2] = 1
    e_alpha[2,case_3] = 1

    e_2b        = l_b * np.cross(e_alpha,e_hat_1b,axis=0)
    norm_e_2b   = np.sqrt( np.sum(e_2b**2) )
    e_hat_2b    = e_2b/norm_e_2b
    
    ## tertiary bar direction
    e_3b        = np.cross(e_hat_1b,e_hat_2b,axis=0)
    norm_e_3b   = np.sqrt( sum(e_3b**2) )
    e_hat_3b    = e_3b/norm_e_3b

    ## Jacobian transformation (rotation) matrix R
    R_b = np.zeros( (3,3,n_bar) )
    R_b[:,0,:] = e_hat_1b
    R_b[:,1,:] = e_hat_2b
    R_b[:,2,:] = e_hat_3b
        
    # create the reference-sphere mesh
    if FE['dim'] == 3:
        theta = np.linspace( -np.pi , 0 , N+1 )[:,None]
        phi = np.linspace( 0 , 2*np.pi , N+1 )[:,None]

        z = np.cos( theta ) * np.ones((N+1,N+1))
        y = np.sin( theta ) @ np.sin( phi ).T
        x = np.sin( theta ) @ np.cos( phi ).T

        sx1 = z[0:N//2,:]
        sy1 = x[0:N//2,:]
        sz1 = y[0:N//2,:]
        sx2 = z[N//2+1:,:]
        sy2 = x[N//2+1:,:]
        sz2 = y[N//2+1:,:]
        X1 = [sx1, sy1, sz1].T
        X2 = [sx2, sy2, sz2].T
    else:
        N = N**2
        t =  np.linspace( -np.pi/2 , -np.pi/2+2*np.pi , N )
        x = -np.cos(t)
        y =  np.sin(t)
        z =  np.zeros(t.shape)

        cxo = x[0:N//2]
        cyo = y[0:N//2]
        czo = z[0:N//2]

        cxf = x[N//2:]
        cyf = y[N//2:]
        czf = z[N//2:]

        X1 = np.array([cxo, cyo, czo])
        X2 = np.array([cxf, cyf, czf])

    ## create the surface for each bar and plot it
    vertices = np.zeros((n_bar,256,2)) 
    r_b     = bar_mat[:,-1]
    alpha   = bar_mat[:,-2]

    for b in range(0,n_bar):
        bar_X1 = r_b[b] * R_b[:,:,b] @ X1 + x_1b[:,b][:,None]
        bar_X2 = r_b[b] * R_b[:,:,b] @ X2 + x_2b[:,b][:,None]

        if FE['dim'] == 3:
            bar_x1 = np.reshape(bar_X1[0,:], [N/2, N])
            bar_y1 = np.reshape(bar_X1[1,:], [N/2, N])
            bar_z1 = np.reshape(bar_X1[2,:], [N/2, N])

            bar_x2 = np.reshape(bar_X2[0,:], [N/2, N])
            bar_y2 = np.reshape(bar_X2[1,:], [N/2, N])
            bar_z2 = np.reshape(bar_X2[2,:], [N/2, N])
        else:
            bar_x1 = bar_X1[0,:].T
            bar_y1 = bar_X1[1,:].T
            bar_z1 = bar_X1[2,:].T

            bar_x2 = bar_X2[0,:].T
            bar_y2 = bar_X2[1,:].T
            bar_z2 = bar_X2[2,:].T

        bar_x = np.concatenate( (bar_x1 , bar_x2) )
        bar_y = np.concatenate( (bar_y1 , bar_y2) )
        bar_z = np.concatenate( (bar_z1 , bar_z2) )

        vertices[b,:,0] = bar_x
        vertices[b,:,1] = bar_y
        

    plt.ion()
    plt.figure(fig)    
    ax = plt.gca()
    ax.cla()

    Color = bar_color
    Alpha = alpha[b]**2

    # C = colormap('gray')
    # colormap(C.*Color) # color the gray-scale map

    if FE['dim'] == 3:
        s = surfl(bar_x,bar_y,bar_z); # shaded surface with lighting
        s.LineStyle = 'none'
        s.FaceAlpha = Alpha

        plt.zlim( (FE['coord_min'][2], FE['coord_max'][2]) )
        # shading interp
    else:
        pc  = matplotlib.collections.PolyCollection(vertices,cmap='gray')
        pc.set_array(128*np.ones(8))
        ax.add_collection(pc)
        # s.FaceAlpha = Alpha

    plt.xlim( (FE['coord_min'][0], FE['coord_max'][0]) )
    plt.ylim( (FE['coord_min'][1], FE['coord_max'][1]) )
    plt.gca().set_aspect('equal', adjustable='box')

    plt.pause(0.0001)
    plt.draw()


def plot_history(fig):
    global FE, OPT, GEOM

    plt.figure(fig)

    plt.subplot(2,1,1)
    a = plt.semilogy( OPT['history']['fval'].T )
    plt.title( 'Objective history' )
    plt.xlabel( 'Iteration' )
    plt.legend( OPT['functions']['f'][0]['name'] )

    if 'fconsval' in OPT['history']:
        print(OPT['history']['fconsval'].shape)
        g = OPT['history']['fconsval'].\
            reshape( ( -1 , OPT['functions']['n_func']-1 ) ) + \
            OPT['functions']['constraint_limit']
        
        label = {}
        scale = np.ones((1,OPT['functions']['n_func']-1))
        
        for i in range( 1 , OPT['functions']['n_func'] ):
            label[i-1] = OPT['functions']['f'][i]['name']
            if 'angle constraint' ==  OPT['functions']['f'][i]['name']:
                scale[i-1] = OPT['options']['angle_constraint']['scale']
        
        plt.subplot(2,1,2)
        plt.plot( g/scale )
        plt.title( 'Constraint history' )
        plt.xlabel( 'Iteration' )
        plt.legend( label )
    
    plt.show()


def writevtk(folder, name_prefix, iteration):
    # This function writes a vtk file with the mesh and the densities that can
    # be plotted with, e.g., ParaView
    #
    # This function writes an unstructured grid (vtk format) to folder (note
    # that the folder is relative to the rooth folder where the main script is
    # located).
    #
    # NOTE: if a vtk file with the same name exists in the folder, it will be
    # overwritten.
    # global FE, OPT

    # Make sure the output folder exists, and if not, create it
    if not os.path.isdir( OPT['options']['vtk_output_path'] ):
        os.mkdir( OPT['options']['vtk_output_path'] )

    num_digits = len( str(OPT['options']['max_iter']) )
    name_sufix = ('#0' + str(num_digits) + '{it}' ).format( it=iteration )
    filename = name_prefix + name_sufix + '.vtk'
    filename = folder + '/' + filename
    
    if os.path.isfile( filename ):
        os.remove(filename)

    fid = open( filename , 'a')

    # Write header
    fid.write( "# vtk DataFile Version 1.0 \n" )
    fid.write( "Bar_TO_3D \n" )
    fid.write(  "ASCII \n" )
    fid.write(  "DATASET UNSTRUCTURED_GRID \n" )

    # Write nodal coordinates
    coords = np.zeros((3, FE['n_node']))
    coords[np.arange(0,FE['dim']),:] = FE['coords'][np.arange(0,FE['dim']),:]   

    fid.write( "POINTS " + str(FE['n_node']) + " float \n" )
    for inode in range(0,FE['n_node']):
        if FE['dim'] == 2:
            fid.write(  '{0:f} {1:f} \n'.format( coords[0, inode] , coords[1, inode] ) )
        elif FE['dim'] == 3:
            fid.write(  '{0:f} {1:f} {2:f} \n'.format( coords[0, inode] , 
                coords[1, inode] , coords[2, inode] ) )

    # Write elements
    nnodes = 2**FE['dim']  # 4 for quads, 8 for hexas

    fid.write( "CELLS " + str(FE['n_elem'] ) + " " + \
            str( FE['n_elem']*(nnodes+1) ) + " \n" )

    # IMPORTANT! Vtk numbers nodes from 0, so we subtract 1
    for iel in range(0,FE['n_elem']):
        if FE['dim'] == 2:
            nel = 4
            fid.write( '{0:d} {1:d} {2:d} {3:d} {4:d} \n'.format( nel , FE['elem_node'][0, iel] ,
                FE['elem_node'][1, iel] , FE['elem_node'][2, iel] , FE['elem_node'][3, iel] ) )
        elif FE['dim'] == 3:
            nel = 8
            fid.write( '{0:d} {1:d} {2:d} {3:d} {4:d} {5:d} {6:d} {7:d} {8:d} \n'.format( nel , 
                FE['elem_node'][0, iel] , FE['elem_node'][1, iel] , FE['elem_node'][2, iel] ,
                FE['elem_node'][3, iel] , FE['elem_node'][4, iel] , FE['elem_node'][5, iel] , 
                FE['elem_node'][8, iel] , FE['elem_node'][7, iel] ) )
        

    # Write element types
    fid.write( "CELL_TYPES " + str(FE['n_elem']) + " \n" )

    if FE['dim'] == 2:
        elem_type = 9  # Corresponding to VTK_QUAD
    elif FE['dim'] == 3:
        elem_type = 12 # Corresponding to VTK_HEXAHEDRON
    
    for iel in range(0,FE['n_elem']):
        fid.write( '{} \n'.format( elem_type ) )

    # Write elemental densities
    fid.write( "CELL_DATA " + str(FE['n_elem']) + " \n" )
    fid.write( "SCALARS density float 1 \n" )
    fid.write( "LOOKUP_TABLE default \n" )
    for iel in range(0,FE['n_elem']):
        density = OPT['elem_dens'][iel]
        fid.write( '{0:f} \n'.format( density ) )

    fid.close()


def compute_compliance(FE,OPT,GEOM):
    # This function computes the mean compliance and its sensitivities
    # based on the last finite element analysis
    # global FE, OPT

    # compute the compliance (Eq. (15))
    c = np.dot( FE['U'].flatten() , FE['P'].toarray().flatten() )

    # compute the design sensitivity
    Ke = FE['Ke']
    Ue = FE['U'][FE['edofMat']].repeat(FE['n_edof'],axis=2).transpose((1,2,0))
    Ue_T = Ue.transpose((1,0,2))

    Dc_Dpenalized_elem_dens = np.sum( np.sum( 
        - Ue_T * Ke * Ue , 
        0 ) , 0 )   # Eq. (24)

    Dc_Ddv = Dc_Dpenalized_elem_dens @ OPT['Dpenalized_elem_dens_Ddv'] # Eq. (25)
    grad_c = Dc_Ddv.T
    # save these values in the OPT structure
    OPT['compliance'] = c
    OPT['grad_compliance'] = grad_c

    return c , grad_c


def compute_volume_fraction(FE,OPT,GEOM):
    #
    # This function computes the volume fraction and its sensitivities
    # based on the last geometry projection
    #
    # global FE, OPT

    # compute the volume fraction
    v_e = FE['elem_vol'] # element
    V = np.sum(v_e) # full volume
    v = np.dot( v_e , OPT['elem_dens'] ) # projected volume
    volfrac =  v/V # Eq. (16)

    # compute the design sensitivity
    Dvolfrac_Ddv = (v_e @ OPT['Delem_dens_Ddv'] )/V   # Eq. (31)
    grad_volfrac = Dvolfrac_Ddv
    
    # output
    OPT['volume_fraction'] = volfrac
    OPT['grad_volume_fraction'] = grad_volfrac

    return volfrac , grad_volfrac


def evaluate_relevant_functions(FE,OPT,GEOM):
    # Evaluate_relevant_functions() looks at OPT['functions'] and evaluates the
    # relevant functions for this problem based on the current OPT['dv']
    # global OPT

    OPT['functions']['n_func'] =  len( OPT['functions']['f'] )

    for i in range(0,OPT['functions']['n_func']):
        value , grad = eval( OPT['functions']['f'][i]['function'] + "(FE,OPT,GEOM)" )
        OPT['functions']['f'][i]['value'] = value
        OPT['functions']['f'][i]['grad'] = grad


def nonlcon(dv):
    global  FE, OPT, GEOM
    
    OPT['dv_old'] = OPT['dv'].copy()
    OPT['dv'] = dv.copy()
    
    # Update or perform the analysis
    if ( OPT['dv'] != OPT['dv_old'] ).any():
        update_geom_from_dv(FE, OPT, GEOM)
        perform_analysis(FE, OPT, GEOM)

    n_con   = OPT['functions']['n_func']-1 # number of constraints
    g       = np.zeros((n_con,1))

    for i in range(0,n_con):
        g[i] = OPT['functions']['f'][i+1]['value']
        g = g - OPT['functions']['constraint_limit']

    return g.flatten()


def nonlcongrad(dv):
    global  FE, OPT, GEOM

    n_con   = OPT['functions']['n_func']-1 # number of constraints
    gradg = np.zeros((OPT['n_dv'],n_con))

    for i in range(0,n_con):
        gradg[:,i] = OPT['functions']['f'][i+1]['grad']
    
    return gradg.flatten()
        

def obj(dv):
    global  FE, OPT, GEOM
    
    OPT['dv_old'] = OPT['dv'].copy() # save the previous design
    OPT['dv'] = dv.copy() # update the design
    
    # If different, update or perform the analysis
    if ( OPT['dv'] != OPT['dv_old'] ).any():
        update_geom_from_dv(FE,OPT,GEOM)
        perform_analysis(FE,OPT,GEOM)

    f = OPT['functions']['f'][0]['value'].flatten().copy()
    g = OPT['functions']['f'][0]['grad'].copy()

    return f, g


def perform_analysis(FE,OPT,GEOM):
    # Perform the geometry projection, solve the finite
    # element problem for the displacements and reaction forces, and then
    # evaluate the relevant functions.
    tic_proj = time.perf_counter()
    project_element_densities(FE,OPT,GEOM)
    toc_proj = time.perf_counter()
    GEOM['proj_t'] += toc_proj - tic_proj

    tic_FEA = time.perf_counter()
    FE_analysis(FE,OPT,GEOM)
    toc_FEA = time.perf_counter()
    FE['FEA_t'] += toc_FEA - tic_FEA

    tic_Fun = time.perf_counter()
    evaluate_relevant_functions(FE,OPT,GEOM)
    toc_Fun = time.perf_counter()
    OPT['Fun_t'] += toc_Fun - tic_Fun


def init_optimization(FE,OPT,GEOM):
    # Initialize functions to compute
    # Concatenate list of functions to be computed
    f_list = {}
    f_list[0] = OPT['functions']['objective']
    f_list[1] = OPT['functions']['constraints']

    # here we list all the functions that are available to compute as f{i}
    f = {}

    f[0] = {}
    f[0]['name'] = 'compliance'
    f[0]['function'] = 'compute_compliance'

    f[1] = {}
    f[1]['name'] = 'volume fraction'
    f[1]['function'] = 'compute_volume_fraction'

    # compare all functions available with the ones specified in inputs.m
    n = len(f)
    m = len(f_list)

    OPT['functions']['f'] = {}
    for j in range(0,m):
        for i in range(0,n):
            if f[i]['name'] == f_list[j]:
                OPT['functions']['f'][j] = f[i]

    OPT['functions']['n_func'] =  len(OPT['functions']['f'])

    ## initialize sample window size
    if not ('elem_r' in OPT['parameters']): 
        # compute sampling radius
        # The radius corresponds to the circle (or sphere) that circumscribes a
        # square (or cube) that has the edge length of elem_size.
        OPT['parameters']['elem_r'] = np.sqrt(FE['dim'])/2 * FE['elem_vol']**(1/FE['dim']) 
    
    ##
    # Initilize the design variable and its indexing schemes

    # we are designing the points, the size variables, and the radii of the
    # bars:

    OPT['n_dv'] = FE['dim']*GEOM['n_point'] + 2*GEOM['n_bar']
    OPT['dv']   = np.zeros( (OPT['n_dv'],1) )

    OPT['point_dv']  = np.arange(0,FE['dim']*GEOM['n_point']) # such that dv(point_dv) = point
    OPT['size_dv']   = OPT['point_dv'][-1] + 1 + np.arange(0,GEOM['n_bar'])
    OPT['radius_dv'] = OPT['size_dv'][-1] + 1 + np.arange(0,GEOM['n_bar'])

    if OPT['options']['dv_scaling']:
        OPT['scaling'] = {}
        # Compute variable limits for Eq. (32)
        OPT['scaling']['point_scale'] = FE['coord_max']-FE['coord_min']
        OPT['scaling']['point_min']   = FE['coord_min']
        # Consider possibility that max_bar_radius and min_bar_radius are
        # the same (when bars are of fixed radius)
        delta_radius = GEOM['max_bar_radius'] - GEOM['min_bar_radius']
        if delta_radius < 1e-12:
            OPT['scaling']['radius_scale'] = 1
        else:
            OPT['scaling']['radius_scale'] = delta_radius
        
        OPT['scaling']['radius_min'] = GEOM['min_bar_radius']
    else:
        OPT['scaling']['point_scale']  = 1.0
        OPT['scaling']['point_min']    = 0.0
        OPT['scaling']['radius_scale'] = 1.0
        OPT['scaling']['radius_min']   = 0.0

    # fill in design variable vector based on the initial design
    update_dv_from_geom(FE,OPT,GEOM)

    # set the current design to the initial design:
    GEOM['current_design'] = {}
    GEOM['current_design']['point_matrix'] = GEOM['initial_design']['point_matrix']
    GEOM['current_design']['bar_matrix'] = GEOM['initial_design']['bar_matrix']

    # consider the bar design variables
    # Extract index of first and secont point of each bar
    x_1b_id = GEOM['current_design']['bar_matrix'][:,1]
    x_2b_id = GEOM['current_design']['bar_matrix'][:,2]
    
    # Extract index of first (second) point of each matrix
    pt1 = GEOM['point_mat_row'][x_1b_id,0].toarray()
    pt2 = GEOM['point_mat_row'][x_2b_id,0].toarray()
    
    pt_dv = OPT['point_dv'].reshape((FE['dim'],GEOM['n_point']),order='F').copy()
    
    OPT['bar_dv'] = np.concatenate( ( pt_dv[:,pt1][:,:,0] , pt_dv[:,pt2][:,:,0] ,
        OPT['size_dv'].reshape((1,-1)) , OPT['radius_dv'].reshape((1,-1)) ) , axis = 0 ).copy()
    # print( OPT['bar_dv'] ) 


def runopt(FE,OPT,GEOM,x0,obj,nonlcon):
    # Perform the optimization using Scilab minimize with
    # constrained trust region or mma
    
    def plotfun(iter):
        if OPT['options']['plot'] == True:
            plt.figure(0)
            plot_design(0)
            plt.title( 'design, iteration = {iter}'.format(iter=iter) )

            if FE['dim'] == 3:
                plt.zlim( (FE['coord_min'][2], FE['coord_max'][2] ) )


            plt.figure(1)
            plot_density(1)

            stop = False
        return stop   

    history = {}

    # Design variables constraint
    if OPT['options']['dv_scaling']:   # Eq. (33)
        lb_point = np.zeros( (FE['dim'],1) )
        ub_point = np.ones( (FE['dim'],1) )
        lb_radius = 0

        # Consider case when max_bar_radius and min_bar_radius are
        # the same (when bars are of fixed radius)
        if GEOM['max_bar_radius'] - GEOM['min_bar_radius'] < 1e-12:
            ub_radius = 0
        else:
            ub_radius = 1
    else:
        lb_point = FE['coord_min']            # Eq. (18)
        ub_point = FE['coord_max']            # Eq. (18)
        lb_radius = GEOM['min_bar_radius']    # Eq. (19)
        ub_radius = GEOM['max_bar_radius']    # Eq. (19)

    lb_size = 0    # Eq. (20)
    ub_size = 1    # Eq. (20)

    lb_bar = np.vstack( ( lb_point , lb_point , np.array( (lb_size, lb_radius) )[:,None] ) ) 
    ub_bar = np.vstack( ( ub_point , ub_point , np.array( (ub_size, ub_radius) )[:,None] ) ) 

    lb = np.zeros( OPT['dv'].shape )
    ub = np.zeros( OPT['dv'].shape )

    lb[OPT['bar_dv']] = np.tile( lb_bar, (1,GEOM['n_bar']) )[:,:,None]
    ub[OPT['bar_dv']] = np.tile( ub_bar, (1,GEOM['n_bar']) )[:,:,None]

    # Optimization routines
    if  'default' == OPT['options']['optimizer']:
        def output(x,state):
            stop = False
            # print( state.status )
            print( "Iteration: " + str(state.nit) + \
                "\n\tCompliance: " + str(OPT['functions']['f'][0]['value']) +\
                "\n\tVolume fra: " + str(OPT['functions']['f'][1]['value']) )
            
            if state.nit == 1:
                history['fval']     = state['fun'][:,None]
                history['fconsval'] = state['constr'][0][:,None]
                history['x']        = x[:,None]
            else:
                # Concatenate current point and obj value with history
                history['fval']     = np.concatenate( ( history['fval'] , state['fun'][:,None] ) , axis = 1 )
                history['fconsval'] = np.concatenate( ( history['fconsval'] , state['constr'][0][:,None] ) , axis = 1 )
                history['x']        = np.concatenate( ( history['x'] , x[:,None] ) , axis = 1 ) # here we make x into a column vector

            folder, baseFileName = os.path.split(GEOM['initial_design']['path'] )
            mat_filename = folder + '/' + baseFileName + '.mat'
            savemat( mat_filename, GEOM )

            if OPT['options']['write_to_vtk'] == 'all':
                writevtk( OPT['options']['vtk_output_path'] , 'dens' , state.nit )

            plotfun(state.nit)

            return stop    

        # Initialize history object
        bounds = Bounds(lb.flatten(),ub.flatten())

        nonlinear_constraint = NonlinearConstraint(nonlcon,
            -np.inf, 0,
            jac=nonlcongrad)

        # This is the call to the optimizer
        res = minimize(obj,x0.flatten(),method='trust-constr',jac=True,
            constraints=nonlinear_constraint,bounds=bounds,
            options={'verbose': 1,'maxiter':OPT['options']['max_iter']},
            tol=OPT['options']['kkt_tol'],callback=output) 
        
        finalIt = res.nit
        
        # Plot
        plotfun(res.nit)

    elif 'mma' == OPT['options']['optimizer']:
        ncons = OPT['functions']['n_func'] - 1  # Number of optimization constraints
        ndv = OPT['n_dv'] # Number of design variables

        # Initialize vectors that store current and previous two design iterates
        x       = x0.copy()
        xold1   = x0.copy() 
        xold2   = x0.copy()

        # Initialize move limits 
        ml_step = OPT['options']['move_limit'] * abs(ub - lb)  # Compute move limits once

        # Initialize lower and upper asymptotes
        low = lb.copy()
        upp = ub.copy()

        # These are the MMA constants (Svanberg, 1998 DACAMM Course)
        c   = 1000*np.ones( (ncons,1) )
        d   = np.ones( (ncons,1) )
        a0  = 1
        a   = np.zeros( (ncons, 1) )

        # Evaluate the initial design and print values to screen 
        iter = 1
        f0val , df0dx = obj(x)
        fval = nonlcon(x)
        dfdx = nonlcongrad(x).T

        df0dx = df0dx[:,None]
        dfdx = dfdx[:,None].T

        print('It. ' + str(iter) + ', Obj= ' + str(f0val) +
            ', ConsViol = ' + str(max(max(fval, np.zeros((ncons,1))))) ) 

        # Save history
        history['fval']     = f0val[:,None]
        history['fconsval'] = fval[:,None]
        history['x']        = x[:,None]
                
        #### Initialize stopping values
        kktnorm         = 10*OPT['options']['kkt_tol']
        dv_step_change  = 10*OPT['options']['step_tol']

        # Plot 
        plotfun(0)

        ## MMA Loop
        while kktnorm > OPT['options']['kkt_tol'] and \
            iter < OPT['options']['max_iter'] and \
            dv_step_change > OPT['options']['step_tol']:

            iter = iter+1

            # Impose move limits by modifying lower and upper bounds passed to MMA, Eq. (33)
            mlb = np.maximum(lb, x - ml_step)
            mub = np.minimum(ub, x + ml_step)            

            #### Solve MMA subproblem for current design x
            xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = \
                mmasub(ncons,ndv,iter,x,mlb,mub,xold1,
                xold2, f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d,
                0.5)

            #### Updated design vectors of previous and current iterations
            xold2, xold1, x = xold1, x, xmma
            
            # Update function values and gradients
            f0val , df0dx = obj(x)
            fval = nonlcon(x)
            dfdx = nonlcongrad(x)
            
            df0dx = df0dx[:,None]
            dfdx = dfdx[:,None].T

            # Compute change in design variables
            # Check only after first iteration
            if iter > 1:
                dv_step_change = np.linalg.norm(x - xold1)
                if dv_step_change < OPT['options']['step_tol']:
                    print('Design step convergence tolerance satisfied.\n')
                
            
            if iter == OPT['options']['max_iter']:
                print('Reached maximum number of iterations.\n')
                
            
            # Compute norm of KKT residual vector
            residu, kktnorm, residumax = \
                kktcheck(ncons,ndv,xmma,ymma,zmma,lam,xsi,eta,mu,zet,s, \
                lb,ub,df0dx,fval,dfdx,a0,a,c,d)
            
            # Produce output to screen
            print('It. ' + str(iter) + ', Obj= ' + str(f0val) +
                ', ConsViol = ' + str(max(max(fval, np.zeros((ncons,1))))) )
            print( '\tKKT-norm = ' + str(kktnorm) + 'DV norm change ' + str(dv_step_change) )
                
            # Concatenate current point and obj value with history
            history['fval']     = np.concatenate( ( history['fval'] , f0val[:,None] ) , axis = 1 )
            history['fconsval'] = np.concatenate( ( history['fconsval'] , fval[:,None] ) , axis = 1 )
            history['x']        = np.concatenate( ( history['x'] , x[:,None] ) , axis = 1 ) # here we make x into a

            if OPT['options']['write_to_vtk'] == 'all':
                writevtk( OPT['options']['vtk_output_path'] , 'dens' , iter )

            # Plot current design
            plotfun(iter)

        finalIt = iter   

    if OPT['options']['write_to_vtk'] == 'all' or \
        OPT['options']['write_to_vtk'] == 'last':
        writevtk( OPT['options']['vtk_output_path'] , 
            'dens', finalIt )

    return history


def generate_mesh(FE,O_pt,GEOM):
    # This function generates a uniform quadrilateral or hexahedral mesh for 
    # rectangular or parallelepiped design regions respectively. 
    #
    # The two arguments needed (box_dimensions & elements_per_side) must have
    # been assigned in the FE['mesh_input structure prior to calling this
    # routine.
    #
    # box_dimensions is a vector (2 x 1 in 2D, 3 x 1 in 3D) with the dimensions
    #                of the design region.
    # elements_per_side is a vector of the same dimensions as box_dimensions
    #                   with the number of elements to be created in each
    #                   corresponding dimension.
    #
    # The function updates the necessary arrays in the global FE structure.
    # This function does not need modification from the user.
    #

    #global FE

    box_dimensions = FE['mesh_input']['box_dimensions']
    elements_per_side = FE['mesh_input']['elements_per_side']

    # A sanity check:
    if not len(box_dimensions) == len(elements_per_side):
        print('Inconsistent number of dimensions & elements per side.')

    if 3 == len(box_dimensions):
        FE['dim'] = 3
    elif 2 == len(box_dimensions):
        FE['dim'] = 2
    else:
        print('FE[\'mesh_input.dimensions\'] must be of length 2 or 3')

    ## create nodal coordinates
    n_i = elements_per_side + 1 # number of nodes in coord i

    x_i = {}
    for i in range(0,FE['dim']):
        x_i[i]  = np.linspace( 0 , box_dimensions[i] , num = n_i[i] )

    FE['n_elem'] = np.prod( elements_per_side[:] )
    FE['n_node'] = np.prod( elements_per_side[:]+1 )

    if 2 == FE['dim']:
        xx, yy      = np.meshgrid( x_i[0] , x_i[1] )
        node_coords = np.stack( ( xx.T.flatten(), yy.T.flatten() ) , axis = 1 ) 
    elif 3 == FE['dim']:
        xx, yy, zz  = np.meshgrid( x_i[0] , x_i[1] , x_i[2] )
        node_coords = np.stack( ( xx.T.flatten(), yy.T.flatten(), zz.T.flatten() ) , axis = 1 )

    ## define element connectivity
    nelx = elements_per_side[0]
    nely = elements_per_side[1]

    if 2 == FE['dim']:
        row= np.array(range(0,nely)).reshape(-1,1)
        col= np.array(range(0,nelx)).reshape(1,-1)

        n1 = row + col*(nely+1)
        n2 = row + (col+1)*(nely+1)
        n3 = n2 + 1
        n4 = n1 + 1
        elem_mat = np.stack( ( 
            n1.flatten(order='F'), 
            n2.flatten(order='F'), 
            n3.flatten(order='F'), 
            n4.flatten(order='F') ) )
    elif 3 == FE['dim']:
        nelz = elements_per_side[2]

        row = np.array(range(0,nely) ).reshape(-1,1,1)
        col = np.array(range(0,nelx) ).reshape(1,-1,1)
        pile= np.array(range(0,nelz) ).reshape(1,1,-1)

        n1 = row + col*(nely+1) + (pile)*(nelx+1)*(nely+1)
        n2 = row + (col+1)*(nely+1) + (pile)*(nelx+1)*(nely+1)
        n3 = n2 + 1
        n4 = n1 + 1
        n5 = n1 + (nelx+1)*(nely+1)
        n6 = n2 + (nelx+1)*(nely+1)
        n7 = n6 + 1
        n8 = n5 + 1
        elem_mat = np.stack( ( 
            n1.flatten(order='F'), 
            n2.flatten(order='F'), 
            n3.flatten(order='F'), 
            n4.flatten(order='F'), 
            n5.flatten(order='F'), 
            n6.flatten(order='F'), 
            n7.flatten(order='F'), 
            n8.flatten(order='F') ) )

    ## export the mesh to the FE object by updating relevant fields 
    FE['coords']    = node_coords[:,np.array(range(0,FE['dim']) )].T
    FE['elem_node'] = elem_mat  # 4 nodes for quads, 8 for hexas

    ## print to terminal details of mesh generation
    print('generated '  + str(FE['dim']) + 'D cuboid mesh with ' + str(FE['n_elem']) +
        ' elements & ' + str(FE['n_node']) + ' nodes.\n')


def compute_predefined_node_sets(FE,requested_node_set_list):
    #
    # This function computes the requested node sets & stores them as 
    # members of FE['node_set']['
    #
    # Input is a cell array of strings identifying the node sets to compute
    # e.g. {'T_edge','BR_pt'}
    #
    # this function predefines certain sets of nodes (requested by the user)
    # that you can use for convenience to define displacement boundary
    # conditions & forces.  IMPORTANT: they only make sense (in general) for
    # rectangular / cuboidal meshes, & you must be careful to use them &/or
    # change the code according to your needs.
    #
    # Note that an advantage of using these node sets is that you can use them
    # with meshes that have different element sizes (but same dimensions)
    #
    #--------------------------------------------------------------------------
    # 2D:
    #
    #  Edges:                  Points:
    #   -----Top (T)-----          TL-------MT-------TR
    #  |                 |          |                 |
    #  |                 |          |                 |   
    # Left (L)         Right (R)   ML                MR   | y
    #  |                 |          |                 |   |
    #  |                 |          |                 |   |
    #   ---Bottom (B)----          BL-------MB-------BR    ----- x
    #
    #--------------------------------------------------------------------------
    # 3D:
    #
    #  Faces:                                Edges:            
    #                     Back (K)     
    #               -----------------                   -------TK--------          
    #             /|                /|                /|                /|
    #            / |   Top (T)     / |              TL |              TR |
    #           /  |              /  |              /  LK             /  RK
    #          |-----------------|   |             |-------TF--------|   |
    # Left (L) |   |             |   | Right (R)   |   |             |   |
    #          |  / -------------|--/             LF  / -------BK----RF-/
    #          | /   Bottom (B)  | /               |BL               | BR 
    #          |/                |/                |/                |/
    #           -----------------                   -------BF--------
    #                Front (F)
    #
    #  Points:                                       
    #         TLK---------------TRK    For edge midpoints:       
    #         /|                /|       Add 'M' to the edge    
    #        / |               / |       notation, e.g.,           
    #       /  |              /  |       'MTK' is the midpoint    | y
    #     TLF---------------TRF  |       of edge 'TK'.            |
    #      |   |             |   |                                | 
    #      |  BLK------------|--BRK    For face midpoints:         ---- x    
    #      | /               | /          Add 'M' to the face    /    
    #      |/                |/           notation, e.g.,       / 
    #     BLF---------------BRF           'MT' is the midpoint   z
    #                                     of face 'T'.


    ## The user should not modify this function.

    ## determine which node sets to compute from input list
    msg = {'odd_n_elem':'The number of elements along a dimension ' + 
        'requesting a midpoint is odd,\nreturning em_pty list of nodes.'}

    nel_odd = np.mod( FE['mesh_input']['elements_per_side'][:],2 ) != 0

    FE['node_set'] = {}

    coord_x = FE['coords'][0,:]
    coord_y = FE['coords'][1,:]
    if FE['dim'] == 3:
        coord_z = FE['coords'][2,:]

    tol = max(abs(FE['coord_max'] - FE['coord_min']) )/1e6
    
    minX = FE['coord_min'][0] 
    maxX = FE['coord_max'][0] 
    avgX = (maxX - minX)/2
    
    minY = FE['coord_min'][1] 
    maxY = FE['coord_max'][1] 
    avgY = (maxY - minY)/2

    if FE['dim'] == 3:
        minZ = FE['coord_min'][2] 
        maxZ = FE['coord_max'][2] 
        avgZ = (maxZ - minZ)/2
    
    if FE['dim'] == 2:
        for i in range(0,len(requested_node_set_list) ):
            # == Edges ==
            if requested_node_set_list[i] == 'T_edge':
                FE['node_set']['T_edge'] = np.where( coord_y > maxY - tol )[0]
            elif requested_node_set_list[i] == 'B_edge':
                FE['node_set']['B_edge'] = np.where( coord_y < minY + tol )[0]
            elif requested_node_set_list[i] == 'L_edge': 
                FE['node_set']['L_edge'] = np.where( coord_x < minX + tol )[0]
            elif requested_node_set_list[i] == 'R_edge':
                FE['node_set']['R_edge'] = np.where( coord_x > maxX - tol )[0]
            # == Points ==
            elif requested_node_set_list[i] == 'BL_pt':
                FE['node_set']['BL_pt'] =\
                    np.where( ( coord_x < minX + tol ) & ( coord_y < minY + tol ) )[0]
            elif requested_node_set_list[i] == 'BR_pt':
                FE['node_set']['BR_pt'] =\
                    np.where( ( coord_x > maxX - tol ) & ( coord_y < minY + tol ) )[0]
            elif requested_node_set_list[i] == 'TR_pt':
                FE['node_set']['TR_pt'] =\
                    np.where( ( coord_x > maxX - tol ) & ( coord_y > maxY - tol ) )[0]
            elif requested_node_set_list[i] == 'TL_pt':
                FE['node_set']['TL_pt'] =\
                    np.where( ( coord_x < minX + tol ) & ( coord_y > maxY - tol ) )[0]
            # Note: the following ones only work if there is an even number of
            # elements on the corresponding sides, i.e., there is a node exactly in
            # the middle of the side.
            elif requested_node_set_list[i] == 'ML_pt':
                FE['node_set']['ML_pt'] =\
                    np.where( ( coord_x < minX + tol ) & ( coord_y > avgY - tol ) & 
                        ( coord_y < avgY + tol ) )[0]
                if nel_odd[1]: 
                    # # of y elements is odd
                    print(msg['odd_n_elem'])
                
            elif requested_node_set_list[i] == 'MR_pt':
                FE['node_set']['MR_pt'] =\
                    np.where( ( coord_x > maxX - tol ) & ( coord_y > avgY - tol ) \
                        & ( coord_y < avgY + tol ) )[0]
                if nel_odd[1]: 
                    # # of y elements is odd
                    print(msg['odd_n_elem'])
                
            elif requested_node_set_list[i] == 'MB_pt':
                FE['node_set']['MB_pt'] =\
                    np.where( ( coord_y < minY + tol ) & ( coord_x > avgX - tol ) \
                        & ( coord_x < avgX + tol ) )[0]
                if nel_odd[0]: # # of x elements is odd
                    print(msg['odd_n_elem'])
                
            elif requested_node_set_list[i] == 'MT_pt':
                FE['node_set']['MT_pt'] =\
                    np.where( ( coord_y > maxY - tol ) & ( coord_x > avgX - tol ) \
                        & ( coord_x < avgX + tol ) )[0]
                if nel_odd[0]: # # of x elements is odd
                    print(msg['odd_n_elem'])
                
                # Volume-center point
            elif requested_node_set_list[i] == 'C_pt':
                if nel_odd[0] or nel_odd[1]: # # of x or y elements is odd
                    print(msg['odd_n_elem'])    
            FE['node_set']['C_pt'] = np.where( ( coord_y > avgY - tol ) & ( coord_y < avgY + tol ) &
                ( coord_x > avgX - tol ) & ( coord_x < avgX + tol ) )[0]
        
    elif FE['dim'] == 3:
        for i in range(0,len(requested_node_set_list) ):
            # == Faces ==
            if requested_node_set_list[i] == 'T_face':
                FE['node_set']['T_face'] = np.where( coord_y > maxY - tol )[0]
            elif requested_node_set_list[i] == 'B_face':
                FE['node_set']['B_face'] = np.where( coord_y < minY + tol)[0]
            elif requested_node_set_list[i] == 'L_face':
                FE['node_set']['L_face'] = np.where( coord_x < minX + tol)[0]
            elif requested_node_set_list[i] == 'R_face':
                FE['node_set']['R_face'] = np.where( coord_x > maxX - tol )[0]
            elif requested_node_set_list[i] == 'K_face':
                FE['node_set']['K_face'] = np.where( coord_z < minZ + tol )[0]
            elif requested_node_set_list[i] == 'F_face':
                FE['node_set']['F_face'] = np.where( coord_z > maxZ - tol )[0]
            # == Edges ==
            elif requested_node_set_list[i] == 'TK_edge':
                FE['node_set']['TK_edge'] = np.where( ( coord_y > maxY - tol ) & ( coord_z < minZ + tol ) )[0]
            elif requested_node_set_list[i] == 'BK_edge':
                FE['node_set']['BK_edge'] = np.where( ( coord_y < minY + tol ) & ( coord_z < minZ + tol ) )[0]
            elif requested_node_set_list[i] == 'LK_edge':
                FE['node_set']['LK_edge'] = np.where( ( coord_x < minX + tol ) & ( coord_z < minZ + tol ) )[0]
            elif requested_node_set_list[i] == 'RK_edge':
                FE['node_set']['RK_edge'] = np.where( ( coord_x > maxX - tol ) & ( coord_z < minZ + tol ) )[0]
            elif requested_node_set_list[i] == 'TF_edge':
                FE['node_set']['TF_edge'] = np.where( ( coord_y > maxY - tol ) & ( coord_z > maxZ - tol ) )[0]
            elif requested_node_set_list[i] == 'BF_edge':
                FE['node_set']['BF_edge'] = np.where( ( coord_y < minY + tol ) & ( coord_z > maxZ - tol ) )[0]
            elif requested_node_set_list[i] == 'LF_edge':
                FE['node_set']['LF_edge'] = np.where( ( coord_x < minX + tol ) & ( coord_z > maxZ - tol ) )[0]
            elif requested_node_set_list[i] == 'RF_edge':
                FE['node_set']['RF_edge'] = np.where( ( coord_x > maxX - tol ) & ( coord_z > maxZ - tol ) )[0]
            elif requested_node_set_list[i] == 'TL_edge':
                FE['node_set']['TL_edge'] = np.where( ( coord_y > maxY - tol ) & ( coord_x < minX - tol ) )[0]
            elif requested_node_set_list[i] == 'TR_edge':
                FE['node_set']['TR_edge'] = np.where( ( coord_y > maxY - tol ) & ( coord_x > maxX - tol ) )[0]
            elif requested_node_set_list[i] == 'BL_edge':
                FE['node_set']['BL_edge'] = np.where( ( coord_y < minY + tol ) & ( coord_x < minX - tol ) )[0]
            elif requested_node_set_list[i] == 'BR_edge':
                FE['node_set']['BR_edge'] = np.where( ( coord_y < minY + tol ) & ( coord_x > maxX - tol ) )   [0]
            # == Points ==
            elif requested_node_set_list[i] == 'BLK_pt':
                FE['node_set']['BLK_pt'] = np.where( ( coord_x < minX + tol ) & ( coord_y < minY + tol ) &
                    ( coord_z < minZ + tol ) )[0]
            elif requested_node_set_list[i] == 'BRK_pt':
                FE['node_set']['BRK_pt'] = np.where( ( coord_x > maxX - tol ) & ( coord_y < minY + tol ) &
                    ( coord_z < minZ + tol ) )[0]
            elif requested_node_set_list[i] == 'TRK_pt':
                FE['node_set']['TRK_pt'] = np.where( ( coord_x > maxX - tol ) & ( coord_y > maxY - tol ) &
                    ( coord_z < minZ + tol ) )[0]
            elif requested_node_set_list[i] == 'TLK_pt':
                FE['node_set']['TLK_pt'] = np.where( ( coord_x < minX + tol ) & ( coord_y > maxY - tol ) &
                    ( coord_z < minZ + tol ) )[0]
            elif requested_node_set_list[i] == 'BLF_pt':
                FE['node_set']['BLF_pt'] = np.where( ( coord_x < minX + tol ) & ( coord_y < minY + tol ) &
                    ( coord_z > maxZ - tol ) )[0]
            elif requested_node_set_list[i] == 'BRF_pt':
                FE['node_set']['BRF_pt'] = np.where( ( coord_x > maxX - tol ) & ( coord_y < minY + tol ) &
                    ( coord_z > maxZ - tol ) )[0]
            elif requested_node_set_list[i] == 'TRF_pt':
                FE['node_set']['TRF_pt'] = np.where( ( coord_x > maxX - tol ) & ( coord_y > maxY - tol ) &
                    ( coord_z > maxZ - tol ) )[0]
            elif requested_node_set_list[i] == 'TLF_pt':
                FE['node_set']['TLF_pt'] = np.where( ( coord_x < minX + tol ) & ( coord_y > maxY - tol ) &
                    ( coord_z > maxZ - tol ) )[0]
            # *****
            # Note: the following ones only work if there is an even number of
            # elements on the corresponding sides, i.e., there is a node exactly in
            # the middle of the side.
            #
            # Mid-edge points
            elif requested_node_set_list[i] == 'MLK_pt':
                if nel_odd[1]: # # of y elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MLK_pt'] = np.where( ( coord_x < minX + tol ) & ( coord_y > avgY - tol ) \
                    & ( coord_y < avgY + tol ) & ( coord_z < minZ + tol ) )[0]
            elif requested_node_set_list[i] == 'MRK_pt':
                if nel_odd[1]: # # of y elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MRK_pt'] = np.where( ( coord_x > maxX - tol ) & ( coord_y > avgY - tol ) \
                    & ( coord_y < avgY + tol ) & ( coord_z < minZ + tol ) )[0]
            elif requested_node_set_list[i] == 'MBK_pt':
                if nel_odd[0]: # # of x elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MBK_pt'] = np.where( ( coord_y < minY + tol ) & ( coord_x > avgX - tol ) \
                    & ( coord_x < avgX + tol ) & ( coord_z < minZ + tol ) )[0]
            elif requested_node_set_list[i] == 'MTK_pt':
                if nel_odd[0]: # # of x elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MTK_pt'] = np.where( ( coord_y > maxY - tol ) & ( coord_x > avgX - tol ) \
                    & ( coord_x < avgX + tol ) & ( coord_z < minZ + tol ) )[0]
            elif requested_node_set_list[i] == 'MLF_pt':
                if nel_odd[1]: # # of y elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MLF_pt'] = np.where( ( coord_x < minX + tol ) & ( coord_y > avgY - tol ) \
                    & ( coord_y < avgY + tol ) & ( coord_z > maxZ - tol ) )[0]
            elif requested_node_set_list[i] == 'MRF_pt':
                if nel_odd[1]: # # of y elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MRF_pt'] = np.where( ( coord_x > maxX - tol ) & ( coord_y > avgY - tol ) \
                    & ( coord_y < avgY + tol ) & ( coord_z > maxZ - tol ) )[0]
            elif requested_node_set_list[i] == 'MBF_pt':
                if nel_odd[0]: # # of x elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MBF_pt'] = np.where( ( coord_y < minY + tol ) & ( coord_x > avgX - tol ) \
                    & ( coord_x < avgX + tol ) & ( coord_z > maxZ - tol ) )[0]
            elif requested_node_set_list[i] == 'MTF_pt':
                if nel_odd[0]: # # of x elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MTF_pt'] = np.where( ( coord_y > maxY - tol ) & ( coord_x > avgX - tol ) \
                    & ( coord_x < avgX + tol ) & ( coord_z > maxZ - tol ) )[0]  
            elif requested_node_set_list[i] == 'MBL_pt':
                if nel_odd[2]: # # of z elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MBL_pt'] = np.where( ( coord_x < minX + tol ) & ( coord_z > avgZ - tol ) \
                    & ( coord_z < avgZ + tol ) & ( coord_y < minY + tol ) )[0]
            elif requested_node_set_list[i] == 'MBR_pt':
                if nel_odd[2]: # # of z elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MBR_pt'] = np.where( ( coord_x > maxX - tol ) & ( coord_z > avgZ - tol ) \
                    & ( coord_z < avgZ + tol ) & ( coord_y < minY + tol ) )[0]
            elif requested_node_set_list[i] == 'MTL_pt':
                if nel_odd[2]: # # of z elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MTL_pt'] = np.where( ( coord_x < minX + tol ) & ( coord_z > avgZ - tol ) \
                    & ( coord_z < avgZ + tol ) & ( coord_y > maxY - tol ) )[0]
            elif requested_node_set_list[i] == 'MTR_pt':
                if nel_odd[2]: # # of z elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MTR_pt'] = np.where( ( coord_x > maxX - tol ) & ( coord_z > avgZ - tol ) \
                    & ( coord_z < avgZ + tol ) & ( coord_y > maxY - tol ) )[0]
            # Mid-face points
            elif requested_node_set_list[i] == 'MB_pt':
                if nel_odd[0] or nel_odd[2]: # # of x or z elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MB_pt'] = np.where( ( coord_y < minY + tol ) &
                ( coord_x > avgX - tol ) & ( coord_x < avgX + tol ) &
                ( coord_z > avgZ - tol ) & ( coord_z < avgZ + tol ) )[0]
            elif requested_node_set_list[i] == 'MT_pt':
                if nel_odd[0] or nel_odd[2]: # # of x or z elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MT_pt'] = np.where( ( coord_y > maxY - tol ) &
                    ( coord_x > avgX - tol ) & ( coord_x < avgX + tol ) &
                    ( coord_z > avgZ - tol ) & ( coord_z < avgZ + tol ) )[0]
            elif requested_node_set_list[i] == 'ML_pt':
                if nel_odd[1] or nel_odd[2]: # # of y or z elements is odd
                        print(msg['odd_n_elem'])
                
                FE['node_set']['ML_pt'] = np.where( ( coord_x < minX + tol ) &
                        ( coord_y > avgY - tol ) & ( coord_y < avgY + tol ) &
                        ( coord_z > avgZ - tol ) & ( coord_z < avgZ + tol ) )[0]
            elif requested_node_set_list[i] == 'MR_pt':
                if nel_odd[1] or nel_odd[2]: # # of y or z elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MR_pt'] = np.where( ( coord_x > maxX - tol ) &
                        ( coord_y > avgY - tol ) & ( coord_y < avgY + tol ) &
                        ( coord_z > avgZ - tol ) & ( coord_z < avgZ + tol ) )[0]
            elif requested_node_set_list[i] == 'MK_pt':
                if nel_odd[0] or nel_odd[1]: # # of x or y elements is odd
                    print(msg['odd_n_elem'])
                
                FE['node_set']['MK_pt'] = np.where( coord_z < minZ + tol &
                        ( coord_y > avgY - tol ) & ( coord_y < avgY + tol ) &
                        ( coord_x > avgX - tol ) & ( coord_x < avgX + tol ) )[0] 
            elif requested_node_set_list[i] == 'MF_pt':
                if nel_odd[0] or nel_odd[1]: # # of x or y elements is odd
                    print(msg['odd_n_elem'])

                FE['node_set']['MF_pt'] = np.where( coord_z > maxZ - tol &
                    ( coord_y > avgY - tol ) & ( coord_y < avgY + tol ) &
                    ( coord_x > avgX - tol ) & ( coord_x < avgX + tol ) )[0]
            # Volume-center point
            elif requested_node_set_list[i] == 'C_pt':
                if nel_odd[0] or nel_odd[1] or nel_odd[2]: # # of x,y or z elements is odd
                    print(msg['odd_n_elem'])                
                
                FE['node_set']['C_pt'] = np.where( coord_z < minZ + tol & ( coord_z > maxZ - tol ) &
                    ( coord_y > avgY - tol ) & ( coord_y < avgY + tol ) &
                    ( coord_x > avgX - tol ) & ( coord_x < avgX + tol ) )[0]


def init_geometry(FE,OPT,GEOM):
    #
    # Initialize GEOM structure with initial design
    #

    if not GEOM['initial_design']['restart']:
        exec( open(GEOM['initial_design']['path']).read() )
        
        # To use non contiguous numbers in the point_mat, we need to grab the
        # points whose ID matches the number specified by bar_mat. We achieve 
        # this via a map (sparse vector) between point_mat_rows and pt_IDs st
        # point_mat_row(point_ID) = row # of point_mat for point_ID        
        pt_IDs = GEOM['initial_design']['point_matrix'][:,0]

        GEOM['point_mat_row'] = sp.csc_matrix( (np.arange(0,pt_IDs.shape[0]),
            (pt_IDs.astype(int), np.zeros(pt_IDs.shape[0],dtype=int) ) ) )
    else:
        exec( open(GEOM['initial_design']['path']).read() )
        GEOM['initial_design']['point_matrix'] = GEOM['current_design']['point_matrix']
        GEOM['initial_design']['bar_matrix'] = GEOM['current_design']['bar_matrix']
    
    GEOM['n_point'] = np.size( GEOM['initial_design']['point_matrix'] , 0 )
    GEOM['n_bar']   = np.size( GEOM['initial_design']['bar_matrix'] , 0 )


def compute_bar_elem_distance(FE,OPT,GEOM):
    # global FE, GEOM, OPT

    tol = 1e-12

    n_elem = FE['n_elem']
    dim = FE['dim']
    n_bar = GEOM['n_bar']
    n_bar_dof = 2*dim

    # (dim,bar,elem)
    points = GEOM['current_design']['point_matrix'][:,1:].T

    x_1b = points.T.flatten()[OPT['bar_dv'][0:dim,:]] # (i,b) 
    x_2b = points.T.flatten()[OPT['bar_dv'][dim:2*dim,:]] # (i,b) 
    x_e = FE['centroids']                        # (i,1,e)

    a_b  = x_2b - x_1b
    l_b  = np.sqrt( np.sum( a_b**2 , 0 ) )  # length of the bars, Eq. (10)
    l_b[ np.where(l_b < tol) ] = 1          # To avoid division by zero
    a_b = np.divide( a_b , l_b )            # normalize the bar direction to unit vector, Eq. (11)

    x_e_1b = (x_e.T[:,None] - x_1b.T).swapaxes(0,2)               # (i,b,e) 
    x_e_2b = (x_e.T[:,None] - x_2b.T).swapaxes(0,2)                 # (i,b,e) 
    norm_x_e_1b = np.sqrt( np.sum( x_e_1b**2 , 0 ) )  # (1,b,e)
    norm_x_e_2b = np.sqrt( np.sum( x_e_2b**2 , 0 ) )   # (1,b,e) 

    l_be     = np.sum( x_e_1b * a_b[:,:,None] , 0 )                 # (1,b,e), Eq. (12)
    vec_r_be = x_e_1b - ( l_be.T * a_b[:,None] ).swapaxes(1,2)      # (i,b,e)
    r_be     = np.sqrt( np.sum( vec_r_be**2 , 0 ) )    # (1,b,e), Eq. (13)

    l_be_over_l_b = (l_be.T / l_b).T

    branch1 = l_be <= 0.0   # (1,b,e)
    branch2 = l_be_over_l_b >= 1   # (1,b,e)
    branch3 = np.logical_not( np.logical_or( branch1 , branch2 ) )    # (1,b,e)

    # Compute the distances, Eq. (14)
    dist = branch1 * norm_x_e_1b + \
        branch2 * norm_x_e_2b + \
        branch3 *  r_be         # (1,b,e)

    ## compute sensitivities
    Dd_be_Dx_1b = np.zeros( ( FE['dim'] , n_bar , n_elem ) )
    Dd_be_Dx_2b = np.zeros( ( FE['dim'] , n_bar , n_elem ) )

    d_inv = dist**(-1)           # This can rer a division by zero 
    d_inv[ np.isinf( d_inv ) ] = 0 # lies on medial axis, and so we now fix it

    ## The sensitivities below are obtained from Eq. (30)
    ## sensitivity to x_1b    
    Dd_be_Dx_1b = -x_e_1b * d_inv * branch1 + \
        -vec_r_be * d_inv * ( 1 - l_be_over_l_b ) * branch3
    
    Dd_be_Dx_2b = -x_e_2b * d_inv * branch2 + \
        -vec_r_be * d_inv * l_be_over_l_b * branch3

    ## assemble the sensitivities to the bar design parameters (scaled)
    Dd_be_Dbar_ends = np.concatenate((Dd_be_Dx_1b,Dd_be_Dx_2b),
        axis=0).transpose((1,2,0)) * \
        np.concatenate( ( OPT['scaling']['point_scale'] , OPT['scaling']['point_scale'] ) )
    # print( Dd_be_Dx_1b[:,1000:1005].transpose((2,0,1)) )
    # time.sleep(10)

    return dist , Dd_be_Dbar_ends


def penalize(*args):

    # [P, dPdx] = penalize(x, p, penal_scheme)
    #     penalize(x) assumes x \in [0,1] and decreases the intermediate values
    #
    #	  For a single input, the interpolation is SIMP with p = 3
    #
    #	  The optional second argument is the parameter value p.
    #
    #     The optional third argument is a string that indicates the way the 
    #	  interpolation is defined, possible values are:
    #       'SIMP'      : default 
    # 	  	'RAMP'      : 
    #

    # consider input
    n_inputs = len(args)
    x = args[0]
    if n_inputs == 1:
        # set the definition to be used by default.
        p = 3 
        penal_scheme = 'SIMP' 
    elif n_inputs == 2:
        p = args[1]
        penal_scheme = 'SIMP'
    elif n_inputs == 3:
        p = args[1]
        penal_scheme = args[2]
    
    # consider output
    ### not implemented

    # define def
    if penal_scheme == 'SIMP':    
        P    = x**p 
        dPdx = p * x**(p-1)
    elif penal_scheme == 'RAMP':
        P    = x / (1 + p*(1-x) )
        dPdx = (1+p) / (1 + p*(1-x) )**2
    else:
        print('Unidentified parameters')

    # compute the output
    return P , dPdx


def project_element_densities(FE,OPT,GEOM):
    # This def computes the combined unpenalized densities (used to
    # compute the volume) and penalized densities (used to compute the ersatz
    # material for the analysis) and saves them in the global variables
    # FE['elem_dens'] and FE['penalized_elem_dens'].  
    #
    # It also computes the derivatives of the unpenalized and penalized
    # densities with respect to the design parameters, and saves them in the
    # global variables FE['Delem_dens_Ddv'] and FE['Dpenalized_elem_dens_Ddv']. 
    #

    ##  Distances from the element centroids to the medial segment of each bar
    d_be , Dd_be_Dbar_ends = compute_bar_elem_distance(FE,OPT,GEOM)

    ## Bar-element projected densities
    r_b =  GEOM['current_design']['bar_matrix'][:,-1] # bar radii
    r_e =  OPT['parameters']['elem_r'] # sample window radius

    # X_be is \phi_b/r in Eq. (2).  Note that the numerator corresponds to
    # the signed distance of Eq. (8).
    X_be = ( r_b[:,None] - d_be ) / r_e[None,:]

    ## Projected density 
    # Initial definitions
    rho_be = np.zeros( (GEOM['n_bar'],FE['n_elem']) )
    Drho_be_Dx_be = np.zeros( (GEOM['n_bar'],FE['n_elem']) )
    # In the boundary
    inB = np.abs(X_be) < 1
    # Inside
    ins = 1 <= X_be
    rho_be[ins] = 1

    if FE['dim'] == 2:  # 2D  
        rho_be[inB] = 1 + ( X_be[inB]*np.sqrt( 1.0 - X_be[inB]**2 ) - np.arccos(X_be[inB]) ) / np.pi
        Drho_be_Dx_be[inB] = ( np.sqrt( 1.0 - X_be[inB]**2 )*2.0 ) / np.pi # Eq. (28)
        # rho_be = np.arctan(3*X_be)/np.pi + 0.5
        # Drho_be_Dx_be = 3/(np.pi*(1+9*X_be**2))
    elif FE['dim'] == 3:
        rho_be[inB] = ( (X_be[inB]-2.0)*(-1.0/4.0)*(X_be[inB]+1.0)**2 )
        Drho_be_Dx_be[inB] = ( X_be[inB]**2*(-3.0/4.0)+3.0/4.0 ) # Eq. (28)

    # Sensitivities of raw projected densities, Eqs. (27) and (29)
    Drho_be_Dbar_ends = ( Drho_be_Dx_be * -1/r_e * 
        Dd_be_Dbar_ends.transpose((2,0,1)) ).transpose((1,2,0))
    
    Drho_be_Dbar_radii  = OPT['scaling']['radius_scale'] * Drho_be_Dx_be * np.transpose(1/r_e)

    ## Combined densities
    # Get size variables    
    alpha_b = GEOM['current_design']['bar_matrix'][:,-2] # bar size

    # Without penalization:
    # ====================
    # X_be here is \hat{\rho}_b in Eq. (4) with the value of q such that
    # there is no penalization (e.g., q = 1 in SIMP).
    X_be = alpha_b[:,None] * rho_be

    # Sensitivities of unpenalized effective densities, Eq. (26) with
    # ?\partial \mu / \partial (\alpha_b \rho_{be})=1
    DX_be_Dbar_s = Drho_be_Dbar_ends * alpha_b[:,None,None]
    DX_be_Dbar_size = rho_be.copy()  
    DX_be_Dbar_radii = Drho_be_Dbar_radii * alpha_b[:,None]

    # Combined density of Eq. (5).
    rho_e, Drho_e_DX_be = smooth_max(X_be, \
        OPT['parameters']['smooth_max_param'], \
        OPT['parameters']['smooth_max_scheme'], \
        FE['material']['rho_min'] )
    
    # Sensitivities of combined densities, Eq. (25)
    Drho_e_Dbar_s = Drho_e_DX_be[:,:,None] * DX_be_Dbar_s
    Drho_e_Dbar_size = Drho_e_DX_be * DX_be_Dbar_size
    Drho_e_Dbar_radii = Drho_e_DX_be * DX_be_Dbar_radii

    # Stack together sensitivities with respect to different design
    # variables into a single vector per element
    Drho_e_Ddv = np.zeros(( FE['n_elem'], OPT['n_dv'] ))
    for b in range(0,GEOM['n_bar']):
        Drho_e_Ddv[:,OPT['bar_dv'][:,b]] = \
            Drho_e_Ddv[:,OPT['bar_dv'][:,b]] + np.concatenate( ( \
            Drho_e_Dbar_s[b,:,:].reshape( ( FE['n_elem'], 2*FE['dim'] )  , order='F' ) ,  \
            Drho_e_Dbar_size[b,:].reshape( ( FE['n_elem'], 1 ) ) ,  \
            Drho_e_Dbar_radii[b,:].reshape( ( FE['n_elem'], 1 ) ) ) , axis=1 )

    # With penalization:   
    # =================
    # In this case X_be *is* penalized (Eq. (4)).
    penal_X_be , Dpenal_X_be_DX_be  = penalize(X_be,\
        OPT['parameters']['penalization_param'],\
        OPT['parameters']['penalization_scheme'])
    
    # Sensitivities of effective (penalized) densities, Eq. (26)
    Dpenal_X_be_Dbar_s     = Dpenal_X_be_DX_be[:,:,None] * DX_be_Dbar_s
    Dpenal_X_be_Dbar_size  = Dpenal_X_be_DX_be * DX_be_Dbar_size
    Dpenal_X_be_Dbar_radii = Dpenal_X_be_DX_be * DX_be_Dbar_radii

    # Combined density of Eq. (5).    
    penal_rho_e , Dpenal_rho_e_Dpenal_X_be = smooth_max(penal_X_be,
            OPT['parameters']['smooth_max_param'],
            OPT['parameters']['smooth_max_scheme'],
            FE['material']['rho_min'])

    # Sensitivities of combined densities, Eq. (25)
    Dpenal_rho_e_Dbar_s     = Dpenal_rho_e_Dpenal_X_be[:,:,None] * Dpenal_X_be_Dbar_s
    Dpenal_rho_e_Dbar_size  = Dpenal_rho_e_Dpenal_X_be * Dpenal_X_be_Dbar_size
    Dpenal_rho_e_Dbar_radii = Dpenal_rho_e_Dpenal_X_be * Dpenal_X_be_Dbar_radii
    
    # Sensitivities of projected density
    Dpenal_rho_e_Ddv = np.zeros( ( FE['n_elem'], OPT['n_dv'] ) )

    # Stack together sensitivities with respect to different design
    # variables into a single vector per element
    for b in range(0,GEOM['n_bar']):
        Dpenal_rho_e_Ddv[:,OPT['bar_dv'][:,b]] = \
            Dpenal_rho_e_Ddv[:,OPT['bar_dv'][:,b]] + np.concatenate( \
                ( Dpenal_rho_e_Dbar_s[b,:,:].reshape( ( FE['n_elem'], 2*FE['dim'] ) , order='F' ) ,  \
                Dpenal_rho_e_Dbar_size[b,:].reshape( ( FE['n_elem'], 1 ) ) ,  \
                Dpenal_rho_e_Dbar_radii[b,:].reshape( ( FE['n_elem'], 1 ) ) ) , \
                axis = 1 )

    ## Write the element densities and their sensitivities to OPT 
    OPT['elem_dens'] = rho_e
    OPT['Delem_dens_Ddv'] = Drho_e_Ddv
    OPT['penalized_elem_dens'] = penal_rho_e
    OPT['Dpenalized_elem_dens_Ddv'] = Dpenal_rho_e_Ddv


def smooth_max(x,p,form_def,x_min):
    #
    # This def computes a smooth approximation of the maximum of x.  The
    # type of smooth approximation (listed below) is given by the argument
    # form_def, and the corresponding approximation parameter is given by p.
    # x_min is a lower bound to the smooth approximation for the modified
    # p-norm and modified p-mean approximations.
    #
    #
    #     The optional third argument is a string that indicates the way the 
    #	  approximation is defined, possible values are:
    # 		'mod_p-norm'   : overestimate using modified p-norm (supports x=0)
    # 		'mod_p-mean'   : underestimate using modified p-norm (supports x=0)
    #		'KS'           : Kreisselmeier-Steinhauser, overestimate
    #		'KS_under'     : Kreisselmeier-Steinhauser, underestimate
    #

    if form_def == 'mod_p-norm':
        # Eq. (6)
        # in this case, we assume x >= 0 
        S = ( x_min**p + (1-x_min**p)*np.sum(x**p,axis=0) )**(1/p)
        dSdx = (1-x_min**p)*(x/S)**(p-1)

    elif form_def == 'mod_p-mean':
        # in this case, we assume x >= 0 
        N = np.size(x,0)
        S = ( x_min**p + (1-x_min**p)*np.sum(x**p,axis=0)/N )**(1/p)
        dSdx = (1-x_min**p)*(1/N)*(x/S)**(p-1)         

    elif form_def == 'KS':
        epx = np.exp(x)
        S = x_min + (1-x_min)*np.log( np.sum(epx(x),axis=0) )/p
        dSdx = (1-x_min)*np.epx( p*x )/np.sum(epx(x),axis=0)

    elif form_def == 'KS_under':
        # note: convergence might be fixed with Euler-Gamma
        N = x.shape[0]
        S = x_min + (1-x_min)*np.log( np.sum( np.exp(x) ,axis=0) /N) / p 
        dSdx = (1-x_min)*np.exp( p*x ) / np.sum(np.exp(x),axis=0)
    else:
        print('\nsmooth_max received invalid form_def.\n')
    
    return S, dSdx


def update_dv_from_geom(FE,OPT,GEOM):
    #
    # This def updates the values of the design variables (which will be
    # scaled if OPT.options['dv']_scaling is true) based on the unscaled bar 
    # geometric parameters. It does the opposite from the def 
    # update_geom_from_dv.
    #

    # global GEOM, OPT 

    # Fill in design variable vector based on the initial design
    # Eq. (32
    OPT['dv'][ OPT['point_dv'],0 ] = ( ( GEOM['initial_design']['point_matrix'][:,1:] -
        OPT['scaling']['point_min'] ) / OPT['scaling']['point_scale'] ).flatten()
    
    OPT['dv'][ OPT['size_dv'],0 ] = GEOM['initial_design']['bar_matrix'][:,-2].copy()

    OPT['dv'][ OPT['radius_dv'],0 ] = ( GEOM['initial_design']['bar_matrix'][:,-1] \
        - OPT['scaling']['radius_min'] ) / OPT['scaling']['radius_scale'] 


def update_geom_from_dv(FE,OPT,GEOM):
    # This def updates the values of the unscaled bar geometric parameters
    # from the values of the design variableds (which will be scaled if
    # OPT.options['dv']_scaling is true). It does the
    # opposite from the def update_dv_from_geom.
    #
    # global GEOM , OPT , FE

    # Eq. (32)
    GEOM['current_design']['point_matrix'][:,1:] = ( OPT['scaling']['point_scale'][:,None] * \
            OPT['dv'][ OPT['point_dv'] ].reshape( (FE['dim'],GEOM['n_point']) ,order='F') + \
            OPT['scaling']['point_min'][:,None] ).T

    GEOM['current_design']['bar_matrix'][:,-2] = OPT['dv'][OPT['size_dv']].copy().flatten()

    GEOM['current_design']['bar_matrix'][:,-1] = ( OPT['dv'][OPT['radius_dv']] * \
        OPT['scaling']['radius_scale'] + \
        OPT['scaling']['radius_min'] ).flatten()


def FE_analysis(FE,OPT,GEOM):
    # Assemble the Global stiffness matrix and solve the FEA
    # Assemble the stiffness matrix partitions Kpp Kpf Kff
    FE_assemble_stiffness_matrix(FE,OPT,GEOM)
    # Solve the displacements and reaction forces
    FE_solve(FE,OPT,GEOM)
    FE['FEA_n'] += 1


def FE_assemble_BC(FE,OPT,GEOM):
    # FE_ASSEMBLE_BC assembles the boundary conditions the known portions of 
    # the load vector and displacement vector.
    ## Reads: FE['n_global_dof'], dim, BC
    ## Writes: FE['U'], P

    ## Declare global variables
    ## global FE

    ## Assemble prescribed displacements
    # Inititialize a sparse global displacement vector
    FE['U'] = np.zeros( ( FE['n_global_dof'] , 1 ) )

    # determine prescribed xi displacement components:
    for idisp in range( 0 , FE['BC']['n_pre_disp_dofs'] ):
        idx = FE['dim'] * FE['BC']['disp_node'][idisp] + FE['BC']['disp_dof'][idisp]
        FE['U'][idx] = FE['BC']['disp_value'][idisp]
    
    ## Assemble prescribed loads
    # initialize a sparse global force vector
    
    FE['P'] = sp.csc_matrix( ( FE['n_global_dof'],1) )

    # determine prescribed xi load components:
    for iload in range( 0 , FE['BC']['n_pre_force_dofs'] ):
        idx = FE['dim'] * FE['BC']['force_node'][iload] + \
            FE['BC']['force_dof'][iload]
        FE['P'][idx] = FE['BC']['force_value'][iload]


def FE_assemble_stiffness_matrix(FE,OPT,GEOM):
    # FE_ASSEMBLE assembles the global stiffness matrix, partitions it by 
    # prescribed and free dofs, and saves the known partitions in the FE structure.

    ## Reads: FE['iK'], jK, sK, fixeddofs_ind, freedofs_ind
    # OPT['penalized_elem_dens']
    ## Writes: FE['Kpp'], Kpf, Kff

    ## Declare global variables
    # global FE, OPT

    ## assemble and partition the global stiffness matrix
    # Retrieve the penalized stiffness
    penalized_rho_e = np.tile( OPT['penalized_elem_dens'][:,None,None] , 
        (1,FE['n_edof'],FE['n_edof']) ).transpose((1,2,0))
            
    # Ersatz material: (Eq. (7))
    penalized_Ke = penalized_rho_e * FE['Ke']
    FE['sK_penal'] = penalized_Ke.flatten('F')

    # assemble the penalized global stiffness matrix
    K = sp.csc_matrix( ( FE['sK_penal'] , ( FE['iK'] , FE['jK'] ) ) )

    # partition the stiffness matrix and return these partitions to FE
    FE['Kpp'] = K[FE['fixeddofs_ind'],:][:,FE['fixeddofs_ind']]
    FE['Kfp'] = K[FE['freedofs_ind'] ,:][:,FE['fixeddofs_ind']]

    # note: by symmetry Kpf = Kfp', so we don't store Kpf. Tall and thin
    # matrices are stored more efficiently as sparse matrices, and since we
    # generally have more free dofs than fixed, we choose to store the rows as
    # free dofs to save on memory.
    FE['Kff'] = K[FE['freedofs_ind'] ,:][:,FE['freedofs_ind']]


def FE_compute_constitutive_matrices(FE,OPT,GEOM):
    # Compute elasticity matrix for given elasticity modulus and Poisson's ratio
    # global FE

    # compute the elastic matrix for the design-material 
    C = FE_compute_constitutive_matrix( FE,OPT,GEOM, FE['material']['E'] , FE['material']['nu'])

    FE['material']['C'] = C


def FE_compute_constitutive_matrix(FE,OPT,GEOM,E,nu):
    # global FE

    # Compute constitutive matrix  
    if 2 == FE['dim']:
        # Plane Stress ( For thin out of plane dimension)
        a = E/(1-nu**2) 
        b = nu 
        c = (1-nu)/2 

        # Plane Strain ( For thick out of plane dimension)
        #     a=E*(1 - nu)/((1 + nu)*(1 - 2*nu)) b=nu/(1-nu) c=(1-2*nu)/(2*(1-nu)) 
        C = a * np.array( ( ( 1 , b , 0 ) , ( b , 1 , 0 ) , ( 0 , 0 , c ) ) )
    
    elif 3 == FE['dim']:
        a = E/( (1+nu) * (1-2*nu) )
        b = nu
        c = 1-nu
        d = (1-2*nu)/2

        C = a * np.array( 
            ( ( c , b , b , 0 , 0 , 0 ) ,
            ( b , c , b , 0 , 0 , 0 ) ,
            ( b , b , c , 0 , 0 , 0 ) ,
            ( 0 , 0 , 0 , d , 0 , 0 ) ,
            ( 0 , 0 , 0 , 0 , d , 0 ) ,
            ( 0 , 0 , 0 , 0 , 0 , d ) ) )
    

    C = 0.5 * (C + np.transpose(C))
    
    return C


def FE_compute_element_info(FE,OPT,GEOM):
    # This function computes element volumes, element centroid locations and
    # maximum / minimum nodal coordinates values for the mesh. 
    # It assumes the FE structure has already been populated.
    # global FE

    dim = FE['dim']
    CoordArray = np.zeros( ( 2**dim , dim , FE['n_elem'] ) )
    FE['elem_vol'] = np.zeros( ( FE['n_elem'] , 1 ) )

    # Node, dimension, coord
    for n in range( 0 , FE['n_elem'] ):
        CoordArray[:,:,n]= FE['coords'][:,FE['elem_node'][:,n]].T
    
    FE['centroids'] = np.mean(CoordArray,axis=0)

    # Create arrays with nodal coordinates for all nodes in an element e.g., 
    # n1[:,e) is the array of coordinates of node 1 for element e. Then use these to
    # compute the element volumes using the vectorized vector-functions `cross'
    # and `dot'.
    #
    n1 = CoordArray[0,:,:].T
    n2 = CoordArray[1,:,:].T
    n3 = CoordArray[2,:,:].T
    n4 = CoordArray[3,:,:].T
    if 3 == dim:
        n5 = CoordArray[4,:,:].T
        n6 = CoordArray[5,:,:].T
        n7 = CoordArray[6,:,:].T
        n8 = CoordArray[7,:,:].T
        
        # In the general case where the hexahedron is not a parallelepiped and 
        # does not necessarily have parallel sides, we use 3 scalar triple 
        # products to compute the volume of the Tetrakis Hexahedron.
        #     ( see J. Grandy, October 30, 1997,
        #       Efficient Computation of Volume of Hexahedral Cells )
        FE['elem_vol'] = ( \
            np.sum( ( (n7-n2) + (n8-n1) ) * np.cross( (n7-n4)          , (n3-n1)           ) , axis=1 ) + \
            np.sum( (n8-n1)               * np.cross( (n7-n4) + (n6-n1), (n7-n5)           ) , axis=1  ) + \
            np.sum( (n7-n2)               * np.cross( (n6-n1)          , (n7-n5) + (n3-n1) ) , axis=1  ) \
        )/12 
   
    elif 2 == dim:
        # Here we can take advantage of the planar quadrilaterals and use
        # Bretschneider's formula (cross product of diagonals):
        diag1 = np.zeros( np.shape(n1) ) 
        diag1[:,0:2] = n3 - n1

        diag2 = np.zeros( np.shape(n1) )
        diag2[:,0:2] = n4 - n2

        # Note: since only the 3rd componenet is nonzero, we use it instead 
        # of, 0.5*sqrt(sum(np.cross(diag1,diag2)**2)).T
        v1 = np.cross( diag1 , diag2 )
        FE['elem_vol'] = 0.5 * np.abs( v1[:] ) 

    FE['coord_max'] = np.amax( FE['coords'],axis=1 )
    FE['coord_min'] = np.amin( FE['coords'],axis=1 ) 


def jacobian( FE,OPT,GEOM, xi , eta , argA , *argB ):
    # global FE
    # Jacobian matrix
    if 0 == len( argB ): 
        elem = argA

        grad = gradient( FE,OPT,GEOM, xi , eta )
        jac = np.matmul( grad , FE['coords'][:,FE['elem_node'][:,elem]].T )

    elif 1 == len( argB ):
        zeta = argA
        elem = argB[0]

        grad = gradient( FE,OPT,GEOM, xi , eta , zeta )
        jac = np.matmul( grad , FE['coords'][:,FE['elem_node'][:,elem]].T )

    return jac


def gradient( FE,OPT,GEOM, xi , eta , *zeta ):
    # Gradient of shape function matrix in parent coordinates
    
    if 0 == len( zeta ):
        grad = np.array( ( ( eta-1 , xi-1 ) , ( 1-eta , -xi-1 ) , \
            ( 1+eta , 1+xi ) , ( -eta-1 , 1-xi ) ) ).T
        grad *= 0.25
    
    if 1 == len( zeta ):
        zeta = zeta[0]

        grad = np.array( ( ( -(1-zeta)*(1-eta) , -(1-zeta)*(1-xi) , -(1-eta)*(1-xi) ) ,
            ( (1-zeta)*(1-eta) , -(1-zeta)*(1+xi) , -(1-eta)*(1+xi) ) ,
            (  (1-zeta)*(1+eta) , (1-zeta)*(1+xi) , -(1+eta)*(1+xi) ) , \
            ( -(1-zeta)*(1+eta) , (1-zeta)*(1-xi) , -(1+eta)*(1-xi) ) , \
            ( -(1+zeta)*(1-eta) , -(1+zeta)*(1-xi) , (1-eta)*(1-xi) ) , \
            (  (1+zeta)*(1-eta) , -(1+zeta)*(1+xi) , (1-eta)*(1+xi) ) , \
            (  (1+zeta)*(1+eta) ,  (1+zeta)*(1+xi) , (1+eta)*(1+xi) ) , \
            ( -(1+zeta)*(1+eta) , (1+zeta)*(1-xi) ,  (1+eta)*(1-xi) ) ) ).T
        grad *= 0.125

    return grad


def B( GN ):
    if 2 == GN.shape[0]:
        B_ = np.array( ( ( GN[0,0] , 0 ,  GN[0,1] , 0 , GN[0,2] , 0 , GN[0,3] , 0 ) , \
            ( 0       , GN[1,0] , 0       , GN[1,1] , 0       , GN[1,2] , 0       , GN[1,3] ) , \
            ( GN[1,0] , GN[0,0] , GN[1,1] , GN[0,1] , GN[1,2] , GN[0,2] , GN[1,3] , GN[0,3] ) ) )
    elif 3 == GN.shape[0]:
        B_ = np.array( ( ( GN[0,0] , 0 , 0 , GN[0,1] , 0       , 0       , GN[0,2] , 0       , 0       , GN[0,3] , 0       , 0       , GN[0,4] , 0       , 0       , GN[0,5] , 0       , 0       , GN[0,6] , 0       , 0       , GN[0,7] , 0       , 0       ) , \
            ( 0       , GN[1,0] , 0       , 0       , GN[1,1] , 0       , 0       , GN[1,2] , 0       , 0       , GN[1,3] , 0       , 0       , GN[1,4] , 0       , 0       , GN[1,5] , 0       , 0       , GN[1,6] , 0       , 0       , GN[1,7] , 0       ) , \
            ( 0       , 0       , GN[2,0] , 0       , 0       , GN[2,1] , 0       , 0       , GN[2,2] , 0       , 0       , GN[2,3] , 0       , 0       , GN[2,4] , 0       , 0       , GN[2,5] , 0       , 0       , GN[2,6] , 0       , 0       , GN[2,7] ) , \
            ( GN[1,0] , GN[0,0] , 0       , GN[1,1] , GN[0,1] , 0       , GN[1,2] , GN[0,2] , 0       , GN[1,3] , GN[0,3] , 0       , GN[1,4] , GN[0,4] , 0       , GN[1,5] , GN[0,5] , 0       , GN[1,6] , GN[0,6] , 0       , GN[1,7] , GN[0,7] , 0       ) , \
            ( 0       , GN[2,0] , GN[1,0] , 0       , GN[2,1] , GN[1,1] , 0       , GN[2,2] , GN[1,2] , 0       , GN[2,3] , GN[1,3] , 0       , GN[2,4] , GN[1,4] , 0       , GN[2,5] , GN[1,5] , 0       , GN[2,6] , GN[1,6] , 0       , GN[2,7] , GN[1,7] ) , \
            ( GN[2,0] , 0       , GN[0,0] , GN[2,1] , 0       , GN[0,1] , GN[2,2] , 0       , GN[0,2] , GN[2,3] , 0       , GN[0,3] , GN[2,4] , 0       , GN[0,4] , GN[2,5] , 0       , GN[0,5] , GN[2,6] , 0       , GN[0,6] , GN[2,7] , 0       , GN[0,7] ) ) )

    return B_


def FE_compute_element_stiffness(FE,OPT,GEOM,C):
    # This function computes the element stiffness matrix fo all elements given
    # an elasticity matrix.
    # It computes the 'fully-solid' (i.e., unpenalized) matrix.
    # global FE
    
    ## Solid Stiffness Matrix Computation
    # We will use a 2 point quadrature to integrate the stiffness matrix:
    gauss_pt = np.array((-1,1))/np.sqrt(3)
    W = np.array((1,1,1))
    num_gauss_pt = len(gauss_pt)

    # inititalize element stiffness matrices
    Ke = np.zeros( ( FE['dim']*2**FE['dim'] , FE['dim']*2**FE['dim'] , FE['n_elem'] ) )
    # loop over elements, any element with a negative det_J will be flagged
    bad_elem = 0 != np.zeros( ( FE['n_elem'] , 1 ) )

    for e in range( 0 , FE['n_elem'] ):
        # loop over Gauss Points
        for i in range(0,num_gauss_pt):
            xi = gauss_pt[i]
            for j in range(0,num_gauss_pt):
                eta = gauss_pt[j]

                if 2 == FE['dim']:
                    # Compute Jacobian
                    J       = jacobian(FE,OPT,GEOM,xi,eta,e)
                    det_J   = np.linalg.det(J)
                    inv_J   = np.linalg.inv(J)

                    # Compute shape function derivatives (strain displacement matrix)  
                    GN = inv_J @ gradient(FE,OPT,GEOM,xi,eta)
                    B_ = B(GN)  

                    Ke[:,:,e] = Ke[:,:,e] + W[i] * W[j] * det_J * B_.T @ C @ B_
                
                elif 3 == FE['dim']:
                    for k in range(0,num_gauss_pt):
                        zeta    = gauss_pt[k]

                        # Compute Jacobian
                        J       = jacobian( FE,OPT,GEOM,xi,eta,zeta,e)
                        det_J   = np.linalg.det(J)
                        inv_J   = np.linalg.inv(J)
                        
                        # Compute shape function derivatives (strain displacement matrix)  
                        GN = inv_J @ gradient( FE,OPT,GEOM,xi,eta,zeta)
                        B_ = B(GN)
                        
                        Ke[:,:,e] = Ke[:,:,e] + W[i] * W[j] * W[k] * det_J * B_.T @ C @ B_

        if det_J < 0:
            bad_elem[e] = True 

    if 0 < np.sum(bad_elem):
        #print('The following elements have nodes in the wrong order:\n#s',sprintf('#i\n',find(bad_elem))) 
        print( 'There are elements in the wrong order' )
    
    sK = Ke.flatten()
    return sK


def FE_init_element_stiffness(FE,OPT,GEOM):
    # This function computes FE['sK_void, the vector of element 
    # stiffess matrix entries for the void material.
    # global FE

    ## Void Stiffness Matrix Computation
    n_edof = FE['n_edof']

    FE['Ke'] = FE_compute_element_stiffness( FE,OPT,GEOM,FE['material']['C'] ).reshape(
            ( n_edof, n_edof, FE['n_elem'] ) )


def FE_init_partitioning(FE,OPT,GEOM):
    # Partition finite element matrix and RHS vector for solution
    # global FE

    FE['n_global_dof'] = FE['dim'] * FE['n_node']

    FE['fixeddofs'] = 0 != np.zeros((FE['n_global_dof'],1))

    if 2 == FE['dim']:
        FE['fixeddofs'][ 2*FE['BC']['disp_node'][ FE['BC']['disp_dof'] == 0 ] ] = True # set prescribed x1 DOFs 
        FE['fixeddofs'][ 2*FE['BC']['disp_node'][ FE['BC']['disp_dof'] == 1 ] + 1 ] = True   # set prescribed x2 DOFs 
    elif 3 == FE['dim']:
        FE['fixeddofs'][ 3*FE['BC']['disp_node'][ FE['BC']['disp_dof'] == 0 ] ] = True # set prescribed x1 DOFs 
        FE['fixeddofs'][ 3*FE['BC']['disp_node'][ FE['BC']['disp_dof'] == 1 ] + 1 ] = True # set prescribed x2 DOFs 
        FE['fixeddofs'][ 3*FE['BC']['disp_node'][ FE['BC']['disp_dof'] == 2 ] + 2 ] = True   # set prescribed x3 DOFs 
        
    FE['freedofs'] = np.logical_not( FE['fixeddofs'] )

    # by calling find() once to these functions to get the indices, the 
    # overhead of logical indexing may be removed.
    FE['fixeddofs_ind'] = np.where( FE['fixeddofs'] )[0]
    FE['freedofs_ind']  = np.where( FE['freedofs'] )[0]

    FE['n_free_dof']    = FE['freedofs_ind'].shape[0] # the number of free DOFs

    # To vectorize the assembly for the global system in
    # the FEA, the method employed in the 88-lines paper is adopted: 

    n_elem_dof = FE['dim'] * 2**FE['dim']
    FE['n_edof'] = n_elem_dof

    n, m = np.shape( FE['elem_node'] )
    FE['edofMat'] = np.zeros( ( m , n*FE['dim'] ), dtype=int )

    for elem in range(0,m):
        enodes = FE['elem_node'][:,elem]
        if 2 == FE['dim']:
            edofs = np.stack( ( 2*enodes , 2*enodes+1 ) , axis=1 )\
                .reshape( ( 1 , n_elem_dof ) )
        elif 3 == FE['dim']:
            edofs = np.stack( ( 3*enodes , 3*enodes+1 , 3*enodes+2 ) , axis=1 )\
                .reshape( ( 1 , n_elem_dof ) )
        
        FE['edofMat'][elem,:] = edofs

    FE['iK'] = np.kron( FE['edofMat'] , np.ones((n_elem_dof,1),dtype=int) ).T\
        .reshape( FE['n_elem']*n_elem_dof**2 , order ='F' )
    FE['jK'] = np.kron( FE['edofMat'] , np.ones((1,n_elem_dof),dtype=int) ).T\
        .reshape( FE['n_elem']*n_elem_dof**2 , order ='F' )


def FE_solve(FE,OPT,GEOM):
    # This function solves the system of linear equations arising from the
    # finite element discretization of Eq. (17).  It stores the displacement 
    # and the reaction forces in FE['U'] and FE['P'].

    # global FE
    p = FE['fixeddofs_ind']
    f = FE['freedofs_ind']


    # save the system RHS
    FE['rhs'] = FE['P'][f] - FE['Kfp'] @ FE['U'][p]

    if 'direct' == FE['analysis']['solver']['type']:
        if FE['analysis']['solver']['use_gpu'] == True:
            print('GPU solver selected, but only available for iterative solver, solving on CPU.')
        FE['analysis']['solver']['use_gpu'] = False
        
        FE['U'][f] = linalg.spsolve( FE['Kff'] , FE['rhs'] )[:,None]

    elif 'iterative' == FE['analysis']['solver']['type']:
        tol = FE['analysis']['solver']['tol']
        maxit = FE['analysis']['solver']['maxit']
    
        msg = []

        # LU incomplete preconditioner
        # x0 = linalg.spsolve( sp.diags(FE['Kff'].diagonal(),0) , FE['rhs'])

        solution, __ = linalg.cg(FE['Kff'],FE['rhs'],
            x0=FE['U'][f],tol=tol,
            # x0=x0,tol=tol,
            maxiter=maxit)
            
        FE['U'][f] = solution[:,None] 
        # print(msg)

    # solve the reaction forces:
    FE['P'][p] = FE['Kpp'] @ FE['U'][p] + FE['Kfp'].T @ FE['U'][f]


def init_FE(FE,OPT,GEOM):
    # Initialize the FE structure
    # global FE

    if 'generate' ==  FE['mesh_input']['type']:
        generate_mesh(FE,OPT,GEOM)
    elif 'read-home-made' ==  FE['mesh_input']['type']:
        load( FE['mesh_input'].mesh_filename )
    elif 'read-gmsh' == FE['mesh_input']['type']:
        read_gmsh(FE,OPT,GEOM)
    else:
        print('Unidentified mesh type')


    # Compute element volumes and centroidal coordinates
    FE_compute_element_info(FE,OPT,GEOM)

    # Setup boundary conditions
    exec( open( FE['mesh_input']['bcs_file'] ).read() )

    # initialize the fixed/free partitioning scheme:
    FE_init_partitioning(FE,OPT,GEOM)

    # assemble the boundary conditions
    FE_assemble_BC(FE,OPT,GEOM)

    # compute elastic coefficients
    FE_compute_constitutive_matrices(FE,OPT,GEOM)

    # compute the element stiffness matrices
    FE_init_element_stiffness(FE,OPT,GEOM)


def read_gmsh(FE,OPT,GEOM):
    # This function reads a mesh in Matlab format generated using Gmsh and
    # stores the node coordinates and element connectivity in the global FE
    # structure. 
    #
    # The name of the Matlab .m file (which must include extension) should be
    # previously stored in FE['mesh_input.gmsh_filename.
    #
    # This function has been tested with Gmsh v. 4.0.7.  Gmsh is distributed
    # under the terms of the GNU General Public License (GPL), and can be
    # downloaded from
    #
    # gmsh.info
    #

    # *** Do not modify this code unless necessary **
    ldict = locals()
    exec( open( FE['mesh_input']['gmsh_filename'] ).read() ,globals() , ldict )
    msh = ldict['msh']

    # Note we assume the Matlab file generated by Gmsh creates a structure
    # named msh, and that this structure is available in the workspace for our
    # code to then grab.

    # If HEXAS elements are not present, then assume the mesh is 2D (in that
    # case, the field msh.QUADS must exist
    if 'HEXAS' in msh:
        FE['dim'] = 3
        el_array = msh['HEXAS']
    else:
        FE['dim'] = 2
        el_array = msh['QUADS']
        
    # at this point, there is no garuntee that the element nodes are in the
    # correct order for this code. We assume that the ordering of nodes yields
    # a positive determinant, so we must now ensure that this is the case.

    nodes_per_elem = 2**FE['dim']
    coords      = msh['POS'][:,0:FE['dim']].T
    elem_node   = el_array[:,0:nodes_per_elem].T
    
    e_coord = coords[:,elem_node].reshape( ( FE['dim'] , elem_node.shape[0] , elem_node.shape[1] ) , order='F' )

    if FE['dim'] == 2:
        # ensure that elemnt nodes 1,2,3 are in ccw order:
        homogenous_coord_123 = np.ones( ( 3 , 3 , e_coord.shape[2] ) )
        homogenous_coord_123[ 0:FE['dim'] , 0:(FE['dim']+1) , : ] = \
            e_coord[ 0:FE['dim'] , 0:(FE['dim']+1) , : ]

        cw = np.where( np.sum( np.cross( homogenous_coord_123[:,0,:] , 
            homogenous_coord_123[:,1,:] , axis=0 ) *  
            homogenous_coord_123[:,2,:] , axis=0 ) < 0 )[0]
        
        # swap nodes 2 and 4 of the cw elements
        elem_node[1,cw], elem_node[3,cw] = elem_node[3,cw], elem_node[1,cw]
    else:
        print('Not yet verified that the element nodes are in the canonical order')

    # Populate appropriate fields in FE structure
    FE['n_node']    = msh['nbNod']
    FE['n_elem']    = el_array.shape[0]
    FE['coords']    = coords
    FE['elem_node'] = elem_node

    # FE['mesh_input']['elements_per_side'] = \
    #     np.array( ( np.unique(msh['POS'][:,0]).shape[0] - 1 ,
    #         np.unique(msh['POS'][:,1]).shape[0] - 1 ) )
    
    # FE['mesh_input']['box_dimensions'] = \
    #     msh['max'] - msh['min']

    # print( FE['mesh_input']['elements_per_side'] )
    # print( FE['mesh_input']['box_dimensions'] )



exec(open('input_files/cantilever2d/inputs_cantilever2d.py').read())
# exec(open('input_files/Lbracket2d/inputs_Lbracket2d.py').read())
# exec(open('input_files/mbb2d/inputs_mbb2d.py').read())
# exec(open('input_files/cantilever3d/inputs_cantilever3d.py').read())

## Start timer
tic = time.perf_counter()

## Initialization
init_FE(FE,OPT,GEOM)
init_geometry(FE,OPT,GEOM)
init_optimization(FE,OPT,GEOM)

# ## Analysis
perform_analysis(FE,OPT,GEOM) 
# plot_density()
## Finite difference check of sensitivities
# (If requested)
if OPT['make_fd_check']:
    run_finite_difference_check()

## Optimization
# OPT['history'] = runopt(FE,OPT,GEOM,OPT['dv'], obj , nonlcon )
OPT['history'] = runopt(FE,OPT,GEOM,OPT['dv'], obj , nonlcon )

# ## Report time
toc = time.perf_counter()
print( "Time in seconds: " + str(toc-tic) )
print( "Time FE analysis: " + str(FE['FEA_t']) ) 
print( "FE analysis times: " + str(FE['FEA_n']) ) 
print( "Time projection: " + str(GEOM['proj_t']) ) 
print( "Time f. eval: " + str(OPT['Fun_t']) ) 

# hold graph
plt.ioff()

# ## Plot History
if True == OPT['options']['plot']:
    plot_history(3)
