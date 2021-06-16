import os
import numpy as np
import matplotlib.collections
import matplotlib.pyplot as plt

from FE_routines import *
from scipy.sparse import linalg

def plot_density(fig):
    if FE['mesh_input']['type'] =='read-gmsh':
        # mesh was made by gmsh
        plot_density_cells(fig)
    else:
        # mesh was generated and comforms to meshgrid format. 
        # we then default to plotting level-sets of the density as
        # linearly interpolated between the centroids of the mesh.
        plot_density_levelsets(fig)

    title_string = 'density, %s = %f' % ( OPT['functions']['objective'] , OPT['functions']['f'][0]['value'] )
    plt.title( title_string )    
    
    plt.pause(0.0001)
    plt.draw()


def plot_density_levelsets(fig):
    global FE, OPT, GEOM
    
    if FE['mesh_input']['type'] != 'generate' and \
        FE['mesh_input']['type'] != 'read-home-made': # mesh was not generated

        print('not yet implemented for non meshgrid conforming meshes')

    ## Change here whether you want to plot the penalized (i.e., effective) or 
    ## the unpenalized (i.e., projected) densities.  By default, we plot the 
    ## effective densities.
    #
    # For penalized, use OPT.penalized_elem_dens;
    # For unpenalized, use OPT.elem_dens;

    plot_dens = OPT['elem_dens']
    # plot_dens = OPT['penalized_elem_dens']

    plt.ion()
    figu = plt.figure(fig)
    ax = plt.gca()

    if FE['dim'] == 2:
        # Level sets 
        n = 64
        levels = np.linspace(0,1,n)

        if not 'centroid_mesh' in OPT['options']:
            OPT['options']['centroid_mesh'] = {}

            mn = FE['mesh_input']['elements_per_side']
            nm = mn[np.array((0,1))]  # for meshgrid

            OPT['options']['centroid_mesh']['shape'] = nm
            OPT['options']['centroid_mesh']['X'] = FE['centroids'][0,:].reshape(nm)
            OPT['options']['centroid_mesh']['Y'] = FE['centroids'][1,:].reshape(nm)
        
        X = OPT['options']['centroid_mesh']['X']
        Y = OPT['options']['centroid_mesh']['Y']
        V = plot_dens.reshape( OPT['options']['centroid_mesh']['shape'] )
        
        ax.cla()
        fv = plt.contourf( X , Y , V , levels , cmap='gray_r' , extend='both' )
        
        plt.xlim( (FE['coord_min'][0], FE['coord_max'][0]) )
        plt.ylim( (FE['coord_min'][1], FE['coord_max'][1]) )
        plt.gca().set_aspect('equal', adjustable='box')


    elif FE['dim'] == 3:
        levels = [.25,.5,.75]



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
        F = FE['elem_node'].T # matrix of faces to be sent to PolyCollection
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


    # for n levels of opacity color
    n = 64
    alpha = 1 - np.minimum( 1 , np.round( n * plot_dens ) / n )

    if FE['dim'] == 2:
        verts = V[F]
        pc  = matplotlib.collections.PolyCollection(verts,  cmap='gray' )
        pc.set_array( alpha )
        ax.add_collection(pc)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim( (FE['coord_min'][0], FE['coord_max'][0]) )
        plt.ylim( (FE['coord_min'][1], FE['coord_max'][1]) )


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
    bar_color = np.array((1,0,0))    # red 
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
    norm_e_2b   = np.sqrt( np.sum(e_2b**2,0) )
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
        sx2 = z[N//2:,:]
        sy2 = x[N//2:,:]
        sz2 = y[N//2:,:]
        X1  = np.stack( ( sx1.flatten('F') , sy1.flatten('F') , sz1.flatten('F') ) , axis=1 ).T
        X2  = np.stack( ( sx2.flatten('F') , sy2.flatten('F') , sz2.flatten('F') ) , axis=1 ).T
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
    r_b     = bar_mat[:,-1]
    alpha   = bar_mat[:,-2]

    Color = bar_color
    # C = colormap('gray')
    # colormap(C.*Color) # color the gray-scale map

    plt.ion()
    figu = plt.figure(fig)    

    if FE['dim'] == 2:
        ax = plt.gca()
        ax.cla()
    elif FE['dim'] == 3: 
        ax = plt.axes(projection='3d')
        ax.cla()
    
    for b in range(0,n_bar):
        Alpha = alpha[b]**2

        if Alpha > size_tol:
            bar_X1 = r_b[b] * R_b[:,:,b] @ X1 + x_1b[:,b][:,None]
            bar_X2 = r_b[b] * R_b[:,:,b] @ X2 + x_2b[:,b][:,None]

            if FE['dim'] == 3:
                bar_x1 = np.reshape(bar_X1[0,:], [N//2, N+1] , order='F' )
                bar_y1 = np.reshape(bar_X1[1,:], [N//2, N+1] , order='F' )
                bar_z1 = np.reshape(bar_X1[2,:], [N//2, N+1] , order='F' )

                bar_x2 = np.reshape(bar_X2[0,:], [N//2+1, N+1] , order='F' )
                bar_y2 = np.reshape(bar_X2[1,:], [N//2+1, N+1] , order='F' )
                bar_z2 = np.reshape(bar_X2[2,:], [N//2+1, N+1] , order='F' )

                bar_x = np.concatenate( (bar_x1 , bar_x2) )
                bar_y = np.concatenate( (bar_y1 , bar_y2) )
                bar_z = np.concatenate( (bar_z1 , bar_z2) )

                # Create surface
                s = ax.plot_surface(bar_x,bar_y,bar_z,
                    cmap='Reds_r', antialiased=False)
                ax.set_facecolor = np.array((1,0,0,Alpha))

                ax.set_xlim( (FE['coord_min'][0], FE['coord_max'][0]) )
                ax.set_ylim( (FE['coord_min'][1], FE['coord_max'][1]) )
                ax.set_zlim( (FE['coord_min'][2], FE['coord_max'][2]) )
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

                # Create object in axis
                vertices = np.array((bar_x,bar_y)).T[None,:,:]
                pc  = matplotlib.collections.PolyCollection(vertices)

                pc.set_facecolor( np.append(Color,Alpha) )
                pc.set_edgecolor( np.array((0,0,0,1)) )
                ax.add_collection(pc)

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
                FE['elem_node'][6, iel] , FE['elem_node'][7, iel] ) )

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

