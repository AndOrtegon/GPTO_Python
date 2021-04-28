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
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.sparse import linalg
import scipy.sparse as sp 

OPT  = {}
GEOM = {}
FE   = {}


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
    grad_vofrac = Dvolfrac_Ddv
    
    # output
    OPT['volume_fraction'] = volfrac
    OPT['grad_volume_fraction'] = grad_vofrac

    return volfrac , grad_vofrac


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
    
    OPT['dv_old'] = OPT['dv']
    OPT['dv'] = dv
    
    # Update or perform the analysis
    if ( OPT['dv'] != OPT['dv_old'] ).any():
        update_geom_from_dv(FE, OPT, GEOM)
        perform_analysis(FE, OPT, GEOM)

    n_con   = OPT['functions']['n_func']-1 # number of constraints
    g       = np.zeros((n_con,1))

    for i in range(0,n_con):
        g[i] = OPT['functions']['f'][i+1]['value']
        
    return g.flatten()


def obj(dv):
    global  FE, OPT, GEOM
    
    OPT['dv_old'] = OPT['dv'] # save the previous design
    OPT['dv'] = dv # update the design
    
    # If different, update or perform the analysis
    if ( OPT['dv'] != OPT['dv_old'] ).any():
        update_geom_from_dv(FE,OPT,GEOM)
        perform_analysis(FE,OPT,GEOM)

    f = OPT['functions']['f'][0]['value'].flatten()
    g = OPT['functions']['f'][0]['grad'].flatten()

    return f, g


def perform_analysis(FE,OPT,GEOM):
    # Perform the geometry projection, solve the finite
    # element problem for the displacements and reaction forces, and then
    # evaluate the relevant functions.
    project_element_densities(FE,OPT,GEOM)
    FE_analysis(FE,OPT,GEOM)
    evaluate_relevant_functions(FE,OPT,GEOM)


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
        OPT['parameters']['elem_r'] = np.sqrt(FE['dim'])/2 * FE['elem_vol']*(1/FE['dim']) 
    
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
    # constrained trust region
    history = {}

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
    if  'fmincon-active-set' == OPT['options']['optimizer']:
        def output(x,state):
            stop = False
            # print( state.status )
            print( "Iteration: " + str(state.nit) + \
                "\n\tCompliance: " + str(OPT['functions']['f'][0]['value']) +\
                "\n\tVolume fra: " + str(OPT['functions']['f'][1]['value']) )
            # if state.status == 'init':
            #     # do nothing
            #     pass
            # elif state == 'iter':
            #     ## Concatenate current point and obj value with history
            #     # history['fval'] = np.stack( ( history['fval'] , optimValues.fval ) , axis = 1 )
            #     # history['fconsval'] = np.stack( ( history['fconsval'] , nonlcon(OPT['dv']) ) , axis = 1 )
            #     # history['x'] = np.stack( ( history['x'] , x[:] ) , axis = 1 ) # here we make x into a column vector
                
            #     # # Write to vtk file if requested.  
            #     # if OPT['options']['write_to_vtk'] == 'all':
            #     #     writevtk(OPT['options']['vtk_output_path'], 'dens', optimValues.iteration)
                
            # elif state == 'done':
            #     # do nothing
            #     pass
            
            return stop    

        # Initialize history object
        history['x'] = np.array(())
        history['fval'] = np.array(())
        history['fconsval'] = np.array(())

        bounds = Bounds(lb.flatten(),ub.flatten())
        nonlinear_constraint = NonlinearConstraint(nonlcon,
            -np.inf, OPT['functions']['constraint_limit'],
            jac='2-point')

        # This is the call to the optimizer
        res = minimize(obj,x0.flatten(),method='trust-constr',jac=True,
            constraints=nonlinear_constraint,
            options={'verbose': 1},bounds=bounds,
            callback=output) 

    elif 'mma' == OPT['options']['optimizer']:
        ncons = OPT['functions']['n_func'] - 1  # Number of optimization constraints
        ndv = OPT['n_dv'] # Number of design variables

        # Initialize vectors that store current and previous two design iterates
        x = x0
        xold1 = x0 
        xold2 = x0

        # Initialize move limits 
        ml_step = OPT['options']['move_limit'] * abs(ub - lb)  # Compute move limits once

        # Initialize lower and upper asymptotes
        low = lb
        upp = ub

        # These are the MMA constants (Svanberg, 1998 DACAMM Course)
        c   = 1000*np.ones( (ncons,1) )
        d   = np.ones( (ncons,1) )
        a0  = 1
        a   = np.zeros( (ncons, 1) )

        # Evaluate the initial design and print values to screen 
        iter = 0
        f0val , df0dx = obj(x)
        fval, dummy, dfdx, dummy2 = nonlcon(x)
        dfdx = dfdx.T

        fprintf('It. #i, Obj= #-12.5e, ConsViol = #-12.5e\n', \
            iter, f0val, max(max(fval, zeros(ncons,1))))

        ###
        # Save initial design to history
        history['fval'] = np.stack( ( history['fval'] , f0val ) , axis=1 )
        history['fconsval'] = np.stack( ( history['fconsval'] , fval ) , axis=1 )
        history['x'] = np.stack( ( history['x'] , x[:] ) , axis = 1 )

        ###
        # Plot initial design 
        plotfun(iter)
                
        #### Initialize stopping values
        kktnorm         = 10*OPT['options']['kkt_tol']
        dv_step_change  = 10*OPT['options']['step_tol']


        ## MMA Loop
        while kktnorm > OPT['options']['kkt_tol'] and iter < OPT['options']['max_iter'] and \
                dv_step_change > OPT['options']['step_tol']:

            iter = iter+1

            # Impose move limits by modifying lower and upper bounds passed to MMA
            # Eq. (33)
            mlb = np.max(lb, x - ml_step)
            mub = np.min(ub, x + ml_step)


            #### Solve MMA subproblem for current design x
            xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = \
            mmasub(ncons,ndv,iter,x,mlb,mub,xold1, \
                xold2, f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d)

            #### Updated design vectors of previous and current iterations
            xold2 = xold1
            xold1 = x
            x  = xmma
            
            # Update function values and gradients
            f0val , df0dx  = obj(x)
            fval, dummy, dfdx, dummy2 = nonlcon(x)
            dfdx = dfdx.T
            
            # Compute change in design variables
            # Check only after first iteration
            if iter > 1:
                dv_step_change = norm(x - xold1)
                if dv_step_change < OPT['options']['step_tol']:
                    fprintf('Design step convergence tolerance satisfied.\n')
                
            
            if iter == OPT['options']['max_iter']:
                fprintf('Reached maximum number of iterations.\n')
                
            
            # Compute norm of KKT residual vector
            [residu,kktnorm,residumax] = \
            kktcheck(ncons,ndv,xmma,ymma,zmma,lam,xsi,eta,mu,zet,s, \
                lb,ub,df0dx,fval,dfdx,a0,a,c,d)
            
            # Produce output to screen
            fprintf('It. #i, Obj= #-12.5e, ConsViol = #-12.5e, KKT-norm = #-12.5e, DV norm change = #-12.5e\n', \
                iter, f0val, max(max(fval, zeros(ncons,1))), kktnorm, dv_step_change)
            
            # # Save design to .mat file
            # folder, baseFileName, dummy = fileparts(GEOM['initial_design']['path'])
            # mat_filename = fullfile(folder, strcat(baseFileName, '.mat'))
            # save(mat_filename, 'GEOM')
            
            # # Write to vtk file if requested.  
            # if OPT['options']['write_to_vtk'] == 'all':
            #     writevtk(OPT['options']['vtk_output_path'], 'dens', iter)
                
            
            # Update history
            history['fval'] = np.stack( ( history['fval'] ,f0val) , axis=1 )
            history['fconsval'] = np.stack( ( history['fconsval'] , fval ) , axis=1 )
            history['x'] = np.stack( ( history['x'] , x[:] ) , axis=1 )
            
            # Plot current design
            plotfun(iter)

    if True:
        var = True
        # # Write vtk for final iteration if requested
        # if OPT['options']['write_to_vtk'] == 'all' or \
        #         strcmp(OPT['options']['write_to_vtk'] == 'last'
        #     writevtk(OPT['options']['vtk_output_path'], 'dens', optim_output.iterations)
    
        # =========================================================================

        # def plotfun(x,optimValues,state):
        #     if state = 'init':
        #         iter = 0 
        #     else:
        #         iter = optimValues.iteration

        #     plotall(iter)


        # def plotall(iter):
        #     if OPT['options']['plot'] == True:
        #         figure(1)
        #         plot_design(1)
        #         title(sprintf('design, iteration = #i',iter))
        #         axis equal
        #         xlim([FE['coord_min'][0], FE['coord_max'][0]])
        #         ylim([FE['coord_min'][1], FE['coord_max'][1]])
        #         if FE['dim'] == 2:
        #             view(2)
        #         else:
        #             zlim([FE['coord_min'][2], FE['coord_max'][2])
        #             view([50,22])

        #         if iter == 0:
        #             pos1 = get(gcf,'Position') # get position of fig 1
        #             # This assume Matlab places figure centered at center of
        #             # screen
        #             fig1_x = pos1(1)       # fig1_y = pos1(2) 
        #             fig1_width = pos1(3)   # fig1_height = pos1(4)
        #             # Shift position left by half figure width
        #             set(gcf,'Position', pos1 - [fig1_width/2,0,0,0]) # Shift position of Figure(1) 
                
        #         figure(2)
        #         plot_density(2)
        #         axis equal
        #         xlim([FE['coord_min'][1), FE['coord_max'][1)])
        #         ylim([FE['coord_min'][2), FE['coord_max'][2)])

        #         if FE['dim'] == 2:
        #             view(2)
        #         else:
        #             zlim([FE['coord_min'][3), FE['coord_max'][3)])
        #             view([50,22])
                
        #         if iter == 0:
        #             # fig2_x = pos1(1) 
        #             fig2_y = pos1[1]
        #             fig2_width = pos1[2]
        #             fig2_height = pos1[3]

        #             # Shift position of fig 2 so that its left-bottom
        #             # corner coincides with the right-bottom corner of fig 1
        #             set(gcf,'Position', [fig1_x + fig1_width/2,fig2_y,fig2_width,fig2_height]) 
                
        #         if mma:
        #             drawback

        #         stop = False
        #     return stop   

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
    
    ## No implementado
    # plot the initial design:
    # if GEOM['initial_design'].plot:
    #     plot_design(1, \
    #         GEOM['initial_design']['point_matrix'], \
    #         GEOM['initial_design']['bar_matrix'])
    #     axis equal
    #     title('initial design')
    #     drawnow
    
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
    norm_x_e_1b = np.sqrt( np.sum( np.power( x_e_1b , 2 ) , 0 ) )   # (1,b,e)
    norm_x_e_2b = np.sqrt( np.sum( np.power( x_e_2b , 2 ) , 0 ) )   # (1,b,e) 

    l_be     = np.sum( x_e_1b * a_b[:,:,None] , 0 )                 # (1,b,e), Eq. (12)
    vec_r_be = x_e_1b - ( l_be.T * a_b[:,None] ).swapaxes(1,2)      # (i,b,e)
    r_be     = np.sqrt( np.sum( np.power( vec_r_be , 2 ) , 0 ) )    # (1,b,e), Eq. (13)

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
        -vec_r_be * d_inv * ( 1 - l_be_over_l_b ) * branch3
    
    ## assemble the sensitivities to the bar design parameters (scaled)
    Ddist_Dbar_s = np.concatenate((Dd_be_Dx_1b,Dd_be_Dx_2b),
        axis=0).transpose((1,2,0)) * \
        OPT['scaling']['point_scale'].repeat( 2 , axis=0 )

    return dist , Ddist_Dbar_s


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
    d_be , Dd_be_Dbar_s = compute_bar_elem_distance(FE,OPT,GEOM)

    ## Bar-element projected densities
    r_b =  GEOM['current_design']['bar_matrix'][:,-1] # bar radii
    r_e =  OPT['parameters']['elem_r'] # sample window radius

    # X_be is \phi_b/r in Eq. (2).  Note that the numerator corresponds to
    # the signed distance of Eq. (8).
    X_be = ( ( r_b - d_be.T ).T / r_e )

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

    elif FE['dim'] == 3:
        rho_be[inB] = ( (X_be[inB]-2.0)*(-1.0/4.0)*(X_be[inB]+1.0)**2 )
        Drho_be_Dx_be[inB] = ( X_be[inB]**2*(-3.0/4.0)+3.0/4.0 ) # Eq. (28)

    # Sensitivities of raw projected densities, Eqs. (27) and (29)
    Drho_be_Dbar_s = ( Drho_be_Dx_be * -1/r_e * 
        Dd_be_Dbar_s.transpose((2,0,1)) ).transpose((1,2,0))
    
    Drho_be_Dbar_radii  = OPT['scaling']['radius_scale'] * Drho_be_Dx_be * np.transpose(1/r_e)


    ## Combined densities
    # Get size variables    
    alpha_b = GEOM['current_design']['bar_matrix'][:,-1] # bar size

    # Without penalization:
    # ====================
    # X_be here is \hat{\rho}_b in Eq. (4) with the value of q such that
    # there is no penalization (e.g., q = 1 in SIMP).
    X_be = alpha_b[:,None] * rho_be

    # Sensitivities of unpenalized effective densities, Eq. (26) with
    # ?\partial \mu / \partial (\alpha_b \rho_{be})=1
    DX_be_Dbar_s = Drho_be_Dbar_s * alpha_b[:,None,None]
    DX_be_Dbar_size = rho_be  
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
            Drho_e_Dbar_s[b,:,:].reshape( ( FE['n_elem'], 2*FE['dim'] )  , order='F' ).copy() ,  \
            Drho_e_Dbar_size[b,:].reshape( ( FE['n_elem'], 1 ) ).copy() ,  \
            Drho_e_Dbar_radii[b,:].reshape( ( FE['n_elem'], 1 ) ).copy() ) , axis=1 )

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
                ( Dpenal_rho_e_Dbar_s[b,:,:].reshape( ( FE['n_elem'], 2*FE['dim'] ) , order='F' ).copy() ,  \
                Dpenal_rho_e_Dbar_size[b,:].reshape( ( FE['n_elem'], 1 ) ).copy() ,  \
                Dpenal_rho_e_Dbar_radii[b,:].reshape( ( FE['n_elem'], 1 ) ).copy() ) , \
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
        S = x_min + (1-x_min)*np.log( np.sum(epx(x),axis=0) )/p
        dSdx = (1-x_min)*np.epx( p*x )/np.sum(epx(x),axis=0)

    elif form_def == 'KS_under':
        # note: convergence might be fixed with Euler-Gamma
        N = size(x,1)
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
    
    OPT['dv'][ OPT['size_dv'],0 ] = GEOM['initial_design']['bar_matrix'][:,-2]

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

    GEOM['current_design']['bar_matrix'][:,-2] = OPT['dv'][OPT['size_dv']].copy()

    GEOM['current_design']['bar_matrix'][:,-1] = OPT['dv'][OPT['radius_dv']] * \
        OPT['scaling']['radius_scale'] + \
        OPT['scaling']['radius_min']


def FE_analysis(FE,OPT,GEOM):
    # Assemble the Global stiffness matrix and solve the FEA
    # Assemble the stiffness matrix partitions Kpp Kpf Kff
    FE_assemble_stiffness_matrix(FE,OPT,GEOM)
    # Solve the displacements and reaction forces
    FE_solve(FE,OPT,GEOM)


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
        idx = FE['dim'] * ( FE['BC']['disp_node'][idisp]-1) + FE['BC']['disp_dof'][idisp]
        FE['U'][idx] = FE['BC']['disp_value'][idisp]

    ## Assemble prescribed loads
    # initialize a sparse global force vector
    
    FE['P'] = sp.csc_matrix( ( FE['n_global_dof'],1) )

    # determine prescribed xi load components:
    for iload in range( 0 , FE['BC']['n_pre_force_dofs'] ):
        idx = FE['dim'] * ( FE['BC']['force_node'][iload]-1) + \
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
            np.dot( (n7-n2) + (n8-n1), np.cross( (n7-n4)          , (n3-n1)           ) ) + \
            np.dot( (n8-n1)          , np.cross( (n7-n4) + (n6-n1), (n7-n5)           ) ) + \
            np.dot( (n7-n2)          , np.cross( (n6-n1)          , (n7-n5) + (n3-n1) ) ) \
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
                if 2 == FE['dim']:
                    eta = gauss_pt[j]
                    
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
    # This function computes FE.sK_void, the vector of element 
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
        FE['fixeddofs'][ 2*FE['BC']['disp_node'][ FE['BC']['disp_dof'] == 1 ] - 1 ] = True # set prescribed x1 DOFs 
        FE['fixeddofs'][ 2*FE['BC']['disp_node'][ FE['BC']['disp_dof'] == 2 ] ] = True   # set prescribed x2 DOFs 
    elif 3 == FE['dim']:
        FE['fixeddofs'][ 3*FE['BC']['disp_node'][ FE['BC']['disp_dof'] == 1 ] - 2 ] = True # set prescribed x1 DOFs 
        FE['fixeddofs'][ 3*FE['BC']['disp_node'][ FE['BC']['disp_dof'] == 2 ] - 1 ] = True # set prescribed x2 DOFs 
        FE['fixeddofs'][ 3*FE['BC']['disp_node'][ FE['BC']['disp_dof'] == 3 ] ] = True   # set prescribed x3 DOFs 
        
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
        .reshape( FE['n_elem']*n_ewlem_dof**2 , order ='F' )
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
        tol = FE['analysis']['solver'].tol
        maxit = FE['analysis']['solver'].maxit
        # check if the user has specified use of the gpu
        
        if FE['analysis']['solver'].use_gpu == False:
            ## cpu solver
            
            ME.identifier = []
            try:
                L = ichol( FE['Kff'])
            except:
                print("A problem was encountered")
            
            if ME.identifier == 'MATLAB:ichol:Breakdown':
                msg = ['ichol encountered nonpositive pivot, using no preconditioner.']

                # you might consider tring different preconditioners (e.g. LU) in 
                # the case ichol breaks down. We will default to no preconditioner:
                FE['U'][F] = pcg( FE['Kff'], FE['rhs'], \
                    tol,maxit, \
                    [] \
                    )   
            else:
                msg = []
                # L.T preconditioner
                FE['U'][F] = pcg( FE['Kff'], FE['rhs'], \
                        tol, maxit, \
                        L,L.T )  
            
            print(msg)

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



exec(open('input_files/cantilever2d/inputs_cantilever2d.py').read())

## Start timer
tic = time.perf_counter()

## Initialization

init_FE(FE,OPT,GEOM)
init_geometry(FE,OPT,GEOM)
init_optimization(FE,OPT,GEOM)

# # load('matlab.mat','GEOM') update_dv_from_geom

# ## Analysis
perform_analysis(FE,OPT,GEOM) 

## Finite difference check of sensitivities
# (If requested)
if OPT['make_fd_check']:
    run_finite_difference_check()

# ## Optimization
OPT['history'] = runopt(FE,OPT,GEOM,OPT['dv'], obj , nonlcon )

# ## Plot History
# if True == OPT.options.plot:
#     plot_history(3)

# ## Report time
# toc = time.perf_counter()
# print( "Time in seconds: str(toc-tic)" )