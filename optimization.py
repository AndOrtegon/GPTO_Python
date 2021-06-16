import numpy as np
import scipy.sparse as sp 

from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy.io import savemat
from MMA import mmasub, kktcheck

from geometry_projection import *
from FE_routines import *
from functions import *
from plotting import *


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
            plot_design(0)
            plt.title( 'design, iteration = {iter}'.format(iter=iter) )
            figure = plt.figure(0)
            figure.canvas.manager.window.wm_geometry("+0+0")

            plot_density(1)
            figure = plt.figure(1)
            figure.canvas.manager.window.wm_geometry("+500+0")

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

            folder, baseFileName = os.path.split( GEOM['initial_design']['path'] )
            mat_filename = folder + '/' + baseFileName[:-3] + '.mat'
            savemat( mat_filename , GEOM )

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
           
