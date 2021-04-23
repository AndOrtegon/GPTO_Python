import numpy as np
import scipy.sparse as sp 
from gp_util import *

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
        OPTOPT['parameters']['elem_r'] = sqrt(FE['dim'])/2 * FE.elem_vol*(1/FE['dim']) 
    
    ##
    # Initilize the design variable and its indexing schemes

    # we are designing the points, the size variables, and the radii of the
    # bars:

    OPT['n_dv'] = FE['dim']*GEOM['n_point'] + 2*GEOM['n_bar']
    OPT['dv']   = np.zeros( (OPT['n_dv'],1) )

    OPT['point_dv']  = np.arange(0,FE['dim']*GEOM['n_point']) # such that dv(point_dv) = point
    OPT['size_dv']   = OPT['point_dv'][-1] + np.arange(0,GEOM['n_bar'])
    OPT['radius_dv'] = OPT['size_dv'][-1] + np.arange(0,GEOM['n_bar'])

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

    pt_dv = np.reshape(OPT['point_dv'][:],(FE['dim'],GEOM['n_point']))

    OPT['bar_dv'] = ( pt_dv[:,pt1] , pt_dv[:,pt2] ,
        OPT['size_dv'] , OPT['radius_dv'] )


def runfmincon(OPT,GEOM,FE,x0,obj,nonlcon):
    # Perform the optimization using Matlab's fmincon

    # Initialize history object
    history['x'] = {}
    history['fval'] = {}
    history['fconsval'] = {}

    # # call optimization
    # options = optimoptions(@fmincon,\
    #     'OutputFcn', output,
    #     'PlotFcn', plotfun,
    #     'Algorithm','active-set', \
    #     'FiniteDifferenceStepSize', 1e-5, \
    #     'SpecifyObjectiveGradient',True,\
    #     'SpecifyConstraintGradient',True,\
    #     'RelLineSrchBnd', OPT['options']['move_limit'], \ 
    #     'RelLineSrchBndDuration', OPT['options']['max_iter'], \
    #     'ConstraintTolerance', 1e-3, \
    #     'MaxIterations',OPT['options']['max_iter'], \
    #     'OptimalityTolerance',OPT['options']['kkt_tol'], \      
    #     'StepTolerance',OPT['options']['step_tol'],\
    #     'Display', 'iter') # 
        
    # x0                
    A = []
    b = []
    Aeq = []
    beq = []
    
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

    lb_bar = np.array( (lb_pointlb_point, lb_sizelb_radius) )
    ub_bar = np.array( (ub_pointub_point, ub_sizeub_radius) )

    lb = np.zeros( (OPT['dv'].shape) )
    ub = np.zeros( (OPT['dv'].shape) )
    lb[OPT['bar_dv']] = repmat(lb_bar,1,GEOM['n_bar'])
    ub[OPT['bar_dv']] = repmat(ub_bar,1,GEOM['n_bar'])

    # ******
    # This is the call to the optimizer
    #
    x,fval,exitflag,optim_output = fmincon(obj,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)
    # ******
    
    # # Write vtk for final iteration if requested
    # if OPT['options']['write_to_vtk'] == 'all' or \
    #         strcmp(OPT['options']['write_to_vtk'] == 'last'
    #     writevtk(OPT['options']['vtk_output_path'], 'dens', optim_output.iterations)
   
    # =========================================================================

    def output(x,optimValues,state):
        stop = False
        if state == 'init':
            # do nothing
            pass
        elif state == 'iter':
            # Concatenate current point and objective function
            # value with history
            history['fval'] = np.stack( ( history['fval'] , optimValues.fval ) , axis = 1 )
            history['fconsval'] = np.stack( ( history['fconsval'] , nonlcon(OPT['dv']) ) , axis = 1 )
            history['x'] = np.stack( ( history['x'] , x[:] ) , axis = 1 ) # here we make x into a column vector
            
            # # Write to vtk file if requested.  
            # if OPT['options']['write_to_vtk'] == 'all':
            #     writevtk(OPT['options']['vtk_output_path'], 'dens', optimValues.iteration)
            
        elif state == 'done':
            # do nothing
            pass
        
        return stop
    
    # =========================================================================

    # def plotfun(x,optimValues,state):
    #     if OPT['options']['plot'] == True:
    #         figure(1)
    #         plot_design(1)
    #         title(sprintf('design, iteration = #i',optimValues.iteration))
    #         axis equal
    #         xlim([FE['coord_min'][0], FE['coord_max'][0]])
    #         ylim([FE['coord_min'][1], FE['coord_max'][1]])
    #         if FE['dim'] == 2:
    #             view(2)
    #         else:
    #             zlim([FE['coord_min'][3), FE['coord_max'][3)])
    #             view([50,22])

    #         if state == 'init':
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
            
    #         if state == 'init':
    #             # fig2_x = pos1(1) 
    #             fig2_y = pos1[1]
    #             fig2_width = pos1[2]
    #             fig2_height = pos1[3]

    #             # Shift position of fig 2 so that its left-bottom
    #             # corner coincides with the right-bottom corner of fig 1
    #             set(gcf,'Position', [fig1_x + fig1_width/2,fig2_y,fig2_width,fig2_height]) 
            
    #         stop = False
    #     return stop

    return history    

def runmma(OPT,GEOM,FE,x0,obj,nonlcon):
    #
    # Perform the optimization using MMA
    #


    # Initialize history object
    history['x'] = {}
    history['fval'] = {}
    history['fconsval'] = {}

    # Initialize lower and upper bounds vectors
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

    lb_bar = [lb_pointlb_pointlb_sizelb_radius]
    ub_bar = [ub_pointub_pointub_sizeub_radius]

    lb = np.zeros(OPT['dv'].shape)
    ub = np.zeros(OPT['dv'].shape) 

    lb[OPT['bar_dv']] = np.repmat(lb_bar,1,GEOM['n_bar'])
    ub[OPT['bar_dv']] = np.repmat(ub_bar,1,GEOM['n_bar'])

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
    #
    # ******* MAIN MMA LOOP STARTS *******
    #
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
        
        # Save design to .mat file
        folder, baseFileName, dummy = fileparts(GEOM['initial_design']['path'])
        mat_filename = fullfile(folder, strcat(baseFileName, '.mat'))
        save(mat_filename, 'GEOM')
        
        # # Write to vtk file if requested.  
        # if OPT['options']['write_to_vtk'] == 'all':
        #     writevtk(OPT['options']['vtk_output_path'], 'dens', iter)
            
        
        # Update history
        history['fval'] = np.stack( ( history['fval'] ,f0val) , axis=1 )
        history['fconsval'] = np.stack( ( history['fconsval'] , fval ) , axis=1 )
        history['x'] = np.stack( ( history['x'] , x[:] ) , axis=1 )
        
        # Plot current design
        plotfun(iter)
    
    # # Write vtk for final iteration if requested
    # if strcmp(OPT['options']['write_to_vtk'], 'all') or \
    #         strcmp(OPT['options']['write_to_vtk'], 'last')
    #     writevtk(OPT['options']['vtk_output_path'], 'dens', iter)
    

    # ============================================


    # def plotfun(iter):
    #     # Note that this function has a slightly different format than its
    #     # equivalent for fmincon.
        
    #     if OPT['options'].plot == True:
    #         figure(1)
    #         plot_design(1)
    #         title(sprintf('design, iteration = #i',iter))
    #         axis equal
    #         xlim( ( FE['coord_min'][0] , FE['coord_max'][0] ) )
    #         ylim( ( FE['coord_min'][1] , FE['coord_max'][1] ) )
    #         if FE['dim'] == 2:
    #             view(2)
    #         else:
    #             zlim( ( FE['coord_min'][2] , FE['coord_max'][2] ) )
    #             view([50,22])
            
    #         if iter==0:
    #             pos1 = get(gcf,'Position') # get position of fig 1
    #             # This assume Matlab places figure centered at center of
    #             # screen
    #             fig1_x = pos1(1)       # fig1_y = pos1(2) 
    #             fig1_width = pos1(3)   # fig1_height = pos1(4)
    #             # Shift position left by half figure width
    #             set(gcf,'Position', pos1 - [fig1_width/2,0,0,0]) # Shift position of Figure(1) 
            
            
    #         figure(2)
    #         plot_density(2)
    #         axis 
            
    #         xlim([FE['coord_min'][0], FE['coord_max'][0]])
    #         ylim([FE['coord_min'][1], FE['coord_max'][1]])
    #         if FE['dim'] == 2:
    #             view(2)
    #         else:
    #             zlim([FE['coord_min'][2], FE['coord_max'][2]])
    #             view([50,22])
            
    #         drawnow
    #         if iter == 0:
    #             # fig2_x = pos1(1) 
    #             fig2_y = pos1[1] 
    #             fig2_width = pos1[2]
    #             fig2_height = pos1[3]

    #             # Shift position of fig 2 so that its left-bottom
    #             # corner coincides with the right-bottom corner of fig 1
    #             set(gcf,'Position', [fig1_x + fig1_width/2,fig2_y,fig2_width,fig2_height]) 
                
    # return history                
