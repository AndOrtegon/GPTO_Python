import numpy as np
import scipy.sparse as sp

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
            Drho_e_Ddv[:,OPT['bar_dv'][:,b]] + \
            np.concatenate( ( Drho_e_Dbar_s[b,:,:].reshape( ( FE['n_elem'], 2*FE['dim'] ) ) ,  \
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
            Dpenal_rho_e_Ddv[:,OPT['bar_dv'][:,b]] + \
            np.concatenate( \
                ( Dpenal_rho_e_Dbar_s[b,:,:].reshape( ( FE['n_elem'], 2*FE['dim'] ) ) ,  \
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
    #
    # This def updates the values of the unscaled bar geometric parameters
    # from the values of the design variableds (which will be scaled if
    # OPT.options['dv']_scaling is true). It does the
    # opposite from the def update_dv_from_geom.
    #
    # global GEOM , OPT , FE

    # Eq. (32)
    GEOM['current_design']['point_matrix'][:,12:] = \
        ( OPT['scaling']['point_scale'] * OPT['dv'][ OPT['point_dv'] ].reshape( ( FE['dim'] , GEOM.n_point ) ) \
            + np.transpose( OPT['scaling']['point_min'] ) )
    GEOM['current_design']['bar_matrix'][:,-2] = \
        OPT['dv'][ OPT['size_dv'] ]
    GEOM['current_design']['bar_matrix'][:,-1] = OPT['dv'](OPT['radius_dv']) * OPT['scaling']['radius_scale'] \
        + OPT['scaling']['radius_min']
