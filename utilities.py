import numpy as np

def compute_compliance():
    # This function computes the mean compliance and its sensitivities
    # based on the last finite element analysis
    global FE, OPT

    # compute the compliance (Eq. (15))
    #c = full( np.dot( FE['U , FE['P ) )

    # compute the design sensitivity
    Ke = FE['Ke']
    Ue = np.repeat( np.transpose( FE['U'](FE['edofMat']) ) , axis = 2 ).swapaxes(1,2)
    Ue_trans = Ue.swapaxes(0,1)

    Dc_Dpenalized_elem_dens = np.sum( np.sum( 
        np.matmul( np.matmul( -Ue_trans , Ke ) , Ue ) , 
        0 , 1 ) ).reshape( ( 1 , FE['n_elem'] ) )   # Eq. (24)

    Dc_Ddv = Dc_Dpenalized_elem_dens * OPT['Dpenalized_elem_dens_Ddv'] # Eq. (25)
    grad_c = Dc_Ddv.T
    # save these values in the OPT structure
    OPT['compliance'] = c
    OPT['grad_compliance'] = grad_c

    return c , grad_c

def compute_volume_fraction():
    #
    # This function computes the volume fraction and its sensitivities
    # based on the last geometry projection
    #
    global FE, OPT

    # compute the volume fraction
    v_e = FE['elem_vol'] # element
    V = sum(v_e) # full volume
    v = v_e.flatten() * OPT['elem_dens'].flatten() # projected volume
    volfrac =  v/V # Eq. (16)

    # compute the design sensitivity
    Dvolfrac_Ddv = (v_e * OPT['Delem_dens_Ddv'])/V   # Eq. (31)
    grad_vofrac = Dvolfrac_Ddv.T
        
    # output
    OPT['volume_fraction'] = volfrac
    OPT['grad_volume_fraction'] = grad_vofrac

    return volfrac , grad_vofrac


def evaluate_relevant_functions():
    # Evaluate_relevant_functions() looks at OPT['functions'] and evaluates the
    # relevant functions for this problem based on the current OPT['dv']
    global OPT

    OPT['functions']['n_func'] =  numel(OPT['functions']['f'])

    for i in range(0,OPT['functions']['n_func']):
        value , grad = feval(OPT['functions'].f{i}.function)
        OPT['functions'].f{i}.value = value
        OPT['functions'].f{i}.grad = grad


def nonlcon(dv):
    # [g, geq, gradg, gradgeq] = nonlcon(dv) returns the costraints
    global  OPT
    
    OPT['dv_old'] = OPT['dv']
    OPT['dv'] = dv
    
    if OPT['dv'] != OPT['dv_old']:
        # Update or perform the analysis

        update_geom_from_dv() # update GEOM for this design
        perform_analysis()

    n_con   = OPT['functions'].n_func-1 # number of constraints
    g       = np.zeros(n_con,1)
    gradg   = np.zeros(OPT['n_dv,n_con'])

    for i in range(0,n_con):
        g[i] = OPT['functions'].f{i+1}.value
            g = g - OPT['functions'].
        
        gradg[:,i] = OPT['functions'].f{i+1}.grad

    geq = np.empty
    gradgeq = np.empty

    return g, geq, gradg, gradgeq


def = obj(dv):
    global  OPT
    
    OPT['dv_old'] = OPT['dv'] # save the previous design
    OPT['dv'] = dv # update the design
    
    
    if OPT['dv'] !== OPT['dv_old']:
        # If different, update or perform the analysis
        update_geom_from_dv()
        perform_analysis()

    f = OPT['functions'].f{1}.value
    gradf = OPT['functions'].f{1}.grad
    
    return f, gradf


def perform_analysis():
    # Perform the geometry projection, solve the finite
    # element problem for the displacements and reaction forces, and then
    # evaluate the relevant functions.
    project_element_densities()
    FE_analysis()
    evaluate_relevant_functions()

