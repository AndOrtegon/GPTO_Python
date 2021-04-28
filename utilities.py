import numpy as np
from gp_util import *
from FE_routines import *

def compute_compliance(FE,OPT,GEOM):
    # This function computes the mean compliance and its sensitivities
    # based on the last finite element analysis
    # global FE, OPT

    # compute the compliance (Eq. (15))
    c = np.dot( FE['U'] , FE['P'] )
    
    # compute the design sensitivity
    Ke = FE['Ke']
    Ue = FE['U'][FE['edofMat']].repeat(FE['n_edof'],axis=2).transpose((1,2,0))
    Ue_T = Ue.transpose((1,0,2))

    Dc_Dpenalized_elem_dens = np.sum( np.sum( 
        - Ue_T * Ke * Ue , 
        0 ) , 0 )   # Eq. (24)

    Dc_Ddv = Dc_Dpenalized_elem_dens[:,None] * OPT['Dpenalized_elem_dens_Ddv'] # Eq. (25)
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
    # [g, geq, gradg, gradgeq] = nonlcon(dv) returns the costraints
    # global  OPT
    
    OPT['dv_old'] = OPT['dv']
    OPT['dv'] = dv
    
    if OPT['dv'] != OPT['dv_old']:
        # Update or perform the analysis
        update_geom_from_dv() # update GEOM for this design
        perform_analysis()

    n_con   = OPT['functions']['n_func']-1 # number of constraints
    g       = np.zeros(n_con,1)
    gradg   = np.zeros(OPT['n_dv,n_con'])

    for i in range(0,n_con):
        g[i] = OPT['functions']['f'][i+1]['value']
        g    = g - OPT['functions']['constraint_limit']
        
        gradg[:,i] = OPT['functions']['f'][i+1].grad

    return g, geq


def obj(dv):
    global  OPT
    
    OPT['dv_old'] = OPT['dv'] # save the previous design
    OPT['dv'] = dv # update the design
    
    
    if OPT['dv'] != OPT['dv_old']:
        # If different, update or perform the analysis
        update_geom_from_dv(FE,OPT,GEOM)
        perform_analysis(FE,OPT,GEOM)

    f = OPT['functions']['f'][0]['value']
    # gradf = OPT['functions']['f'][0]['grad']
    
    return f


def perform_analysis(FE,OPT,GEOM):
    # Perform the geometry projection, solve the finite
    # element problem for the displacements and reaction forces, and then
    # evaluate the relevant functions.
    project_element_densities(FE,OPT,GEOM)
    FE_analysis(FE,OPT,GEOM)
    evaluate_relevant_functions(FE,OPT,GEOM)

