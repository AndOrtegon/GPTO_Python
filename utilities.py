
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

