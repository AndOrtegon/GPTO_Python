import numpy as np
import scipy.sparse as sp
from mesh_util import *

def FE_analysis(FE,OPT,GEOM):
    # Assemble the Global stiffness matrix and solve the FEA
    # Assemble the stiffness matrix partitions Kpp Kpf Kff
    FE_assemble_stiffness_matrix()
    # Solve the displacements and reaction forces
    FE_solve()


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
    penalized_rho_e = np.repeat( OPT['penalized_elem_dens'].flatten() , 
        (1,FE['n_edof'],FE['n_edof']) ).swapaxes(0,1).swapaxes(1,2)
            
    # Ersatz material: (Eq. (7))
    penalized_Ke = penalized_rho_e * FE['Ke']
    FE['sK_penal'] = penalized_Ke.flatten()

    # assemble the penalized global stiffness matrix
    K = sparse( FE['iK'] , FE['jK'] , FE['sK_penal'] )

    # partition the stiffness matrix and return these partitions to FE
    FE['Kpp'] = K[ FE['fixeddofs_ind'] , FE['fixeddofs_ind'] ]
    FE['Kfp'] = K[ FE['freedofs_ind'] , FE['fixeddofs_ind'] ]

    # note: by symmetry Kpf = Kfp', so we don't store Kpf. Tall and thin
    # matrices are stored more efficiently as sparse matrices, and since we
    # generally have more free dofs than fixed, we choose to store the rows as
    # free dofs to save on memory.
    FE['Kff'] = K[ FE['freedofs_ind'] , FE['freedofs_ind'] ]


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
    
    FE['centroids'] = np.reshape( np.mean(CoordArray,axis=0) , ( FE['dim'] , FE['n_elem'] ) )

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

    FE['Ke'] = np.reshape( FE_compute_element_stiffness( FE,OPT,GEOM,FE['material']['C'] ) ,
            ( n_edof,n_edof , FE['n_elem'] ) )


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
    FE['edofMat'] = np.zeros( ( m , n*FE['dim'] ) )

    for elem in range(0,m):
        enodes = FE['elem_node'][:,elem]
        if 2 == FE['dim']:
            edofs = np.reshape( np.stack( ( 2*enodes-1 , 2*enodes ) , axis=1 ).T ,
                ( 1 , n_elem_dof ) )
        elif 3 == FE['dim']:
            edofs = np.reshape( np.stack( ( 3*enodes-2 , 3*enodes-1 , 3*enodes ) , axis=1 ).T ,
                ( 1 , n_elem_dof ) )
        
        FE['edofMat'][elem,:] = edofs

    FE['iK'] = np.reshape( np.kron( FE['edofMat'] , np.ones((n_elem_dof,1)) ).T , 
        ( FE['n_elem']*n_elem_dof**2 , 1 ) )
    FE['jK'] = np.reshape( np.kron( FE['edofMat'] , np.ones((1,n_elem_dof)) ).T , 
        ( FE['n_elem']*n_elem_dof**2 , 1 ) )


def FE_solve(FE,OPT,GEOM):
    # This function solves the system of linear equations arising from the
    # finite element discretization of Eq. (17).  It stores the displacement 
    # and the reaction forces in FE['U'] and FE['P'].

    # global FE
    p = FE['fixeddofs_ind']
    f = FE['freedofs_ind']

    # save the system RHS
    FE['rhs'] = FE['P'](F) - np.matmul( FE['Kfp'] , FE['U'](p) )

    if 'direct' == FE['analysis']['solver']['type']:
        if FE['analysis']['solver'].use_gpu == True:
            print('GPU solver selected, but only available for iterative solver, solving on CPU.')
        #FE['analysis']['solver'].use_gpu = False
        
        #FE['U'](F) = FE['Kff']\FE['rhs']
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
    FE['P'][p] = FE['Kpp']*FE['U'](p) + np.matmul( FE['Kfp'].T , FE['U'][F] )


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
