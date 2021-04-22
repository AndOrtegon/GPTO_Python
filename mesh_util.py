import numpy as np

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
        node_coords = np.stack( ( xx.flatten(), yy.flatten() ) , axis = 1 ) 
    elif 3 == FE['dim']:
        xx, yy, zz  = np.meshgrid( x_i[0] , x_i[1] , x_i[2] )
        node_coords = np.stack( ( xx.flatten(), yy.flatten(), zz.flatten() ) , axis = 1 )

    ## define element connectivity
    elem_mat = np.zeros( ( FE['n_elem'] , 2**FE['dim'] ) )
    nelx = elements_per_side[0]
    nely = elements_per_side[1]

    if 2 == FE['dim']:
        row= np.array(range(0,nely) ).reshape(-1,1)
        col= np.array(range(0,nelx) ).reshape(1,-1)

        n1 = row + col*(nely+1)
        n2 = row + (col+1)*(nely+1)
        n3 = n2 + 1
        n4 = n1 + 1
        elem_mat = np.stack( ( n1.flatten(), n2.flatten(), n3.flatten(), n4.flatten() ) )
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
        elem_mat = np.stack( ( n1.flatten(), n2.flatten(), n3.flatten(), n4.flatten(), n5.flatten(), n6.flatten(), n7.flatten(), n8.flatten() ) )

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
