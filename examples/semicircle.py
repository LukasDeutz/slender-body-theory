'''
Created on 29 Jun 2022

@author: lukas
'''


def plot_semi_circle():
    '''
    Plot centreline velocity for constant force applied to semi-circle
    '''

    gs = plt.GridSpec(3, 2)
    
    ax0 = plt.subplot(gs[0,:])
    ax10 = plt.subplot(gs[1,0])
    ax11 = plt.subplot(gs[1,1])

    ax20 = plt.subplot(gs[2,0])
    ax21 = plt.subplot(gs[2,1])

    def f(u):
        
        f_y = -1
                                
        f1 = np.zeros_like(u)
        f2 = f_y * np.ones_like(u)
        f3 = np.zeros_like(u)
                
        f = np.vstack((f1,f2,f3))
        
        if isinstance(u, float):
            f = f.flatten()
                
        return f
        
    N_arr = [50, 100, 250, 500, 1000]
        
    u_arr_mat_0 = []
    u_arr_mat_1 = []
        
    u_arr_mat = []
        
    for N in N_arr:

        u_arr = np.linspace(0, 1, N) # material coordinate
        f_arr = f(u_arr) # force line density

        gen = ShapeGenerator(N, centreline = 'semicircle')
            
        x_arr = gen.x_vec_arr
                
        sbt = SBT_Matrix(N)
        sbt.update_shape(x_arr)
        
        inti = SBT_Integrator(N,
                              gen.x_vec,
                              gen.e,
                              gen.s,
                              gen.t,
                              f)

        u_arr_0 = inti.compute_u()        
        u_arr_1 = sbt.compute_u_matrix_vector_product(f_arr)
               
        u_arr_mat_0.append(u_arr_0)
        u_arr_mat_1.append(u_arr_1)
        u_arr_mat.append(u_arr)

    # Plot centreline, force and velocity vectors
    ax0.plot(x_arr[0, :], x_arr[1, :], c = 'r')        
    
    M = 10
    skip = int(np.size(x_arr,1)/M)
            
    for x,y,u_x,u_y in zip(x_arr[0, ::skip], x_arr[1, ::skip], u_arr_0[0, ::skip], u_arr_0[1, ::skip]):
                        
        ax0.arrow(x,y, 0.1*u_x, 0.1*u_y, head_width = 0.01)
        ax0.arrow(x,y, 0, -0.1, head_width = 0.01)
            
    ax0.set_xlabel('$x$', fontsize = 20)
    ax0.set_ylabel('$y$', fontsize = 20)
       
                
    ax10.plot(u_arr, u_arr_0[0, :], label = r'$u_x$')
    ax10.plot(u_arr, u_arr_0[1, :], label = r'$u_y$')
    #ax10.plot(u_arr, u_arr_0[2, :], label = r'$u_z$')    
    ax10.set_ylabel(r'$u$', fontsize = 20)
    ax10.set_ylabel(r'$\mathbf{u}$', fontsize = 20)
    ax10.legend(fontsize = 15)
    ax10.set_xticks([])
                                
    # Plot force density    
    ax11.plot(u_arr, f_arr[0, :], label = r'$f_x$')
    ax11.plot(u_arr, f_arr[1, :], label = r'$f_y$')
    #ax00.plot(u_arr, f_arr[2, :], label = r'$f_z$')    
    ax11.set_ylabel(r'$\mathbf{f}$', fontsize = 20)
    ax11.legend(fontsize = 15)    
    ax11.set_xticks([])
                                            
    c_list_1 = color_list_from_values(N_arr, cmap = cm.get_cmap('plasma'))
    c_list_2 = color_list_from_values(N_arr, cmap = cm.get_cmap('plasma'))
    
    
    for N, c1, c2, u_arr, u_arr_0, u_arr_1 in zip(N_arr, c_list_1, c_list_2, u_arr_mat, u_arr_mat_0, u_arr_mat_1):
        
        err_x = np.abs(u_arr_1[0,:] - u_arr_0[0,:]) / np.abs(u_arr_0[0,:])
        err_y = np.abs(u_arr_1[1,:] - u_arr_0[1,:]) / np.abs(u_arr_0[1,:])
        #err_z = np.abs(u_arr_1[2,:] - u_arr_0[2,:]) / np.abs(u_arr_0[2,:])
                
        ax20.semilogy(u_arr, err_x, ls = '-', c = c1, label = f'$N={N}$')
        ax21.semilogy(u_arr, err_y, ls = '-', c = c2, label = f'$N={N}$')
        #ax11.semilogy(u_arr, err_z, label = f'$N={N}')

    ax20.set_ylabel(r'rel. err $\mathbf{u_x}$', fontsize = 20)    
    ax21.set_ylabel(r'rel. err $\mathbf{u_y}$', fontsize = 20)        
    ax20.legend()
                            
    #plt.tight_layout()
      
    plt.show()
