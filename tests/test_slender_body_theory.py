'''
Created on 29 Jun 2022

@author: lukas
'''

'''
Created on 5 May 2022

@author: lukas
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
from matplotlib import cm
from timeit import Timer

from prolate_spheroid import calc_c_n, calc_c_t
from slender_body_theory.slender_body_theory import DEFAULT_R_MAX, SBT_IntegratorKirchhoff, SBT_MatrixKirchhoff, SBT_IntegratorCosserat,\
    SBT_MatrixCosserat
from slender_body_theory.shape_generator import ShapeGenerator


#------------------------------------------------------------------------------ 
# Test Kirchhoff

def test_straight_line():
    
    # Constant force line density
    f_x = 1
    f_y = 1
    
    f_arr = np.zeros((3, N))
        
    f_arr[0, :] = f_x
    f_arr[1, :] = f_y
    
    x_arr = np.zeros((3, N))
    x_arr[0, :] = np.linspace(0, 1, N)
    L0 = 1.0
        
    sbt = SBT_MatrixKirchhoff(N, L0=L0)
    sbt.update_shape(x_arr)
                
    M = int(1e1)
    
    alpha_arr = np.linspace(1e-5, 1e-3, M)

    c_t_arr = np.zeros(M)
    c_n_arr = np.zeros(M)
     
    c_t_arr_theory = calc_c_t(alpha_arr)
    c_n_arr_theory = calc_c_n(alpha_arr)
        
    for i, alpha in enumerate(alpha_arr): 

        sbt.alpha = alpha

        u_arr_1 = sbt.compute_u_trapezoid(f_arr)
        u_arr_2 = sbt.compute_u_matrix_vector_product(f_arr, method='vectorized')

        u0_1 = u_arr_1[:, 0]
        u0_2 = u_arr_2[:, 0]
        
        # Check that u is a constant
        assert np.sum(np.abs(u_arr_1 - u0_1[:, None])) < 1e-10
        assert np.sum(np.abs(u_arr_2 - u0_2[:, None])) < 1e-10
        
        # Check that both methods yield approximately the same result
        assert np.all(np.isclose(u_arr_1, u_arr_2, atol = 1e-3))
        
        c_t_arr[i] = np.abs(f_x) / np.abs(u0_2[0])  
        c_n_arr[i] = np.abs(f_y) / np.abs(u0_2[1])  
        
    # Check if results agree with theory
    assert np.all(np.isclose(c_t_arr, c_t_arr_theory, atol = 1e-3))
    assert np.all(np.isclose(c_n_arr, c_n_arr_theory, atol = 1e-3))
     
    print('Passed test for straigh configuration and uniform force-line density test!')
     
    return

def test_semicircle():
    
    u_arr = np.linspace(0, 1, N)
    
    n = 2 
    a_n = np.array([1, 1, 1])
    
    def f(u):
        
        p_n = legendre(n)
        x = 2*(u-0.5)
        
        #TODO: Pass coefficents as parameter?       
        f1 = a_n[0]*p_n(x)
        f2 = a_n[1]*p_n(x)
        f3 = a_n[2]*p_n(x)
                
        f = np.vstack((f1,f2,f3))
        
        if isinstance(u, float):
            f = f.flatten()
                
        return f

    f_arr = f(u_arr)
    
    gen = ShapeGenerator(N, centreline = 'semicircle')
                        
    sbt = SBT_MatrixKirchhoff(N)
    sbt.update_shape(gen.r_arr)
        
    inti = SBT_IntegratorKirchhoff(N,
                                   gen.r,
                                   gen.t,
                                   f)                                   


    u_arr_0 = inti.compute_u()
    u_arr_1 = sbt.compute_u_trapezoid(f_arr)
    u_arr_2 = sbt.compute_u_matrix_vector_product(f_arr)

    assert np.all(np.isclose(u_arr_0[0, 1:-1], u_arr_1[0, 1:-1], atol= 1e-3))
    assert np.all(np.isclose(u_arr_0[0, 1:-1], u_arr_2[0, 1:-1], atol= 1e-3))    
    assert np.all(np.isclose(u_arr_1[0, 1:-1], u_arr_2[0, 1:-1], atol= 1e-3))    
    assert np.all(np.isclose(u_arr_1[1, 1:-1], u_arr_2[1, 1:-1], atol= 1e-3))
    assert np.all(np.isclose(u_arr_1[2, 1:-1], u_arr_2[2, 1:-1], atol= 1e-3))
             
    print('Passed test for semicirlce configuration and quadratice force line density')    
    
    gs = plt.GridSpec(3,2)
    
    ax00 = plt.subplot(gs[0,0])
    ax10 = plt.subplot(gs[1,0])
    ax20 = plt.subplot(gs[2,0])
        
    ax01 = plt.subplot(gs[0,1])
    ax11 = plt.subplot(gs[1,1])
    ax21 = plt.subplot(gs[2,1])
                                                   
    ax00.plot(u_arr, u_arr_0[0, :])
    ax00.plot(u_arr, u_arr_1[0, :])    
    ax00.plot(u_arr, u_arr_2[0, :])

    ax10.plot(u_arr, u_arr_0[1, :])
    ax10.plot(u_arr, u_arr_1[1, :])    
    ax10.plot(u_arr, u_arr_2[1, :])

    ax20.plot(u_arr, u_arr_0[2, :])
    ax20.plot(u_arr, u_arr_1[2, :])    
    ax20.plot(u_arr, u_arr_2[2, :])
        
    ax01.plot(u_arr_0[0, 1:-1] -u_arr_1[0, 1:-1])
    ax01.plot(u_arr_0[0, 1:-1] -u_arr_2[0, 1:-1])
                    
    ax11.plot(u_arr_0[1, 1:-1] - u_arr_1[1, 1:-1])
    ax11.plot(u_arr_0[1, 1:-1] - u_arr_2[1, 1:-1])
    
    ax21.plot(u_arr_0[2, 1:-1] - u_arr_1[2, 1:-1])
    ax21.plot(u_arr_0[2, 1:-1] - u_arr_2[2, 1:-1])
        
    plt.show()                
                                                
    return

def test_Kirchhoff_Cosserat():
        
    shape = ShapeGenerator(N, centreline = 'semicircle')

    n = 2 
    a_n = np.array([1, 1, 1])

    def f(u):
        
        p_n = legendre(n)
        x = 2*(u-0.5)
        
        #TODO: Pass coefficents as parameter?       
        f1 = a_n[0]*p_n(x)
        f2 = a_n[1]*p_n(x)
        f3 = a_n[2]*p_n(x)
                
        f = np.vstack((f1,f2,f3))
        
        if isinstance(u, float):
            f = f.flatten()
                
        return f
        
    inti_kirch = SBT_IntegratorKirchhoff(N,
                                         shape.r,
                                         shape.t,
                                         f)


    inti_coss = SBT_IntegratorCosserat(N,                  
                                       shape.r,
                                       shape.s,
                                       shape.e, 
                                       shape.t,
                                       f,
                                       shape.phi, 
                                       shape.d1,
                                       shape.d2,
                                       shape.d3)

    s_arr = np.linspace(0, 1, N)    
    f_arr = f(s_arr)

    sbt_kirch = SBT_MatrixKirchhoff(N)
    sbt_kirch.update_shape(shape.r_arr)

    sbt_coss = SBT_MatrixCosserat(N)
    sbt_coss.update_shape(shape.r_arr, shape.Q_arr)
            
    u1_arr = inti_kirch.compute_u()    
    u2_arr = inti_coss.compute_u()
    u3_arr = sbt_kirch.compute_u_matrix_vector_product(f_arr)
    u4_arr = sbt_coss.compute_u_matrix_vector_product(f_arr)
    
    assert np.allclose(u1_arr, u2_arr, atol = 1e-3)
    assert np.allclose(u3_arr, u4_arr, atol = 1e-3)
    assert np.allclose(u1_arr, u3_arr, atol = 1e-3)

    print('''Passed test: Integrator implementation for Kirchoff and 
    Cosserat rod give same result for semicirlce configuration without shear or stretch
    for a quadratice force line density''')    

    return

# def time_methods():    
#
#     gen = ShapeGenerator(N, centreline = 'semicircle')
#
#     x_arr = gen.x_vec_arr
#
#     sbt = SBT_Matrix(N)
#     sbt.update_shape(x_arr)
#
#     M_vec = sbt.compute_M(method = 'vectorized')
#     M_loop = sbt.compute_M(method = 'loop')
#
#     assert np.all(np.isclose(M_loop, M_vec))
#
#     M = 10
#
#     timer = Timer(lambda: sbt.compute_M(method = 'vectorized'))
#     print(f'Compute M vectorized: {timer.timeit(M)/M}')
#
#     timer = Timer(lambda: sbt.compute_M(method = 'loop'))
#     print(f'Compute M loop: {timer.timeit(M)/M}')
#
#     print('Passed test!')
#
#     return

if __name__ == '__main__':
    
    N = 129
    
    #test_straight_line()
    #test_semicircle()
    test_Kirchhoff_Cosserat()
    
    #time_methods()
    






