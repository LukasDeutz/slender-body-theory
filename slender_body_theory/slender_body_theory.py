'''
Created on 6 May 2022

@author: lukas
'''

# Third-party imports
import numpy as np
from scipy.integrate import quad_vec
from scipy.integrate import trapezoid
from scipy.linalg import block_diag

#------------------------------------------------------------------------------ 
#

DEFAULT_R0 = 0.01 
DEFAULT_L0 = 1.0 

class SBT_Matrix():
    '''
    Implements the slender body theory by Johnson developed for an Kirchhoff
    '''
    
    def __init__(self, 
                 N,
                 R0 = DEFAULT_R0,
                 L0 = DEFAULT_L0,
                 mu = 1.0,
                 C = 1.0):
        '''
        Init class
        
        :param N: Number of vertices
        :param R0: Typical radius
        :param L0: Natural length
        :param mu: Fluid viscosity
        :param C: Normalization
        '''
            
        self.N = N
        
        self.u_arr = np.linspace(0, 1, self.N)
        self.du = self.u_arr[1] - self.u_arr[0] 

        self.R0 = R0             
        self.L0 = L0
                
        self.mu = mu
        self.C = C
        
        self.init_constants()
        
        return
        
        
    def init_constants(self):
        '''
        Init all constant matrices which are independent of the rod's shape 
        to speed up computation
        '''
        
        # sign function |n-n0|
        self.sign_mat = np.ones((self.N, self.N-1))
        n = np.arange(self.N)
        mask = n[:, None] > n
        
        self.sign_mat[mask[:, :self.N-1]] = -1
        
        # mask to identify integral intervals left and right from the "pole"
        mask1 = n[:, None] != n
        mask2 = n[:, None] != n+1
        
        pole_mask =  mask1[:,:-1] & mask2[:,:-1]
        self.pole_mask = pole_mask.astype(np.float64)

                
        self.u_mat_0 = self.u_arr[None, None, None, :-1]
        self.u_mat_1 = self.u_arr[None, None, None, 1:]
        
        du_3 = np.diff(self.u_arr**3) / (self.du * 3)
        du_2 = 0.5 * np.diff(self.u_arr**2) / self.du                                                                            
        
        self.du_2_mat = du_2[None, None, None, :]        
        self.du_3_mat = du_3[None, None, None, :]
        
        #TODO           
        self.M0 = np.zeros((3*self.N, 3*self.N))
        self.M1 = np.zeros((3*self.N, 3*self.N))
        self.M2 = np.zeros((3*self.N, 3*self.N))
                                
        return
                
                                                                  
    def update_shape(self, x_arr):
        '''
        Compute all required shape functions from centreline coordinates
        
        :param x_arr: centreline array
        
        :return x_u_arr: length element array
        :return s_arr: arc-length array
        :return tt_arr: unit tangent vector outer product array
        :return r_mat: euclidean distance matrix
        :return rr_mat: distant vector outer product matrix        
        '''
        
        tt_arr, x_u_arr, x_u_arr_dc = self.compute_tt(x_arr)
        s_arr = self.compute_s(x_u_arr_dc)
        rr_mat, r_mat = self.compute_rr(x_arr)

        self.x_u_arr_dc = x_u_arr_dc
        self.x_u_arr = x_u_arr
        self.s_arr = s_arr
        self.tt_arr = tt_arr
        self.r_mat = r_mat
        self.rr_mat = rr_mat
        
        self.L = s_arr[-1]

        self.I = np.identity(3*self.N)
        self.T = block_diag(*[tt for tt in self.tt_arr])
                         
        return x_u_arr, s_arr, tt_arr, r_mat, rr_mat
                                                                                                                                                                                                                          
    def compute_t(self, x_arr):
        '''
        Compute the unit tangent vector along the centreline
        
        :param x_arr: centreline array
        
        
        :return t_arr: unit tangent array
        :return x_u_arr: length element array 
        '''        
        
        t_arr_dc = np.diff(x_arr, axis = 1) / self.du               
        x_u_arr_dc = np.linalg.norm(t_arr_dc, axis = 0)                        
        t_arr_dc = t_arr_dc/x_u_arr_dc[None, :] 
               
        # The tangent vector is a discontinuous piecewise constant
        # function in each grid interval, i.e. its value at the 
        # vertices is not well defined. To estimate the tangent at 
        # each vertex, we therefore take the average of the tangent 
        # vectors in the neighbouring intervals left and right of 
        # the vertex.  
        t_arr = np.zeros((3, self.N))
        t_arr[:, 0] = t_arr_dc[:, 0]
        t_arr[:, 1:-1] = 0.5 * (t_arr_dc[:, 0:-1] + t_arr_dc[:, 1:])        
        t_arr[:, -1] = t_arr_dc[:, -1]
        t_arr = t_arr / np.linalg.norm(t_arr, axis = 0)[None, :]
        
        # The same is true for the length element
        x_u_arr = np.zeros(self.N)
        x_u_arr[0] = x_u_arr_dc[0]
        x_u_arr[1:-1] = 0.5 * (x_u_arr_dc[0:-1] + x_u_arr_dc[1:])        
        x_u_arr[-1] = x_u_arr_dc[-1]
                                                       
        return t_arr, x_u_arr, x_u_arr_dc
        
    def compute_tt(self, x_arr):
        '''
        Compute outer product of the tangent vector with itself at every grid point
        
        :param x_arr: centreline array
        
        :return tt_arr: unit tangent vector outer product array
        :return x_u_arr: length element array
        '''
                                
        t_arr, x_u_arr, x_u_arr_dc = self.compute_t(x_arr)
                        
        tt_arr = np.zeros((self.N, 3, 3))
                                
        for n, t in enumerate(t_arr.T):            
            
            tt = np.outer(t, t)                      
            tt_arr[n, :, :] =  tt
        
        return tt_arr, x_u_arr, x_u_arr_dc
    
    def compute_s(self, x_u_arr):
        '''
        Computes the arc-length for every grid point 
        
        :param x_u_arr: length element array
        
        :return s_arr: arc-length array
        '''
                            
        s_arr = np.zeros(self.N)
        s_arr[1:] = np.cumsum(x_u_arr) * self.du
                                      
        return s_arr
                
    def compute_rr(self, x_arr):
        '''
        Compute the outer product of the normalized distance vector r at every gridpoint
        relative to the centreline coordinates at a fixed grid point.  
                        
        :param x_arr: centreline array
        
        :returns rr_mat: distant vector outer product matrix  
        :returns r_mat: euclidean distant matrix       
        '''
        
        rr_mat = np.zeros((self.N, 3, 3, self.N))
        r_mat = np.zeros((self.N, self.N))
                                    
        for n, x_n in enumerate(x_arr.T):
            
            r = np.linalg.norm(x_n[:, None] - x_arr, axis = 0)            
            r_vec_arr = (x_n[:, None] - x_arr)   
            
            r_mat[n, :] = r
            
            for i, r_vec in enumerate(r_vec_arr.T):
                
                rr = np.outer(r_vec, r_vec) 
                                                    
                rr_mat[n, :, :, i] = rr / r[i]**3           
            
            rr_mat[n, :, :, n] = 0
                                                                        
        return rr_mat, r_mat 

#------------------------------------------------------------------------------ 
# Compute local term

    def compute_u0(self, f_arr, R_arr = None):
                                                                                                                                                                
        if R_arr is not None:
            l = self.L / 2
            s_arr = self.s_arr - l                        
            ln = np.log( ( 4*(l**2 - s_arr**2 ) ) / R_arr**2 ) 
        else:
            alpha = 2*np.sqrt(self.L0) * self.R0  / self.L**1.5  
            ln = 2*np.log(2./alpha)
        
        u0_arr = np.zeros((3, self.N))
        
        I = np.identity(3)
        
        for n, (tt, f) in enumerate(zip(self.tt_arr, f_arr.T)):
        
            u0_arr[:, n] = ln * np.matmul(I + tt, f) + np.matmul(I - 3*tt, f)
                                                                             
        return u0_arr 

#------------------------------------------------------------------------------ 
# Compute integrands

    def compute_integrand_1(self, f_arr):
                    
        I_mat = np.zeros((self.N, 3, self.N))

        for n, (s_n, f_n) in enumerate(zip(self.s_arr, f_arr.T)):
                             
            I = (f_arr - f_n[:, None]) / np.abs(self.s_arr - s_n)[None, :] * self.x_u_arr[None, :] 
                                                                
            I_mat[n, :, :] = I
            
            I_mat[n, :, n] = 0
            
                                    
        return I_mat

    def compute_A(self):
        '''
        Computes the matrix elements of matrix A at every grid point. 
        Matrix A is defined as the matrix in the integrand of the second 
        slender body integral.                 
        '''
        
        s_arr = self.s_arr
        tt_arr = self.tt_arr
        r_mat = self.r_mat
        rr_mat = self.rr_mat
        
        I = np.identity(3)
        
        A_mat = np.zeros_like(rr_mat)

        for n, (rr_arr, r_arr, tt_n, s_n) in enumerate(zip(rr_mat, r_mat, tt_arr, s_arr)):
            
            for i, (rr, r) in enumerate(zip(rr_arr.T, r_arr)):
                
                A = I / r + rr - ( I + tt_n) / np.abs(s_arr[i] - s_n)                 
                A_mat[n, :, :, i] = A
                
                if i == n:
                    A_mat[n, :, :, i] = 0
                                    
        return A_mat
    
    def compute_A_vectorized(self):
        '''Computes the matrix elements of matrix A at every grid point. 
        Matrix A is defined as the matrix in the integrand of the second 
        slender body integral.'''    
        
        s_arr = self.s_arr
        tt_arr = self.tt_arr
        r_mat = self.r_mat
        rr_mat = self.rr_mat
        
        s_mat = np.vstack(self.N*[s_arr])
        s_minus_s_n = (s_mat - s_mat.T)[:, None, None, :]
                
        I = np.identity(3)[None, :, :, None]
        r_mat = r_mat[:, None, None, :]
        tt_mat = tt_arr[:, :, :, None]
                                        
        A_mat = I / r_mat + rr_mat - ( I + tt_mat) / np.abs(s_minus_s_n)                 

        n = np.arange(self.N)
                        
        A_mat[n, :, :, n] = 0
                                    
        return A_mat
            
    def compute_integrand_2(self, f_arr):
                
        A_mat = self.compute_A()
        
        I_mat = np.zeros((self.N, 3, self.N))
        
        for n, A_arr in enumerate(A_mat):
            
            for i, (A, f) in enumerate(zip(A_arr.T, f_arr.T)):
            
                I_mat[n, :, i] = np.matmul(A, f) * self.x_u_arr[i]
                
        return I_mat
        
#------------------------------------------------------------------------------ 
# Compute nonlocal term (trapezoid integration)

    def compute_u1_trapezoid(self, f_arr):
        
        I_mat = self.compute_integrand_1(f_arr)
        
        ide = np.identity(3)
        
        # integrate        
        I_arr = trapezoid(I_mat, dx = self.du, axis = 2).T
        
        u_arr = np.zeros((3, self.N))
        
        for n, (tt, I) in enumerate(zip(self.tt_arr, I_arr.T)):
            
            u_arr[:, n] = np.matmul(ide + tt, I)
        
        return u_arr
    
    def compute_u2_trapezoid(self, f_arr):
        
        I_mat = self.compute_integrand_2(f_arr)
    
        # integrate        
        u_arr = trapezoid(I_mat, dx = self.du, axis = 2).T
        
        return u_arr
            
    def compute_u_trapezoid(self, f_arr, R_arr = None):
                                                                
        u0_arr = self.compute_u0(f_arr, R_arr)                                
        u1_arr = self.compute_u1_trapezoid(f_arr)
        u2_arr = self.compute_u2_trapezoid(f_arr)
            
        # normalize
        
        # f in sbt is force on fluid, drag force is equal and opposite
        # This is why we need a minus sign here        
        u_arr = - self.C * (u0_arr + u1_arr + u2_arr) / (np.pi * 8 * self.mu)
        
        return u_arr
    
#------------------------------------------------------------------------------ 
# Assemble matrices

    def compute_M(self, R_arr = None, method = 'vectorized'):
        '''
        Compute M matrix for the given shape represented by the centreline coordinates
        '''
            
        M0 = self.compute_M0(R_arr)
                
        if method == 'vectorized':
            M1 = self.compute_M1_vectorized()
            M2 = self.compute_M2_vectorized()
        elif method == 'loop':
            M1 = self.compute_M1()
            M2 = self.compute_M2()
                                            
        M =  - self.C * ( M0 + M1 + M2 ) / (np.pi * 8 * self.mu)
        
        return M


    def compute_M0(self, R_arr = None):
        '''                
        :param tt_arr: unit tangent vector outer product array 
        :param s_arr: arc-length array
        :param L: body length
        :param R_arr: cross-sectional radius array
        
        :return M0: matrix represensation of local sbt term
        '''
                                                                                                                                                      
        if R_arr is not None:
            l = self.L / 2
            s_arr = self.s_arr - l                        
            ln = np.log( ( 4*(l**2 - s_arr**2 ) ) / R_arr**2 ) 
        else:
            alpha = 2 * np.sqrt(self.L0) * self.R0 / self.L**1.5  
            ln = 2*np.log(2./alpha)
                        
        M0 = ln * (self.I + self.T) + (self.I - 3*self.T)
                                                                 
        return M0

    def compute_M1(self):
        '''
         Compute matrix M1 which accounts for the first of the two slender body integrals
        
        :return M1: matrix representation of the first sbt integral
        '''                
        s_arr = self.s_arr
        x_u_arr = self.x_u_arr_dc
        tt_arr = self.tt_arr
        
        a_s_arr = np.diff(s_arr) / self.du

        M1 = np.zeros((3*self.N, 3*self.N))
        
        n_pad = 0
                                                                                                                                      
        #TODO: Can this be vectorized?
        for n, s_n in enumerate(s_arr):
        
            n_pad = n*3
        
            sign = np.ones(self.N - 1)
            sign[:n] = -1
        
            log = np.log((s_arr[1:] - s_n) / (s_arr[:-1] - s_n))
        
            m_f_1 = sign * x_u_arr / a_s_arr * (( s_arr[1:] - s_n ) / (a_s_arr * self.du) * log - 1)
            m_f_2 = sign * x_u_arr / a_s_arr * ( 1 + ( s_n - s_arr[:-1] ) / (a_s_arr * self.du) * log)
            m_f_n = - sign * x_u_arr / a_s_arr * log
        
            #TODO: Include intervals left and right of the pole        
            if n != 0:            
                m_f_1[n-1] = 0
                m_f_2[n-1] = 0
                m_f_n[n-1] = 0
            if n != self.N-1:
                m_f_1[n] = 0
                m_f_2[n] = 0
                m_f_n[n] = 0
        
            m_f_1_pad = np.zeros(3*(self.N-1))
            m_f_1_pad[::3] = m_f_1
        
            m_f_2_pad = np.zeros(3*(self.N-1))
            m_f_2_pad[::3] = m_f_2
        
            M1[n_pad, :-3] = m_f_1_pad            
            M1[n_pad, 3:] += m_f_2_pad
            M1[n_pad, n_pad] += np.sum(m_f_n)
        
            M1[n_pad+1, :-3] = np.roll(m_f_1_pad,1)            
            M1[n_pad+1, 3:] += np.roll(m_f_2_pad,1)
            M1[n_pad+1, n_pad+1] += np.sum(m_f_n)
        
            M1[n_pad+2, :-3] = np.roll(m_f_1_pad,2)            
            M1[n_pad+2, 3:] += np.roll(m_f_2_pad,2)
            M1[n_pad+2, n_pad+2] += np.sum(m_f_n)
        
        T = block_diag(*[tt for tt in tt_arr])
        I_plus_T = np.identity(3*self.N) + T

        M1 = np.matmul(I_plus_T, M1)
        
        return M1

    def compute_M1_vectorized(self):
        '''
         Compute matrix M1 which accounts for the first of the two slender body integrals
        
        :return M1: matrix representation of the first sbt integral
        '''                
        s_arr = self.s_arr
        x_u_arr = self.x_u_arr_dc
        
        x_u_mat = x_u_arr[None, :]
        
        a_s_arr = np.diff(s_arr) / self.du
        a_s_mat = a_s_arr[None, :]

        s_n_mat = np.vstack((self.N - 1) * [s_arr]).T
        s_mat = np.vstack(self.N*[s_arr])
                        
        s_0_minus_s_n = s_mat[:, :-1] - s_n_mat
        s_1_minus_s_n = s_mat[:, 1:] - s_n_mat

        eps = np.finfo(float).eps

        #np.fill_diagonal(s_0_minus_s_n, eps)        
        s_0_minus_s_n += eps
        
        X = s_1_minus_s_n / s_0_minus_s_n  
        
        X += eps            
        #fill_off_diagonal(X, eps, -1)

        log = np.log(X)
                                        
        m_f_1 = self.sign_mat * x_u_mat / a_s_mat * (s_1_minus_s_n / (a_s_mat * self.du) * log - 1)
        m_f_2 = self.sign_mat * x_u_mat / a_s_mat * (1 - s_0_minus_s_n / (a_s_mat * self.du) * log)              
        m_f_n = - self.sign_mat * x_u_mat / a_s_mat * log
                
        m_f_1 = m_f_1 * self.pole_mask 
        m_f_2 = m_f_2 * self.pole_mask 
        m_f_n = m_f_n * self.pole_mask
        
        m_f_n = np.sum(m_f_n, axis = 1)
        
        self.M1[:,:] = 0
        #pad with zeros        
        self.M1[::3, :-3:3] += m_f_1
        self.M1[1::3,1:-2:3] += m_f_1
        self.M1[2::3,2:-1:3] += m_f_1
                
        self.M1[::3, 3::3] += m_f_2
        self.M1[1::3, 4::3] += m_f_2
        self.M1[2::3, 5::3] += m_f_2
        
        np.fill_diagonal(self.M1, self.M1.diagonal() + np.repeat(m_f_n, 3))

        self.M1 = np.matmul(self.I + self.T, self.M1)
        
        return self.M1
                    
    def compute_M2(self):
        '''        
        :return M2: matrix representation of the second sbt integral
        '''
        A_mat = self.compute_A()
        
        x_u_arr = self.x_u_arr_dc
            
        a_A_mat = np.diff(A_mat, axis = 3) / self.du
        b_A_mat = (A_mat[:, :, :, :-1]*self.u_arr[None, None, None, 1:] 
                 - A_mat[:, :, :, 1:]*self.u_arr[None, None, None, :-1]) / self.du
        
        u = self.u_arr

        du_3 = np.diff(u**3) / (self.du * 3)
        du_2 = 0.5 * np.diff(u**2) / self.du                                                                            
                        
        m_f_1_mat = np.zeros((self.N, 3, 3, self.N-1))
        m_f_2_mat = np.zeros((self.N, 3, 3, self.N-1))
                    
        #TODO: Can this be vectorized?
        for n, (a_A_arr, b_A_arr) in enumerate(zip(a_A_mat, b_A_mat)):
                        
            for i, (a_A, b_A) in enumerate(zip(a_A_arr.T, b_A_arr.T)):
                             
                m_f_1 = du_2[i] * u[i+1] * a_A + u[i+1] * b_A  - du_3[i] * a_A - du_2[i] * b_A 
                m_f_2 = du_3[i] * a_A + du_2[i] * b_A - du_2[i] * u[i] * a_A - u[i] * b_A 
                                                                                                                            
                m_f_1_mat[n, :, :, i] = m_f_1 * x_u_arr[i] 
                m_f_2_mat[n, :, :, i] = m_f_2 * x_u_arr[i]

                
        M2 = np.zeros((3*self.N, 3*self.N))

        for n, (m_f_1, m_f_2) in enumerate(zip(m_f_1_mat, m_f_2_mat)):
            
            n_pad = 3 * n
                                            
            m_f_1 = m_f_1.reshape((3, 3*(self.N-1)), order = 'F')
            m_f_2 = m_f_2.reshape((3, 3*(self.N-1)), order = 'F')
                                                                                                                                                                                                
            M2[n_pad:n_pad+3, 0:-3] += m_f_1
            M2[n_pad:n_pad+3, 3:] += m_f_2 
        
        return M2
            
    def compute_M2_vectorized(self):
        
        A_mat = self.compute_A_vectorized()
                
        x_u_arr = self.x_u_arr_dc
            
        x_u_mat = x_u_arr[None, None, None, :]
            
        a_A_mat = np.diff(A_mat, axis = 3) / self.du
        b_A_mat = (A_mat[:, :, :, :-1]*self.u_mat_1 
                 - A_mat[:, :, :, 1:]*self.u_mat_0) / self.du
        
        m_f_1_mat = x_u_mat*(self.du_2_mat * self.u_mat_1 * a_A_mat 
                     + self.u_mat_1 * b_A_mat - self.du_3_mat * a_A_mat
                     - self.du_2_mat * b_A_mat)
        
        m_f_2_mat = x_u_mat*(self.du_3_mat * a_A_mat + self.du_2_mat * b_A_mat
                     - self.du_2_mat * self.u_mat_0 * a_A_mat 
                     - self.u_mat_0 * b_A_mat)
                                                
        m_f_1_mat = m_f_1_mat.reshape((self.N, 3, 3*(self.N-1)), order = 'F')
        m_f_2_mat = m_f_2_mat.reshape((self.N, 3, 3*(self.N-1)), order = 'F')

        m_f_1_mat = m_f_1_mat.reshape((3*self.N, 3*(self.N-1)), order = 'C')
        m_f_2_mat = m_f_2_mat.reshape((3*self.N, 3*(self.N-1)), order = 'C')
        
        self.M2[:,:] = 0
        self.M2[:, 0:-3] += m_f_1_mat
        self.M2[:, 3:] += m_f_2_mat
                                                                                                                                                                                                                 
        return self.M2        
                    
#------------------------------------------------------------------------------ 
# Compute sbt (matrix)

    def compute_u_matrix_vector_product(self, f_arr, R_arr = None, method = 'vectorized'):
        
        F = f_arr.flatten(order = 'F')
        M = self.compute_M(R_arr, method = method)
        
        u_arr = np.matmul(M, F).reshape((3, self.N), order = 'F')
                  
        return u_arr
    
    def compute_f_matrix_vector_product(self, u_arr, R_arr = None):
        
        U = u_arr.flatten(order = 'F')
        M = self.compute_M(R_arr)
        
        f_arr = np.linalg.solve(M, U).reshape((3, self.N), order = 'F')
        
        return f_arr
        
class SBT_Integrator():
    '''
    Implements the slender body theory by Johnson
    
    Computes the nonlocal integrals for a given function 
    for centreline, length-element, tangent  and force 
    density using Gaussian quadrature.   
    '''
    
    def __init__(self, 
                 N, 
                 x, 
                 abs_x_u, 
                 s, 
                 t, 
                 f,
                 R0 = DEFAULT_R0,
                 L0 = DEFAULT_L0,
                 mu = 1.0,
                 C = 1.0, 
                 eps = None):
        '''
        
        :param N: Number of vertices
        :param x: Centreline
        :param abs_x_u: Length-element
        :param s: Arc-length
        :param t: Tangent
        :param f: Force line density
        :param R0: Typical radius 
        :param L0: Natural length
        :param mu: Fluid viscosity
        :param C: Normalization
        :param eps: Integral boundaries are set to "pole" -/+ epsilon
        '''
                
        self.N = N
        self.u_arr = np.linspace(0, 1, N)
        
        self.x = x
        self.abs_x_u = abs_x_u
        self.t = t
        self.s = s
        self.f = f
        
        self.eps = eps        
        self.L0 = L0
        self.R0 = R0
        
        self.mu = mu
        self.C = C
        
        self.vectorize_shape_functions()
                                                  
        return

    def vectorize_shape_functions(self):
        
        self.x_arr = self.x(self.u_arr)
        self.s_arr = self.s(self.u_arr)
        self.t_arr = self.t(self.u_arr)
        self.f_arr = self.f(self.u_arr)                                       
        self.tt_arr = self.compute_tt()
        
        self.L = self.s_arr[-1]
                        
        return
                
    def compute_tt(self):
                
        tt_arr = np.zeros((self.N, 3, 3))
                            
        for n, t in enumerate(self.t_arr.T):

            tt = np.outer(t, t)                      
            tt_arr[n, :, :] =  tt
    
        return tt_arr

#------------------------------------------------------------------------------ 
# 
    
    def compute_u(self, R_arr = None):
    
        u0_arr = self.compute_u0(self.f_arr,
                                 R_arr = R_arr)
    
        u1_arr = self.compute_u1()
        u2_arr = self.compute_u2()
        
        # Normalize

        # f in sbt is force on fluid, drag force is equail and opposite
        # This is why we need a minus sign here        
        u_arr = - self.C * (u0_arr + u1_arr + u2_arr) / (8 * np.pi * self.mu)
        
        return u_arr

#------------------------------------------------------------------------------ 
# Compute local term


    #TODO: Move this outside of the class
    def compute_u0(self, f_arr, R_arr = None, L = None):
        '''
        Local contribution 
        
        :param f_arr:
        :param R_arr:
        :param L:
        '''
                                                                                                                                                                                    
        if R_arr is not None:
            l = self.L/2            
            s_arr = self.s_arr - l                        
            ln = np.log( ( 4*(l**2 - s_arr**2 ) ) / R_arr**2 ) 
        else:
            # Incomrepssibility constraint requires that cross-sectional 
            # radius adjusts to length changes.
            # Here, we assume that the stretch/compression ratio
            # is constant along the centreline                                    
            alpha = 2*np.sqrt(self.L0) * self.R0  / self.L**1.5  
            ln = 2*np.log(2./alpha)
        
        u0_arr = np.zeros((3, self.N))
        
        I = np.identity(3)
        
        for n, (tt, f) in enumerate(zip(self.tt_arr, f_arr.T)):
        
            u0_arr[:, n] = ln * np.matmul(I + tt, f) + np.matmul(I - 3*tt, f)
                                                                             
        return u0_arr 


                            
#------------------------------------------------------------------------------ 
# Compute sbt integral

    def compute_integral_1(self):
        '''
        First sbt integral, numerical integration from analytic expression 
        '''
                      
        I_arr = np.zeros((3, self.N))
                
        for n, (u_n, s, f) in enumerate(zip(self.u_arr, self.s_arr, self.f_arr.T)):            
            print(f'Solve integral {n} out of {self.N}')
            
            integrand = lambda u: (self.f(u) - f) / np.abs(self.s(u) - s) * self.abs_x_u(u) 

            I = np.zeros(3)
            
            if n > 0:
                if self.eps is not None:                
                    u_n_minus = u_n - self.eps
                else:                
                    u_n_minus = self.u_arr[n - 1]                
                    
                I += quad_vec(integrand, 0, u_n_minus)[0]
        
            if n < self.N - 1:                            
                if self.eps is not None:                
                    u_n_plus = u_n + self.eps                
                else:                
                    u_n_plus = self.u_arr[n+1]
                                      
                I += quad_vec(integrand, u_n_plus, 1)[0]
                
            I_arr[:, n] = I
                                        
        return I_arr
            
    
    def compute_u1(self):
        '''
        First, sbt-integral numerical integration from analytic expression 
        '''
        
        I_arr = self.compute_integral_1()
        
        u1_arr = np.zeros((3, self.N))
        
        for i, (tt, I1) in enumerate(zip(self.tt_arr, I_arr.T)):
            
            u1 = np.matmul(np.identity(3) + tt, I1)
            u1_arr[:, i] = u1
    
        return u1_arr  
    
    def compute_u2(self):
        '''
        Compute second slender body integral numerically using from the given analytic expressions
        for the centreline and the force line density vector 
        '''
                
        u2_arr = np.zeros((3, self.N))
            
        I = np.identity(3)
                
        for n, (u_n, x_n, s_n) in enumerate(zip(self.u_arr, self.x_arr.T, self.s_arr)):
            print(f'Solve integral {n} out of {self.N}')
                        
            r = lambda u: np.linalg.norm(self.x(u) - x_n)
            r_vec = lambda u: (self.x(u) - x_n) / r(u)
            rr = lambda u: np.outer(r_vec(u), r_vec(u))
            
            t_n = self.t(u_n)            
            tt_n = np.outer(t_n, t_n)

            A = lambda u: ( I + rr(u) ) / r(u) - ( I + tt_n ) / np.abs(s_n - self.s(u))
                                                                                                 
            integrand = lambda u: np.matmul(A(u), self.f(u)) * self.abs_x_u(u)
            
            u2 = np.zeros(3)
            
            if n > 0:
                if self.eps is not None:                
                    u_n_minus = u_n - self.eps
                else:                
                    u_n_minus = self.u_arr[n - 1]                
                    
                u2 += quad_vec(integrand, 0, u_n_minus)[0]
        
            if n < self.N - 1:            
                if self.eps is not None:                
                    u_n_plus = u_n + self.eps                
                else:                
                    u_n_plus = self.u_arr[n+1]
                                      
                u2 += quad_vec(integrand, u_n_plus, 1)[0]
                
            u2_arr[:, n] = u2
                                            
        return u2_arr   
