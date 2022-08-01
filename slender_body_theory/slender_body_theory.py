'''
Created on 6 May 2022

@author: lukas
'''
#Build-in imports
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
warnings.filterwarnings("ignore", message="invalid value encountered in log")


# Third-party imports
import numpy as np
from scipy.integrate import quad_vec
from scipy.integrate import trapezoid
from scipy.linalg import block_diag

#------------------------------------------------------------------------------ 
#

DEFAULT_R_MAX = 0.01 
DEFAULT_L0 = 1.0 

class SBT_Matrix(ABC):
    '''
    Base class which computes matrix representation of the discretized slender body theory
    '''
    def __init__(self, 
                 N,
                 r_max = DEFAULT_R_MAX,
                 L0 = DEFAULT_L0,
                 mu = 1.0,
                 C = 1.0):
        '''
        Init class
        
        :param N: Number of vertices
        :param r_max: Maximal radius
        :param L0: Natural length
        :param mu: Fluid viscosity
        :param C: Normalization
        '''
            
        self.N = N
                    
        # This is the dimensionless reference arc-length
        # which lives on the unit interval
        self.s_h_arr = np.linspace(0, 1, self.N)
        self.ds_h = self.s_h_arr[1] - self.s_h_arr[0] 

        self.r_max = r_max
        self.L0 = L0
        self.alpha = 2 * r_max / self.L0 
                
        self.mu = mu
        self.C = C

        self.init_constants()
                
        return
    
    def update_shape(self, r_arr, norm_tan = False):
        
        self.t_arr, self.tt_arr, self.e_dc_arr, self.e_arr, self.s_arr = self.compute_t(r_arr, norm_tan)
        self.xx_mat, self.x_mat = self.compute_x(r_arr)
                
        return
         
    def init_constants(self):
        
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
                        
        self.u_mat_0 = self.s_h_arr[None, None, None, :-1]
        self.u_mat_1 = self.s_h_arr[None, None, None, 1:]
        
        du_3 = np.diff(self.s_h_arr**3) / (self.ds_h * 3)
        du_2 = 0.5 * np.diff(self.s_h_arr**2) / self.ds_h                                                                            
        
        self.du_2_mat = du_2[None, None, None, :]        
        self.du_3_mat = du_3[None, None, None, :]
                        
        self.M0 = np.zeros((3*self.N, 3*self.N))
        self.M1 = np.zeros((3*self.N, 3*self.N))
        self.M2 = np.zeros((3*self.N, 3*self.N))

        return

    def compute_u_matrix_vector_product(self, f_arr, method = 'vectorized'):
        
        F = f_arr.flatten(order = 'F')
        M = self.compute_M(method = method)
        
        u_arr = np.matmul(M, F).reshape((3, self.N), order = 'F')
                  
        return u_arr
    
    def compute_f_matrix_vector_product(self, u_arr, R_arr = None):
        
        U = u_arr.flatten(order = 'F')
        M = self.compute_M(R_arr)
        
        f_arr = np.linalg.solve(M, U).reshape((3, self.N), order = 'F')
        
        return f_arr


    def compute_M(self, method = 'vectorized'):
        '''
        Compute M matrix for the given shape represented by the centreline coordinates
        '''
            
        M0 = self.compute_M0()
                
        if method == 'vectorized':
            M1 = self.compute_M1()
            M2 = self.compute_M2()
        elif method == 'loop':
            
            if self.type == 'Cosserat':
                assert False, 'Loop calculation is not supported for a Cosserat rod'
                            
            M1 = self.compute_M1_loop()
            M2 = self.compute_M2_loop()
                                            
        M =  - self.C * ( M0 + M1 + M2 ) / (np.pi * 8) #* self.mu)
        
        return M
        
    @abstractmethod
    def compute_M0(self):
        
        pass

    @abstractmethod    
    def compute_A2(self):
        
        pass
    
    @abstractmethod    
    def compute_A1(self):
        
        pass
    
    def compute_M1(self):
        '''
         Compute matrix M1 representation of the first slender body integral
        
        :return M1: matrix representation of the first sbt integral
        '''                
        s_arr = self.s_arr
        
        eps_mat = self.e_dc_arr[None, :]
        
        a_s_arr = np.diff(s_arr) / self.ds_h
        a_s_mat = a_s_arr[None, :]

        s_n_mat = np.vstack((self.N - 1) * [s_arr]).T
        s_mat = np.vstack(self.N*[s_arr])
                        
        s_0_minus_s_n = s_mat[:, :-1] - s_n_mat
        s_1_minus_s_n = s_mat[:, 1:] - s_n_mat

        eps = np.finfo(float).eps

        s_0_minus_s_n += eps

        # eps is added to avoid division by zero and 
        # a negative algorithm of the logarithm at the "pole"               
        log = np.log(s_1_minus_s_n / (s_0_minus_s_n + eps) + eps)
                                        
        m_f_1 = self.sign_mat * eps_mat / a_s_mat * (s_1_minus_s_n / (a_s_mat * self.ds_h) * log - 1)
        m_f_2 = self.sign_mat * eps_mat / a_s_mat * (1 - s_0_minus_s_n / (a_s_mat * self.ds_h) * log)              
        m_f_n = - self.sign_mat * eps_mat / a_s_mat * log
                
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

        A_mat = self.compute_A2()
                        
        e_mat = self.e_dc_arr[None, None, None, :]
            
        a_A_mat = np.diff(A_mat, axis = 3) / self.ds_h
        b_A_mat = (A_mat[:, :, :, :-1]*self.u_mat_1 
                 - A_mat[:, :, :, 1:]*self.u_mat_0) / self.ds_h
        
        m_f_1_mat = e_mat*(self.du_2_mat * self.u_mat_1 * a_A_mat 
                     + self.u_mat_1 * b_A_mat - self.du_3_mat * a_A_mat
                     - self.du_2_mat * b_A_mat)
        
        m_f_2_mat = e_mat*(self.du_3_mat * a_A_mat + self.du_2_mat * b_A_mat
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
                   
    def compute_t(self, r_arr, norm_tan = False):
        '''
        Compute the unit tangent vector along the centreline
        
        :param r_arr: centreline array
        :param norm_tan: If true, tangent gets normalized
                
        :return t_arr: tangent array
        :return tt_arr: outer product tangent array
        :return e_arr: axial stretch compretion ratio array 
        :return s_arr: current arc-length array         
        '''        
        
        t_arr_dc = np.diff(r_arr, axis = 1) / self.ds_h 
        e_dc_arr = np.linalg.norm(t_arr_dc, axis = 0)
                                                                                            
        # The tangent vector is a discontinuous piecewise constant
        # function in each grid interval, i.e. its value at the 
        # vertices is not well defined. To estimate the tangent at 
        # each vertex, we therefore take the average of the tangent 
        # vectors in the neighbouring intervals left and right of 
        # the vertex. This is equivalent to approximating the first 
        # derivative by the central difference.    
        t_arr = np.zeros((3, self.N))
        #t_arr[:, 0] = t_arr_dc[:, 0]
        t_arr[:, 0] = 0.5*(-3*r_arr[:, 0] + 4*r_arr[:, 1] - 1*r_arr[:, 2])/self.ds_h
        t_arr[:, 1:-1] = 0.5 * (t_arr_dc[:, 0:-1] + t_arr_dc[:, 1:])         
        #t_arr[:, -1] = t_arr_dc[:, -1]
        t_arr[:, -1] = 0.5*(3*r_arr[:, -1] - 4*r_arr[:, -2] + 1*r_arr[:, -3])/self.ds_h

        # Axial stretch compression ratio
        e_arr = np.linalg.norm(t_arr, axis = 0)

        if norm_tan:
            t_arr = t_arr / e_arr[None, :]
                        
        tt_arr = np.zeros((self.N, 3, 3))
                                
        for n, t in enumerate(t_arr.T):            
            
            tt = np.outer(t, t)                      
            tt_arr[n, :, :] =  tt

        if self.type == 'Kirchhoff':
            s_arr = self.s_h_arr
                    
        elif self.type == 'Cosserat':
        
            s_arr = np.zeros(self.N)
            s_arr[1:] = np.cumsum(e_dc_arr) * self.ds_h
            # current length        
            self.L = s_arr[-1]
                                                                                                                         
        return t_arr, tt_arr, e_dc_arr, e_arr, s_arr
                            
    def compute_x(self, r_arr):
        '''
        Compute the outer product of the normalized distance vector r at every gridpoint
        relative to the centreline coordinates at a fixed grid point.  
                        
        :param x_arr: centreline array
        
        :returns rr_mat: distant vector outer product matrix  
        :returns r_mat: euclidean distant matrix       
        '''
        
        xx_mat = np.zeros((self.N, 3, 3, self.N))
        x_mat = np.zeros((self.N, self.N))
                                    
        for n, r_n in enumerate(r_arr.T):
            
            r = np.linalg.norm(r_n[:, None] - r_arr, axis = 0)            
            r_vec_arr = (r_n[:, None] - r_arr)   
            
            x_mat[n, :] = r
            
            for i, x_vec in enumerate(r_vec_arr.T):
                                
                if i == n:
                    xx_mat[n, :, :, i] = 0
                else:
                    xx = np.outer(x_vec, x_vec) 
                    xx_mat[n, :, :, i] = xx / r[i]**3           
                                                                                    
        return xx_mat, x_mat 
    
class SBT_MatrixCosserat(SBT_Matrix):
    '''
    Implements the slender body theory for special a Cosserat rod 
    developed by Garg and Kumar.             
    '''
    
    def __init__(self,
                 N,
                 r_max = DEFAULT_R_MAX,
                 L0 = DEFAULT_L0,
                 mu = 1.0,
                 C = 1.0):
        
        super().__init__(N, r_max, L0, mu, C)
    
        self.type = 'Cosserat'
        
        return
    
    def init_constants(self):
        
        SBT_Matrix.init_constants(self)
        
        #Elliptic shape function for the cross-sectional radius
        self.phi_arr = np.sqrt(1 - (2*self.s_h_arr - 1)**2)        
                        
        return 
            
    def update_shape(self, r_arr, Q_arr):
        '''
        
        :param r_arr:
        :param Q_arr:
        '''

        SBT_Matrix.update_shape(self, r_arr, norm_tan = False)
                
        d1 = Q_arr[0, :, :]
        d2 = Q_arr[1, :, :]
        d3 = Q_arr[2, :, :]

        # t1/t2/t2 are tangent components in local reference frame
        self.t1_arr = np.sum(d1 * self.t_arr, axis = 0)
        self.t2_arr = np.sum(d2 * self.t_arr, axis = 0)
        self.t3_arr = np.sum(d3 * self.t_arr, axis = 0)
                
        self.d1d3_arr = np.zeros((self.N, 3, 3))
        self.d2d3_arr = np.zeros((self.N, 3, 3))
        self.d3d3_arr = np.zeros((self.N, 3, 3))
                                
        for n, (d1,d2,d3)  in enumerate(zip(d1.T, d2.T, d3.T)):            
            
            self.d1d3_arr[n, :, :] =  np.outer(d1,d3)
            self.d2d3_arr[n, :, :] =  np.outer(d2,d1)
            self.d3d3_arr[n, :, :] =  np.outer(d3,d3)
                
        self.I = block_diag(*[np.identity(3)/t3 for t3 in self.t3_arr.T])
        self.T = block_diag(*[tt/t3**3 for (tt, t3) in zip(self.tt_arr, self.t3_arr.T)])
        
        return

    def compute_M0(self):
        '''                        
        :return M0: matrix represensation of local sbt term
        '''
        
        l = 0.5*self.L
        z = self.s_arr - l                 
        # z = 2*self.s_arr - 1
                
        ln = np.log(16 * self.e_arr * self.t3_arr**2 * (l**2 - z**2) / (self.alpha**2 * self.phi_arr**2))

        #TODO:
        # force rod at the ends to be a prolate spheroid
        ln[0]  = 2*np.log(2./self.alpha)
        ln[-1] = 2*np.log(2./self.alpha)

        D1D3 = block_diag(*[3*t1*d1d3/t3**2 for (d1d3,t1,t3) in zip(self.d1d3_arr, self.t1_arr.T, self.t3_arr.T)])
        D2D3 = block_diag(*[3*t2*d2d3/t3**2 for (d2d3,t2,t3) in zip(self.d2d3_arr, self.t2_arr.T, self.t3_arr.T)])        
        D3D3 = block_diag(*[3*d3d3/t3 for (d3d3,t3) in zip(self.d3d3_arr, self.t3_arr.T)])

        ln = np.repeat(ln, 3)
                                                
        M0 = ln[:, None] * (self.I + self.T) + (self.I - D3D3) - (D1D3 + D1D3.T + D2D3 + D2D3.T) 

        if not np.isfinite(M0).all():
            assert False
                                                             
        return M0

    def compute_A1(self):
        
        s_mat = np.vstack(self.N*[self.s_arr])
        s_minus_s_n = (s_mat - s_mat.T)[:, None, None, :]
                
        tt = self.tt_arr[:, :, :, None]
        t3 = self.t3_arr[None, None, None, :]
              
        I = np.identity(3)[None, :, :, None]
                                                
        A1 = (I / t3  + tt / t3**3 ) / np.abs(s_minus_s_n)                 

        n = np.arange(self.N)
                        
        A1[n, :, :, n] = 0
                                    
        return A1
    
    def compute_A2(self):
                                                
        s_mat = np.vstack(self.N*[self.s_arr])
        s_minus_s_n = (s_mat - s_mat.T)[:, None, None, :]
                
        x  = self.x_mat[:, None, None, :]
        tt = self.tt_arr[:, :, :, None]
        t3 = self.t3_arr[None, None, None, :]
              
        I = np.identity(3)[None, :, :, None]
                                                
        A2 = I / x + self.xx_mat - ( I / t3 + tt/ t3**3 ) / np.abs(s_minus_s_n)                 

        n = np.arange(self.N)
                        
        A2[n, :, :, n] = 0
                                    
        return A2
        
                            
class SBT_MatrixKirchhoff(SBT_Matrix):
    '''
    Implements the slender body theory for a Kirchhoff rod developed
    by Johnson
    '''
    
    def __init__(self, 
                 N,
                 r_max = DEFAULT_R_MAX,
                 L0 = DEFAULT_L0,
                 mu = 1.0,
                 C = 1.0):
        '''
        Init class
        
        :param N: Number of vertices
        :param r_max: Maximal natural radius
        :param L0: Natural length
        :param mu: Fluid viscosity
        :param C: Normalization
        '''
            
        super().__init__(N, r_max, L0, mu, C)
                                                    
        self.type = 'Kirchhoff'
                    
        return
            
    def init_constants(self):
        '''
        Init all constant matrices which are independent of the rod's shape 
        to speed up computation
        '''
        
        SBT_Matrix.init_constants(self)

        self.I = np.identity(3*self.N)
                                
        return    
    
    def update_shape(self, r_arr):
        '''
        Compute all required shape functions from centreline coordinates
        
        :param x_arr: centreline array
        
        :return x_u_arr: length element array
        :return s_arr: arc-length array
        :return tt_arr: unit tangent vector outer product array
        :return r_mat: euclidean distance matrix
        :return rr_mat: distant vector outer product matrix        
        '''
        
        SBT_Matrix.update_shape(self, r_arr, norm_tan = True)
                
        self.T = block_diag(*[tt for tt in self.tt_arr])
                         
        return 
                                                                                                                                                                                                                                                                      
#------------------------------------------------------------------------------ 
# Compute local term

    def compute_u0(self, f_arr):
        
        # assuming an ellpitic shape function for the radius                                                                                                                                                                                ln = 2*np.log(2./self.alpha)
        ln = 2*np.log(2./self.alpha)
        
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
            
            I = (f_arr - f_n[:, None]) / np.abs(self.s_arr - s_n)[None, :] * self.e_arr[None, :] 
                                                                
            I_mat[n, :, :] = I            
            I_mat[n, :, n] = 0
                                                
        return I_mat

    def compute_A1(self):
        
        s_mat = np.vstack(self.N*[self.s_arr])
        s_minus_s_n = (s_mat - s_mat.T)[:, None, None, :]
                
        tt = self.tt_arr[:, :, :, None]
              
        I = np.identity(3)[None, :, :, None]
                                                
        A1 = (I + tt) / np.abs(s_minus_s_n)                 

        n = np.arange(self.N)
                        
        A1[n, :, :, n] = 0
                                    
        return A1

    def compute_A2_loop(self):
        '''
        Computes the matrix elements of matrix A at every grid point. 
        Matrix A is defined as the matrix in the integrand of the second 
        slender body integral.                 
        '''
        
        s_arr = self.s_arr
        tt_arr = self.tt_arr
        x_mat = self.x_mat
        xx_mat = self.xx_mat
        
        I = np.identity(3)
        
        A_mat = np.zeros_like(xx_mat)

        for n, (xx_arr, x_arr, tt_n, s_n) in enumerate(zip(xx_mat, x_mat, tt_arr, s_arr)):
            
            for i, (xx, x) in enumerate(zip(xx_arr.T, x_arr)):
                
                
                if i == n:
                    A_mat[n, :, :, i] = 0
                else:                    
                    A = I / x + xx - ( I + tt_n) / np.abs(s_arr[i] - s_n)                 
                    A_mat[n, :, :, i] = A
                                    
        return A_mat
        
    def compute_A2(self):
        '''Computes the matrix elements of matrix A at every grid point. 
        Matrix A is defined as the matrix in the integrand of the second 
        slender body integral.'''    
                
        s_mat = np.vstack(self.N*[self.s_arr])
        s_minus_s_n = (s_mat - s_mat.T)[:, None, None, :]
        
        x  = self.x_mat[:, None, None, :]
        tt = self.tt_arr[:, :, :, None]
                
        I = np.identity(3)[None, :, :, None]
                                        
        A_mat = I / x + self.xx_mat - ( I + tt) / np.abs(s_minus_s_n)                 

        n = np.arange(self.N)
                        
        A_mat[n, :, :, n] = 0
                                    
        return A_mat
            
    def compute_integrand_2(self, f_arr):
                
        A_mat = self.compute_A2_loop()
        
        I_mat = np.zeros((self.N, 3, self.N))
        
        for n, A_arr in enumerate(A_mat):
            
            for i, (A, f) in enumerate(zip(A_arr.T, f_arr.T)):
            
                I_mat[n, :, i] = np.matmul(A, f) * self.e_arr[i]
                
        return I_mat
        
#------------------------------------------------------------------------------ 
# Compute nonlocal term (trapezoid integration)

    def compute_u1_trapezoid(self, f_arr):
        
        I_mat = self.compute_integrand_1(f_arr)
        
        ide = np.identity(3)
        
        # integrate        
        I_arr = trapezoid(I_mat, dx = self.ds_h, axis = 2).T
        
        u_arr = np.zeros((3, self.N))
        
        for n, (tt, I) in enumerate(zip(self.tt_arr, I_arr.T)):
            
            u_arr[:, n] = np.matmul(ide + tt, I)
        
        return u_arr
    
    def compute_u2_trapezoid(self, f_arr):
        
        I_mat = self.compute_integrand_2(f_arr)
    
        # integrate        
        u_arr = trapezoid(I_mat, dx = self.ds_h, axis = 2).T
        
        return u_arr
            
    def compute_u_trapezoid(self, f_arr):
                                                                
        u0_arr = self.compute_u0(f_arr)                                
        u1_arr = self.compute_u1_trapezoid(f_arr)
        u2_arr = self.compute_u2_trapezoid(f_arr)
            
        # normalize      
        u_arr = - self.C * (u0_arr + u1_arr + u2_arr) / (np.pi * 8 * self.mu)
        
        return u_arr
    
    def compute_M0(self, R_arr = None):
        '''                
        :param tt_arr: unit tangent vector outer product array 
        :param s_arr: arc-length array
        :param L: body length
        :param R_arr: cross-sectional radius array
        
        :return M0: matrix represensation of local sbt term
        '''
                                                                                                                                                      
        if R_arr is not None:
            
            l = 0.5*self.L0            
            R_arr = R_arr / l
            
            ln = np.log( (4 - self.s_arr**2 ) / R_arr**2 ) 
        else:
            # elliptic radius amplitude function
            ln = 2*np.log(2./self.alpha)
                        
        M0 = ln * (self.I + self.T) + (self.I - 3*self.T)
                                                                 
        return M0

    def compute_M1_loop(self):
        '''
         Compute matrix M1 which accounts for the first of the two slender body integrals
        
        :return M1: matrix representation of the first sbt integral
        '''                
        s_arr = self.s_arr
        eps_arr = self.e_dc_arr
        
        a_s_arr = np.diff(s_arr) / self.ds_h

        M1 = np.zeros((3*self.N, 3*self.N))
        
        n_pad = 0
                                                                                                                                      
        #TODO: Can this be vectorized?
        for n, s_n in enumerate(s_arr):
        
            n_pad = n*3
        
            sign = np.ones(self.N - 1)
            sign[:n] = -1
        
            log = np.log((s_arr[1:] - s_n) / (s_arr[:-1] - s_n))
        
            m_f_1 = sign * eps_arr / a_s_arr * (( s_arr[1:] - s_n ) / (a_s_arr * self.ds_h) * log - 1)
            m_f_2 = sign * eps_arr / a_s_arr * ( 1 + ( s_n - s_arr[:-1] ) / (a_s_arr * self.ds_h) * log)
            m_f_n = - sign * eps_arr / a_s_arr * log
        
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
        
        M1 = np.matmul(self.I + self.T, M1)
        
        return M1
                    
    def compute_M2_loop(self):
        '''        
        :return M2: matrix representation of the second sbt integral
        '''
        A_mat = self.compute_A2_loop()
        
        eps_arr = self.e_dc_arr
            
        a_A_mat = np.diff(A_mat, axis = 3) / self.ds_h
        b_A_mat = (A_mat[:, :, :, :-1]*self.u_arr[None, None, None, 1:] 
                 - A_mat[:, :, :, 1:]*self.u_arr[None, None, None, :-1]) / self.ds_h
        
        u = self.u_arr

        du_3 = np.diff(u**3) / (self.ds_h * 3)
        du_2 = 0.5 * np.diff(u**2) / self.ds_h                                                                            
                        
        m_f_1_mat = np.zeros((self.N, 3, 3, self.N-1))
        m_f_2_mat = np.zeros((self.N, 3, 3, self.N-1))
                    
        #TODO: Can this be vectorized?
        for n, (a_A_arr, b_A_arr) in enumerate(zip(a_A_mat, b_A_mat)):
                        
            for i, (a_A, b_A) in enumerate(zip(a_A_arr.T, b_A_arr.T)):
                             
                m_f_1 = du_2[i] * u[i+1] * a_A + u[i+1] * b_A  - du_3[i] * a_A - du_2[i] * b_A 
                m_f_2 = du_3[i] * a_A + du_2[i] * b_A - du_2[i] * u[i] * a_A - u[i] * b_A 
                                                                                                                            
                m_f_1_mat[n, :, :, i] = m_f_1 * eps_arr[i] 
                m_f_2_mat[n, :, :, i] = m_f_2 * eps_arr[i]

                
        M2 = np.zeros((3*self.N, 3*self.N))

        for n, (m_f_1, m_f_2) in enumerate(zip(m_f_1_mat, m_f_2_mat)):
            
            n_pad = 3 * n
                                            
            m_f_1 = m_f_1.reshape((3, 3*(self.N-1)), order = 'F')
            m_f_2 = m_f_2.reshape((3, 3*(self.N-1)), order = 'F')
                                                                                                                                                                                                
            M2[n_pad:n_pad+3, 0:-3] += m_f_1
            M2[n_pad:n_pad+3, 3:] += m_f_2 
        
        return M2
                                

class SBT_Integrator(ABC):
    '''
    Base class to solve slender body theory integrals numerically
    
    Computes the nonlocal integrals for a given function 
    for centreline, length-element, tangent  and force 
    density using Gaussian quadrature.   
    '''
    def __init__(self,
                 N, 
                 r, 
                 t,
                 f,
                 phi = None, 
                 r_max = DEFAULT_R_MAX,
                 mu = 1.0,
                 C = 1.0, 
                 eps = None):

        self.N = N
        self.s_h_arr = np.linspace(0, 1, N)
        self.ds = self.s_h_arr[1] - self.s_h_arr[0]
                                        
        self.r = r
        self.t = t
        
        self.phi = phi
        self.r_max = r_max
                
        self.f = f
                
        self.eps = eps
        
        self.mu = mu
        self.C = C
        
        self.vectorize_shape_functions()

    def vectorize_shape_functions(self):
        
        self.r_arr = self.r(self.s_h_arr)
        self.t_arr = self.t(self.s_h_arr)
                
        self.f_arr = self.f(self.s_h_arr)                                       
        self.tt_arr = self.compute_tt()
                                
        return

    def compute_tt(self):
                
        tt_arr = np.zeros((self.N, 3, 3))
                            
        for n, t in enumerate(self.t_arr.T):

            tt = np.outer(t, t)                      
            tt_arr[n, :, :] =  tt
    
        return tt_arr

#------------------------------------------------------------------------------ 
# Compute sbt integral

    def compute_u(self):
    
        u0_arr = self.compute_u0()
    
        u1_arr = self.compute_u1()
        u2_arr = self.compute_u2()
        
        # Normalize
        u_arr = - self.C * (u0_arr + u1_arr + u2_arr) / (8 * np.pi * self.mu)
        
        return u_arr

    def compute_integral_1(self):
        '''
        First sbt integral, numerical integration from analytic expression 
        '''
                      
        I_arr = np.zeros((3, self.N))
                
        for n, (s_h_n, s, f) in enumerate(zip(self.s_h_arr, self.s_arr, self.f_arr.T)):            
            print(f'Solve integral {n} out of {self.N}')
            
            integrand = lambda s_h: (self.f(s_h) - f) / np.abs(self.s(s_h) - s) * self.e(s_h) 

            I = np.zeros(3)
            
            if n > 0:
                if self.eps is not None:                
                    s_h_minus = s_h_n - self.eps
                else:                
                    s_h_minus = self.s_h_arr[n - 1]                
                    
                I += quad_vec(integrand, 0, s_h_minus)[0]
        
            if n < self.N - 1:                            
                if self.eps is not None:                
                    s_h_plus = s_h_n + self.eps                
                else:                
                    s_h_plus = self.s_h_arr[n+1]
                                      
                I += quad_vec(integrand, s_h_plus, 1)[0]
                
            I_arr[:, n] = I
                                        
        return I_arr

class SBT_IntegratorCosserat(SBT_Integrator):
    
    
    def __init__(self,
                 N,                  
                 r,
                 s,
                 e, 
                 t,
                 f,
                 phi, 
                 d1,
                 d2,
                 d3,                                  
                 r_max = DEFAULT_R_MAX,
                 L0 = DEFAULT_L0,
                 mu = 1.0,
                 C = 1.0, 
                 eps = None):
        
        self.L0 = L0
        self.s = s
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3                        
        self.e = e
        
        super().__init__(N, r, t, f, phi, r_max, mu, C, eps)
                
        self.alpha = 2 * self.r_max / self.L
                                                         
        return

    def vectorize_shape_functions(self):
    
        SBT_Integrator.vectorize_shape_functions(self)

        # Cross-sectional radius, shape function
        self.phi_arr = self.phi(self.s_h_arr)
                
        # Current arc-length
        self.s_arr = self.s(self.s_h_arr)    
        self.e_arr = self.e(self.s_h_arr)
        self.L = self.s_arr[-1]
                                                                            
        self.d1_arr = self.d1(self.s_h_arr)
        self.d2_arr = self.d2(self.s_h_arr)
        self.d3_arr = self.d3(self.s_h_arr)
                
        self.t1_arr = np.sum(self.d1_arr*self.t_arr, axis = 0)
        self.t2_arr = np.sum(self.d2_arr*self.t_arr, axis = 0)
        self.t3_arr = np.sum(self.d3_arr*self.t_arr, axis = 0)
                     
        self.d1d3 = self.compute_dxdy(self.d1_arr, self.d3_arr, 3*self.t1_arr/self.t3_arr**2)
        self.d2d3 = self.compute_dxdy(self.d2_arr, self.d3_arr, 3*self.t2_arr/self.t3_arr**2)
        self.d3d3 = self.compute_dxdy(self.d3_arr, self.d3_arr, 3/self.t3_arr)

        self.d3d1 = np.transpose(self.d1d3, axes = (0,2,1))
        self.d3d2 = np.transpose(self.d2d3, axes = (0,2,1))

        self.tt_arr = self.tt_arr / self.t3_arr[:, None, None]
        self.I_arr = np.reshape(self.N*[np.identity(3)], (self.N, 3, 3)) / self.t3_arr[:, None, None]  
                                                             
        return

    def compute_dxdy(self, dx_arr, dy_arr, c_arr):
                
        dxdy_arr = np.zeros((self.N, 3, 3))
                            
        for n, (dy, dx, c) in enumerate(zip(dx_arr.T, dy_arr.T, c_arr)):

            dxdy = np.outer(dx, dy)                      
            dxdy_arr[n, :, :] = c * dxdy  
    
        return dxdy_arr
                
    def compute_u0(self):
        '''
        Local contribution 
        
        :param f_arr:
        :param R_arr:
        :param L:
        '''
        z = 2*self.s_arr - 1
                        
        beta = self.L / self.L0        
        ln = np.log(4 * self.e_arr * self.t3_arr**2 * (beta**2 - z**2) / (self.alpha**2 * self.phi_arr**2))
        
        # force rod at the ends to be a prolate spheroid
        ln[0]  = 2*np.log(2./self.alpha)
        ln[-1] = 2*np.log(2./self.alpha)
        
        
        M0_arr = (ln[:,None,None] * (self.I_arr + self.tt_arr) + (self.I_arr - self.d3d3) 
                - self.d1d3 + self.d3d1 - self.d2d3 + self.d3d2) 
          
        u0_arr = np.zeros((3, self.N))
          
        for i, (M0, f) in enumerate(zip(M0_arr, self.f_arr.T)):
                        
            u0_arr[:, i] = np.matmul(M0, f)
                        
        return u0_arr

    def compute_u1(self):
        '''
        First, sbt-integral numerical integration from analytic expression 
        '''
        
        inte_arr = self.compute_integral_1()
        
        u1_arr = np.zeros((3, self.N))
        
        for i, (tt, I, inte) in enumerate(zip(self.tt_arr, self.I_arr, inte_arr.T)):
            
            u1 = np.matmul(I + tt, inte)
            u1_arr[:, i] = u1
    
        return u1_arr  

    def compute_u2(self):
        '''
        Compute second slender body integral numerically using from the given analytic expressions
        for the centreline and the force line density vector 
        '''
                
        u2_arr = np.zeros((3, self.N))
            
        I = np.identity(3)
                
        for n, (s_h_n, r_n, s_n) in enumerate(zip(self.s_h_arr, self.r_arr.T, self.s_arr)):
            print(f'Solve integral {n} out of {self.N}')
                        
            x = lambda s_h: np.linalg.norm(self.r(s_h) - r_n)
            x_vec = lambda s_h: (self.r(s_h) - r_n) / x(s_h)
            xx = lambda s_h: np.outer(x_vec(s_h), x_vec(s_h))
            
            t_n = self.t(s_h_n)
            t3_n = self.t3_arr[n]
                                                
            tt_n = np.outer(t_n, t_n)

            A = lambda s_h: ( I + xx(s_h) ) / x(s_h) - ( I/t3_n + tt_n/t3_n**3 ) / np.abs(s_n - self.s(s_h))
                                                                                                 
            integrand = lambda s_h: np.matmul(A(s_h), self.f(s_h)) * self.e(s_h)
            
            u2 = np.zeros(3)
            
            if n > 0:
                if self.eps is not None:                
                    s_h_n_minus = s_h_n - self.eps
                else:                
                    s_h_n_minus = self.s_h_arr[n - 1]                
                    
                u2 += quad_vec(integrand, 0, s_h_n_minus)[0]
        
            if n < self.N - 1:            
                if self.eps is not None:                
                    s_h_n_plus = s_h_n + self.eps                
                else:                
                    s_h_n_plus = self.s_h_arr[n+1]
                                      
                u2 += quad_vec(integrand, s_h_n_plus, 1)[0]
                
            u2_arr[:, n] = u2
                                            
        return u2_arr

        
class SBT_IntegratorKirchhoff(SBT_Integrator):
    '''
    Implements the slender body theory by Johnson
    
    Computes the nonlocal integrals for a given function 
    for centreline, length-element, tangent  and force 
    density using Gaussian quadrature.   
    '''
    
    def __init__(self, 
                 N, 
                 r, 
                 t,
                 f,
                 phi = None,                  
                 r_max = DEFAULT_R_MAX,
                 L = DEFAULT_L0,
                 mu = 1.0,
                 C = 1.0, 
                 eps_int = None):
        '''
        
        :param N: Number of vertices
        :param r: Centreline
        :param e: Length-element
        :param s: Arc-length
        :param t: Tangent
        :param f: Force line density
        :param r_max: Typical radius 
        :param L0: Natural length
        :param mu: Fluid viscosity
        :param C: Normalization
        :param eps: Integral boundaries are set to "pole" -/+ epsilon
        '''
        

        super().__init__(N, r, t, f, phi, r_max, mu, C, eps_int)
           
        self.s = lambda s: s
        self.e = lambda s: 1.0 + 0*s
        self.s_arr = self.s(self.s_h_arr)
                      
        self.L = L
        self.alpha = 2 * self.r_max / self.L
                                                  
        return

#------------------------------------------------------------------------------ 
# Compute local term

    def vectorize_shape_functions(self):
        
        SBT_Integrator.vectorize_shape_functions(self)

        if self.phi is None:
            self.phi_arr = None
        else:
            self.phi_arr = self.phi(self.s_h_arr)

    def compute_u0(self, phi_arr = None):
        '''
        Local contribution 
        
        :param f_arr:
        :param R_arr:
        :param L:
        '''
        
        if phi_arr is None:                                                                                                                                                                             
            # assume elliptic shape function                    
            ln = 2*np.log(2./self.alpha)
        else:
            # Wrong
            ln = np.log(4*(1 - 4*self.s_arr**2) / (self.alpha**2 * phi_arr ** 2))
                
        u0_arr = np.zeros((3, self.N))
        
        I = np.identity(3)
        
        for n, (tt, f) in enumerate(zip(self.tt_arr, self.f_arr.T)):
        
            u0_arr[:, n] = ln * np.matmul(I + tt, f) + np.matmul(I - 3*tt, f)
                                                                             
        return u0_arr 
    
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
                
        for n, (s_h_n, r_n, tt_n) in enumerate(zip(self.s_h_arr, self.r_arr.T, self.tt_arr)):
            print(f'Solve integral {n} out of {self.N}')
                        
            x_norm = lambda s_h: np.linalg.norm(self.r(s_h) - r_n)
            x = lambda s_h: (self.r(s_h) - r_n) / x_norm(s_h)
            xx = lambda s_h: np.outer(x(s_h), x(s_h))
            

            A = lambda s_h: ( I + xx(s_h) ) / x_norm(s_h) - ( I + tt_n ) / np.abs(s_h_n - self.s(s_h))
                                                                                                 
            integrand = lambda s_h: np.matmul(A(s_h), self.f(s_h)) * self.e(s_h)
            
            u2 = np.zeros(3)
                        
            if n > 0:
                if self.eps is not None:                
                    s_h_n_minus = s_h_n - self.eps
                else:                
                    s_h_n_minus = self.s_h_arr[n - 1]                
                    
                u2 += quad_vec(integrand, 0, s_h_n_minus)[0]
        
            if n < self.N - 1:            
                if self.eps is not None:                
                    s_h_n_plus = s_h_n + self.eps                
                else:                
                    s_h_n_plus = self.s_h_arr[n+1]
                                      
                u2 += quad_vec(integrand, s_h_n_plus, 1)[0]
                
            u2_arr[:, n] = u2
                                            
        return u2_arr

    
                 
                 
