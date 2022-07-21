'''
Created on 29 Jun 2022

@author: lukas
'''

import numpy as  np


class Shape_Generator():
    '''
    Generates different centreline configurations 
    '''
    
    def __init__(self, 
                 N, 
                 centreline = 'semicircle',
                 L = 1.0):                 
        '''        
        :param N:
        :param L:
        :param centreline:
        '''                       
        self.N = N
        
        self.u_arr = np.linspace(0, 1, N)
        self.ds_h = self.u_arr[1] - self.u_arr[0]
                        
        self.L = L
                                        
        if centreline == 'straight-line':
            self.init_straight_line()                
        elif centreline == 'semicircle':
            self.init_semi_circle()                        
        elif centreline == 'wave':
            #TODO
            assert False, 'Not supported yet' 
            pass
        else:
            assert False, 'wrong centreline'

        self.init_vector_representation()
                                
        return
    
    def init_straight_line(self):
        '''
        Init analytic expressions for centreline, arc-length and tangent 
        for straight-line configuration
        '''
                        
        self.r = lambda u: u
        self.y = lambda u: 0*u
        self.z = lambda u: 0*u
        
        self.s = lambda u: u
        
        self.x_vec = lambda u: np.array([self.r(u), self.y(u), self.z(u)])
        
        self.tx = lambda u: 1 + 0*u
        self.ty = lambda u: 0*u
        self.tz = lambda u: 0*u
        
        self.t = lambda u: np.array([self.tx(u), self.ty(u), self.tz(u)])
        
        return
                    
    def init_semi_circle(self):
        '''
        Init analytic expressions for the centreline, arc-length and tangent 
        for semicircle configuration         
        '''        
        self.R = self.L/np.pi
        self.k = self.L/self.R

        self.r = lambda u: self.R*np.cos(self.k*u)
        self.y = lambda u: self.R*np.sin(self.k*u)
        self.z = lambda u: 0*u
        
        self.s = lambda u: self.R*self.k * u
        
        self.x_vec = lambda u: np.array([self.r(u), self.y(u), self.z(u)])
        
        self.tx = lambda u: - np.sin(self.k*u)
        self.ty = lambda u: + np.cos(self.k*u)
        self.tz = lambda u: 0 * u 

        self.e = lambda u: self.R*self.k 

        self.t = lambda u: np.array([self.tx(u), self.ty(u), self.tz(u)])
                                                                      
    def init_vector_representation(self):
        '''
        Computes vector representation of the centreline, arc-length and tangent 
        from analytic expressions
        '''
                                
        self.x_arr = self.r(self.u_arr)
        self.y_arr = self.y(self.u_arr)
        self.z_arr = self.z(self.u_arr)
                
        self.x_vec_arr = np.vstack((self.x_arr, self.y_arr, self.z_arr))
        
        self.tx_arr = self.tx(self.u_arr)
        self.ty_arr = self.ty(self.u_arr)
        self.tz_arr = self.tz(self.u_arr)
               
        self.t_arr = np.vstack((self.tx_arr, self.ty_arr, self.tz_arr))
                
        self.s_arr = self.s(self.u_arr)

        self.tt_arr = self.compute_tt()
                        
        return

    def compute_tt(self):
        
        tt_arr = np.zeros((self.N, 3, 3))
                            
        for n, t in enumerate(self.t_arr.T):

            tt = np.outer(t, t)                      
            tt_arr[n, :, :] =  tt
    
        return tt_arr

