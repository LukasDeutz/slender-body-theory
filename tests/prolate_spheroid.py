'''
Created on 29 Jun 2022

@author: lukas

This module implements the tranlational and rotational 
drag coefficients derived for prolate spheroid derived
in 

Chwang, A. T., & Wu, T. (1975). Hydromechanics of low-Reynolds-number flow. 
Part 2. Singularity method for Stokes flows.

'''

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

def calc_e(alpha):     
    '''Eccentricity'''
    
    e = np.sqrt(1 - alpha**2)
    
    return e

def calc_L_e(alpha):

    e = calc_e(alpha)

    return np.log((1+e)/(1-e))

#------------------------------------------------------------------------------ 
# Stokeslet strength for uniform flow

def calc_a_t(alpha):

    e = calc_e(alpha)
    L_e = calc_L_e(alpha)

    return 8*np.pi*e**2*(-2*e + (1 + e**2) * L_e)**(-1)

def calc_a_n(alpha):

    e = calc_e(alpha)
    L_e = calc_L_e(alpha)
        
    return 8*np.pi*2*e**2*(2*e + (3*e**2 - 1)*L_e)**(-1)

#------------------------------------------------------------------------------ 
# Translational drag coefficients for uniform flow

def calc_CF1(alpha):

    e = calc_e(alpha)

    CF1 = 8./3 * e**3 * (-2 * e + (1 + e**2) * np.log((1+e)/(1-e)))**(-1)
    
    return CF1

def calc_CF2(alpha):

    e = calc_e(alpha)

    CF1 = 16./3 * e**3 * (2 * e + (3 * e**2 - 1) * np.log((1+e)/(1-e)))**(-1)
    
    return CF1

def calc_c_t(alpha):
    
    CF1 = calc_CF1(alpha)
    c_t = 3*np.pi*CF1
    
    return c_t

def calc_c_n(alpha):
    
    CF2 = calc_CF2(alpha)
    c_t = 3*np.pi*CF2
    
    return c_t
    
def calc_c_n_slender(alpha):
    
    c_n = 4*np.pi/(np.log(2./alpha) + 0.5)
    
    return c_n
    
def calc_c_t_slender(alpha):
    
    c_t = 2*np.pi/(np.log(2./alpha) - 0.5)
    
    return c_t

#------------------------------------------------------------------------------ 
# Rotational drag coefficients

def calc_C_MR(alpha):
    """Rotional drag minor axis"""
    
    e = calc_e(alpha)
    C_MR = 4./3 * e**3 * (2 - e**2) / (1-e**2) * (-2*e + (1 + e**2)*np.log((1+e)/(1-e)))**(-1)
    
    return C_MR

def calc_C_M0(alpha):
    """Rotational drag major axis"""
        
    e = calc_e(alpha)
    C_M0 = 4./3 * e**3 * (2*e - (1 - e**2)*np.log((1+e)/(1-e)))**(-1)
    
    return C_M0

def calc_gamma_n(alpha):

    c_t = calc_c_t(alpha)
    C_MR = calc_C_MR(alpha)

    gamma_n = np.pi * C_MR * alpha ** 2 / c_t
    
    return gamma_n

def calc_gamma_n_2(alpha):
    
    e = calc_e(alpha)        
    c_t = calc_c_t(alpha)
        
    gamma_3_minus_gamma_3_prime = (2 - e**2)*(-2*e + (1 + e**2) * np.log((1+e)/(1-e)))**(-1)
    
    gamma_n = 4/3 * np.pi * e**3 * gamma_3_minus_gamma_3_prime / c_t
    
    return gamma_n
    
def calc_gamma_t(alpha):

    c_t = calc_c_t(alpha)
    C_M0 = calc_C_M0(alpha)

    gamma_t = np.pi * C_M0 * alpha ** 2 / c_t
    
    return gamma_t

def calc_gamma_t_2(alpha):

    e = calc_e(alpha)
    c_t = calc_c_t(alpha)
        
    gamma_t = 4./3 * np.pi * e**3 * (1 - e**2) * (2*e - (1 - e**2)*np.log((1+e)/(1-e)))**(-1) / c_t

    return gamma_t

def calc_gamma_n_slender(alpha):
    
    c_t = calc_c_t(alpha)
    
    gamma_n = 1./3 * np.pi / (np.log(2.0/alpha) - 0.5) / c_t    
    
    return gamma_n

def calc_gamma_t_slender(alpha):
    
    c_t = calc_c_t(alpha)
    
    gamma_t = 2./3 * np.pi * alpha**2 / c_t
    
    return gamma_t

#------------------------------------------------------------------------------ 
# C elegans

L = 1e3 # Worm length um
R = 35 # Radius

# Prolate spheroid
a = L/2 # major axis
b = R # minor axis

def calc_translational_drag_for_C_elegans():
        
    alpha = b/a
    
    c_n = calc_c_n(alpha)
    c_t = calc_c_t(alpha)
    
    return c_n, c_t, alpha
    
def calc_rotational_drag_for_C_elegans():
            
    alpha = b/a
    
    gamma_n = calc_gamma_n(alpha)
    gamma_t = calc_gamma_t(alpha)
    
    return gamma_n, gamma_t, alpha
    
def calc_slender_drag_coefficients_C_elegans():
    
    alpha = b/a

    c_n = calc_c_n_slender(alpha)
    c_t = calc_c_t_slender(alpha)
    
    gamma_n = calc_gamma_n_slender(alpha)
    gamma_t = calc_gamma_t_slender(alpha)
    
    return c_n, c_t, gamma_n, gamma_t, alpha
    
#------------------------------------------------------------------------------ 
# Plotting

def plot_translational_drag_coefficients_over_alpha():
    
    alpha = np.linspace(0.01, 0.99, int(1e3))

    CF1 = calc_CF1(alpha)
    CF2 = calc_CF2(alpha)

    c_n = calc_c_n(alpha)
    c_t = calc_c_t(alpha)
    
    c_n_slender = calc_c_n_slender(alpha)
    c_t_slender = calc_c_t_slender(alpha)

    c_n_worm, c_t_worm, alpha_worm = calc_translational_drag_for_C_elegans()
    
    gs = plt.GridSpec(4,1)
    
    ax0 = plt.subplot(gs[0])
    ax0.plot(alpha, CF1, label = 'CF1')
    ax0.plot(alpha, CF2, label = 'CF2')
    ax0.legend()
    
    ax1 = plt.subplot(gs[1])        
    ax1.plot(alpha, c_n, c = 'r', label = '$c_n$')
    ax1.plot(alpha, c_t, c = 'b', label = '$\c_t$')
    ax1.plot([alpha_worm, alpha_worm], [c_n_worm, c_t_worm], 'x', c = 'k', ms = 10)        
    
    ax1.legend()

    ax2 = plt.subplot(gs[2])        
    ax2.semilogy(alpha, c_n/c_t, c = 'r', label = '$c_n$')
    ax2.semilogy(alpha_worm, c_n_worm/c_t_worm, c = 'r', label = '$c_n$')
    
    ax2.set_xlabel('alpha')
    ax2.set_ylabel('$c_n / c_t$')

    ax3 = plt.subplot(gs[3])        
    ax3.semilogy(alpha, c_n, c = 'r', label = '$c_n$')
    ax3.semilogy(alpha, c_n_slender, c = 'r', ls = '--', label = '$c_n$')
    ax3.semilogy(alpha, c_t, c = 'b', label = '$\c_t$')
    ax3.semilogy(alpha, c_t_slender, c = 'b', ls = '--', label = '$\c_t$')        
    ax3.semilogy(alpha, c_t_slender, c = 'b', ls = '--', label = '$\c_t$')        
    ax3.semilogy([alpha_worm, alpha_worm], [c_n_worm, c_t_worm], 'x', c = 'k', ms = 10)        

    plt.show()
    
    return
    
def plot_rotational_drag_coefficients_over_alpha():
    
    alpha = np.linspace(0.01, 0.99, int(1e3))
    
    e = calc_e(alpha)
    
    C_MR = calc_C_MR(alpha)
    C_M0 = calc_C_M0(alpha)

    gamma_n = calc_gamma_n(alpha)
    gamma_t = calc_gamma_t(alpha)
    gamma_t_2 = calc_gamma_t_2(alpha)
    gamma_n_2 = calc_gamma_n_2(alpha)

    gamma_n_slender = calc_gamma_n_slender(alpha)
    gamma_t_slender = calc_gamma_t_slender(alpha)

    gamma_n_worm, gamma_t_worm, alpha_worm = calc_rotational_drag_for_C_elegans()

    gs = plt.GridSpec(3,1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    
    ax0.plot(alpha, (1 - e**2)*C_MR, label = '$C_{MR}$')
    ax0.plot(alpha, C_M0, label = '$C_{M0}$')
    ax0.legend()
    
    ax1.plot(alpha, gamma_n, label = '$\gamma_n$')
    ax1.plot(alpha, gamma_n_2, label = '$\gamma_n$')
    ax1.plot(alpha, gamma_n_slender, ls = '--', label = '$\gamma_n$ slender')        
    ax1.plot(alpha, gamma_t, label = '$\gamma_t$')
    ax1.plot(alpha, gamma_t_2, label = '$\gamma_t$')        
    ax1.plot(alpha, gamma_t_slender, ls = '--', label = '$\gamma_t$ slender')
    
    ax1.plot(alpha_worm, gamma_n_worm, 'x', c='k', ms = 10)
    ax1.plot(alpha_worm, gamma_t_worm, 'x', c='k', ms = 10)
            
    ax1.set_xlabel('alpha')
    ax1.legend()

    ax2.semilogy(alpha, gamma_n/gamma_t, label = '$\gamma_n$')
    ax2.semilogy(alpha_worm, gamma_n_worm / gamma_t_worm, 'x', c='k', ms = 10)    
    ax2.set_xlabel('alpha')
    ax2.set_ylabel('\$gamma_n / gamma_t$')

    ax2.legend()
    
    plt.show()

    return

def plot_all_coefficitens():

    alpha = np.linspace(0.01, 0.99, int(1e3))

    c_n = calc_c_n(alpha)
    c_t = calc_c_t(alpha)

    c_n = c_n / c_t
    c_t = np.ones_like(c_n)
    
    gamma_n = calc_gamma_n(alpha)
    gamma_t = calc_gamma_t(alpha)
    
    c_n_worm, c_t_worm, alpha_worm = calc_translational_drag_for_C_elegans()
    gamma_n_worm, gamma_t_worm, alpha_worm = calc_rotational_drag_for_C_elegans()
    
    c_n_worm = c_n_worm/c_t_worm
    c_t_worm = 1
    
    ax = plt.subplot(111)
    ax.plot(alpha, c_n, label = '$c_n$')
    ax.plot(alpha, c_t, label = '$c_t$')
    ax.plot(alpha, gamma_n, label = '$\gamma_n$')
    ax.plot(alpha, gamma_t, label = '$\gamma_t$')
    ax.plot(4*[alpha_worm], [c_n_worm, c_t_worm, gamma_n_worm, gamma_t_worm], 'x', c = 'k')
    
    ax.set_xlabel('alpha')
    ax.legend()
    
    print('Exact:')
    print(f'c_n = {c_n_worm}')
    print(f'c_t = {c_t_worm}')
    print(f'K = {c_n_worm / c_t_worm}')        
    print(f'gamma_n = {gamma_n_worm}')
    print(f'gamma_t ={gamma_t_worm} \n')    
    
    c_n_slender_worm, c_t_slender_worm, gamma_n_slender_worm, gamma_t_slender_worm, alpha_worm = calc_slender_drag_coefficients_C_elegans()
    
    print('Slender limit:')
    print(f'c_n = {c_n_slender_worm}')
    print(f'c_t = {c_t_slender_worm}')
    print(f'K = {c_n_slender_worm / c_t_slender_worm}')    
    print(f'gamma_n = {gamma_n_slender_worm}')
    print(f'gamma_t ={gamma_t_slender_worm} \n')
    
    plt.show()
    
    return

def plot_stokeslet_strength():
    
    alpha = np.linspace(0.001, 0.99, int(1e3))
    
    a_t = calc_a_t(alpha)
    a_n = calc_a_n(alpha)
    
    c_n = calc_c_n(alpha)
    c_t = calc_c_t(alpha)

    plt.plot(alpha, a_t, c = 'k', label= r'$\alpha_t$')
    plt.plot(alpha, a_n, c = 'r', label= r'$\alpha_n$')
    plt.plot(alpha, c_t, ls = '--', c = 'k', label= r'$c_t$')
    plt.plot(alpha, c_n, ls = '--', c = 'r', label= r'$c_n$')

    plt.legend()
    
    plt.show()

    return
    
if __name__ == '__main__':

    #plot_rotational_drag_coefficients_over_alpha()    
    #plot_translational_drag_coefficients_over_alpha()    
    #plot_all_coefficitens()        
    #plot_stokeslet_strength()
    
    pass