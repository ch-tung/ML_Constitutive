import numpy as np
import numba as nb

@nb.jit()
def RSHEscore(epsilon):
    e_xx = epsilon[0]
    e_yy = epsilon[1]
    e_xy = epsilon[2]
    
    p = np.zeros(5)
    
    n_theta = 100 # number of slices used for integration
    d_theta = 2*np.pi/n_theta
    
    for i in range(n_theta):
        theta = i*d_theta
        
        # original disc
        x0 = np.cos(theta)
        y0 = np.sin(theta)
        
        # affine transformation
        x = x0*(1+e_xx) + y0*e_xy
        y = y0*(1+e_yy) + x0*e_xy
        
        rho = np.sqrt(x**2 + y**2)
        
        Y_00  = np.sqrt(1/np.pi)/2
        Y_22n = np.sqrt(15/np.pi)/2*x*y/rho**2
        Y_22p = np.sqrt(15/np.pi)/4*(x**2 - y**2)/rho**2
        Y_44n = np.sqrt(35/np.pi)*3/4*x*y*(x**2 - y**2)/rho**4
        Y_44p = np.sqrt(35/np.pi)*3/16*(x**2*(x**2-3*y**2)-y**2*(3*x**2-y**2))/rho**4
        
        # area averaged Y_lm
        p = p + np.array([Y_00 *rho**2/2*d_theta,
                          Y_22n*rho**2/2*d_theta,
                          Y_22p*rho**2/2*d_theta,
                          Y_44n*rho**2/2*d_theta,
                          Y_44p*rho**2/2*d_theta])
    
    # normalization
    p = p/np.pi/(np.sqrt(1/np.pi)/2)
    return p