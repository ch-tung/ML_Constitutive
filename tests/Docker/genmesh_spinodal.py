from fenics import *
from dolfin_adjoint import *
import pygmsh_mesh_functions
from pygmsh_mesh_functions import *
import meshio
import numpy as np
import matplotlib.pyplot as plt

import skimage.io
from skimage.measure import find_contours, subdivide_polygon, approximate_polygon

# img2mesh.py: python script converting 2D binaary image to FEM mesh
from img2mesh import *

alpha = 2e-2

Kx = 4
Ky = 3

def genmesh_spinodal(alpha = 2.5e-2, Kx = 4, Ky = 3,
                        n_wave = 500, n_x = 100, n_y = 100, 
                        filename = "test_spinodal.xdmf"):
    
    x = np.arange(n_x)/n_x
    y = np.arange(n_y)/n_y

    xx, yy = np.meshgrid(x,y)

    phi = np.zeros_like(xx)

    for i in range(n_wave):
        theta = np.random.rand()*2*pi
        k_i = np.array([Kx*np.cos(theta),Ky*np.sin(theta)])

        phase = np.random.rand()
        phi_i = np.exp(1j*(k_i[0]*xx + k_i[1]*yy + phase)*2*np.pi)

        phi = phi + phi_i

    phi = phi/n_wave
    phi_real = np.real(phi)
    
    markers = np.tanh((phi_real-alpha)*100)
    # markers[:, 0] = -1
    # markers[:,-1] = -1
    # markers[ 0,:] = -1
    # markers[-1,:] = -1

    img = markers
    imsize = img.shape
    contours_list = find_contours(img, 0)
    
    img2mesh(img, filename, meshsize_min = 0.025, meshsize_max = 0.05)
    
    return mesh