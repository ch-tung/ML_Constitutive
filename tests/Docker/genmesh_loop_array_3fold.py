from fenics import *
from dolfin_adjoint import *
import pygmsh_mesh_functions
from pygmsh_mesh_functions import *
import meshio
import numpy as np
import matplotlib.pyplot as plt

import skimage.io
from skimage.measure import find_contours, subdivide_polygon, approximate_polygon

def hole_polygon(domain, polygon):
    tool = add_polygon(polygon)
    domain = subtract(domain, tool)

def arrange_array(n_holes_x, n_holes_y, d_x, d_y,
                  freq = 1, phase = 0, shift = np.array([0,0]), twist = 0):
    
    min_n_holes = np.min((n_holes_x,n_holes_y))

#     xx = scale*(np.arange(n_holes_x)-n_holes_x/2+0.5)/size+0.5
#     yy = scale*(np.arange(n_holes_y)-n_holes_y/2+0.5)/size+0.5
    xx = (np.arange(n_holes_x)-n_holes_x/2+0.5)*d_x+0.5
    yy = (np.arange(n_holes_y)-n_holes_y/2+0.5)*d_y+0.5
    cx,cy = np.meshgrid(xx,yy)
    
    cx = cx.reshape(n_holes_x*n_holes_y)
    cy = cy.reshape(n_holes_x*n_holes_y)
    c = np.vstack((cx,cy)) # hole coordinate

    s_xx = (-1)**np.arange(n_holes_x)
    s_yy = (-1)**np.arange(n_holes_y)
    sgn_x,sgn_y = np.meshgrid(s_xx,s_yy)
    sgn = sgn_x*sgn_y
    sgn = sgn.reshape(n_holes_x*n_holes_y)
    theta = np.pi*(sgn/2)*freq + phase # hole orientation
    theta = theta + (np.random.rand(len(theta))-0.5)*twist
    
    return(c, theta)    

A = np.array([1.0,0.75,0.01])

def genmesh_loop_array_3fold(A = np.array([1.0,0.75,0.01]), twist = 0, margin = 0.2,
                     n_phi=60, approximate = 0.0025, 
                     filename = "test_loop_array_3fold.xdmf"):
    
    # arrange the tools
    size = np.sqrt(3)*2
    c1, theta1 = arrange_array(3, 2, np.sqrt(3)/size, 1/size, freq = 0, phase = -np.pi/2, twist = twist)
    c2, theta2 = arrange_array(2, 3, np.sqrt(3)/size, 1/size, freq = 0, phase = -np.pi/2, twist = twist)

    # define tools
    n_phi = 60
    phi = np.arange(n_phi+1)/n_phi*2*np.pi
    scale = 1/size/2*(1-margin)

    r = np.zeros((2,n_phi+1))
    for i in range(len(A)):
        ri = np.array([np.cos(phi), np.sin(phi)])*np.cos(3*(i)*phi)*A[i]
        r = r+ri
#     r = r/np.sum(A)
    r = r*scale

    Lx = 1
    Ly = 1

    meshsize_min = 0.015
    meshsize_max = 0.015

    pygmsh_mesh_functions.hmeshmin = meshsize_min
    pygmsh_mesh_functions.hmeshmax = meshsize_max

    reset_geometry()

    domain = add_polygon([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    X0 = np.array([0, 0])

    #OPERATIONS ON DOMAIN
    for i in range(len(theta1)):
        co, so = np.cos(theta1[i]), np.sin(theta1[i])
        Rotation = np.array(((co, -so), (so, co)))

        polygon = (Rotation@r).T
        polygon = polygon[:-1]

        polygon_subdivide = subdivide_polygon(polygon, degree=2, preserve_ends=True)
        polygon_approx = approximate_polygon(polygon_subdivide,approximate)
        polygon_shift = polygon_approx + c1[:,i]

        hole_polygon(domain, polygon_shift)

    for i in range(len(theta2)):
        co, so = np.cos(theta1[i]), np.sin(theta1[i])
        Rotation = np.array(((co, -so), (so, co)))

        polygon = (Rotation@r).T
        polygon = polygon[:-1]

        polygon_subdivide = subdivide_polygon(polygon, degree=2, preserve_ends=True)
        polygon_approx = approximate_polygon(polygon_subdivide,approximate)
        polygon_shift = polygon_approx + c2[:,i]

        hole_polygon(domain, polygon_shift)

    #CREATE AND STORE MESH
    sample_name = filename
    out_dir = pathlib.Path("output_files")
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = create_fenics_mesh(mesh_path = out_dir / sample_name, verbose=True)
    cell_type = mesh.cell_name()
    with create_XDMFFile(out_dir / sample_name) as xf:
        xf.write(mesh)

    plot(mesh)
    len(mesh.coordinates())