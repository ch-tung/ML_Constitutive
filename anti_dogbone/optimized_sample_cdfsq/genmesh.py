"""
Change logs

20220618 CHTUNG
Updated network models

20220625 CHTUNG
Updated Bezier models
"""

# from fenics import *
# from dolfin_adjoint import *
import pygmsh_mesh_functions
from pygmsh_mesh_functions import *
import pathlib
# import meshio
import numpy as np
# import matplotlib.pyplot as plt

import skimage.io
from skimage.measure import find_contours, subdivide_polygon, approximate_polygon
from skimage.filters import gaussian

#### Harmonic loop holes ####
def hole_polygon(domain, polygon):
    if np.array_equal(polygon[-1],polygon[0]):
        polygon = polygon[:-1]
    tool = pygmsh_mesh_functions.add_polygon(polygon)
    domain = pygmsh_mesh_functions.subtract(domain, tool)

def arrange_array(n_holes_x, n_holes_y, d_x, d_y,
                  freq = 1, phase = 0, twist = 0):
    """
    Returns the center and orientation of tools on a rectangular array
    
    n_holes_x, n_holes_y: number of holes in x/y direction
    
    d_x, d_y: distance between holes in x/y direction
    
    freq, phase: determining the orientation of holes
        theta = np.pi*(sgn/2)*freq + phase
        
    twist: the magnitude of random misorientation (in rad)
    """
    
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



# array of 2-fold symmetry loops --------------------------------------------------------------------
def genmesh_loop_array_2fold(LISTPARAM=[1/8, 0.75, 0.25], Lx = 1.0, Ly = 1.0,
                             twist = 0,
                             n_phi=60, approximate = 0.005, 
                             filename = "test_loop_array_2fold.xdmf",Verbose=True):
    """
    Returns the mesh with array of 2-fold symmetry holes
    The hole shape is determined by:
    
        r(theta) = maxR*[1 + sum(A_i*cos(2n*pi*theta))/sum(abs(A_i))]/2 for n = 0 to N, 
        
    and maxR is in the unit of min(Lx,Ly). A_0 = 1.
    
    LISTPARAM: [maxR, A_1, ... A_N]
    
    Lx, Ly: sample size
    
    twist: the magnitude of random misorientation (in rad)
    
    n_phi: amount of points used to generate the loop
    
    approximate: coefficient for approximating the polygon with Douglas-Peucker algorithm
    https://scikit-image.org/docs/stable/auto_examples/edges/plot_polygon.html
    approximate = 0 returns the original polygon
    
    filename: output filename for the generated mesh
    """
    
    # arrange the tools
    size = 4
    c, theta = arrange_array(5, 3, 1/size, 1/size, freq = 1/2, phase = np.pi/4, twist = twist)

    # define tools
    phi = np.arange(n_phi+1)/n_phi*2*np.pi
    scale = min(Lx,Ly)*LISTPARAM[0]

    r = np.ones((n_phi+1)) 
    v = np.zeros((2,n_phi+1))
    for i in range(len(LISTPARAM)-1):
        ri = np.cos(2*(i+1)*phi)*LISTPARAM[i+1]
        r = r+ri
        
#     r_square_mean = np.mean(r**2)
#     r_mean = np.mean(r)
#     r = r/np.sqrt(r_square_mean)*r_mean
    
    r = (1+r.copy()/np.sum(np.abs(LISTPARAM[1:])))/2
    r = r/np.max(r)
    
    v = np.array([np.cos(phi), np.sin(phi)])*r
    v = v*scale
    
    meshsize_min = 0.07*min(Lx,Ly)#0.015*min(Lx,Ly)
    meshsize_max = 0.2*min(Lx,Ly)
#     print(meshsize_min)
    
    pygmsh_mesh_functions.reset_geometry_parameters(meshsize_min,meshsize_max)

    domain = add_polygon([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    X0 = np.array([0, 0])
    
    #OPERATIONS ON DOMAIN
    for i in range(len(theta)):
        co, so = np.cos(theta[i]), np.sin(theta[i])
        Rotation = np.array(((co, -so), (so, co)))

        polygon = (Rotation@v).T
        polygon = polygon[:-1]

        polygon_subdivide = subdivide_polygon(polygon, degree=2, preserve_ends=True)
        polygon_approx = approximate_polygon(polygon_subdivide,approximate*min(Lx,Ly))
#         polygon_approx = approximate_polygon(polygon,approximate*min(Lx,Ly))
        polygon_shift = polygon_approx + (c*min(Lx,Ly))[:,i]

        hole_polygon(domain, polygon_shift)

    #CREATE AND STORE MESH
    sample_name = filename
    out_dir = pathlib.Path("output_files")
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = create_fenics_mesh(mesh_path = out_dir / sample_name, verbose=Verbose)
    cell_type = mesh.cell_name()
    with create_XDMFFile(out_dir / sample_name) as xf:
        xf.write(mesh)

#     plot(mesh)
    len(mesh.coordinates())
    
    return mesh

# array of 3-fold symmetry loops --------------------------------------------------------------------
def genmesh_loop_array_3fold(LISTPARAM = np.array([1/np.sqrt(3)/4, 0.75, 0.25]), Lx = 1.0, Ly = 1.0,
                             twist = 0, margin = 0.2,
                             n_phi=60, approximate = 0.005, 
                             filename = "test_loop_array_3fold.xdmf",Verbose=True):
    """
    Returns the mesh with array of 3-fold symmetry holes
    The hole shape is determined by:
    
        r(theta) = maxR*[1 + sum(A_i*cos(3n*pi*theta))/sum(abs(A_i))]/2 for n = 0 to N, 
        
    and maxR is in the unit of min(Lx,Ly). A_0 = 1.
    
    LISTPARAM: [maxR, A_1, ... A_N]
    
    Lx, Ly: sample size
    
    twist: the magnitude of random misorientation (in rad)
    
    n_phi: amount of points used to generate the loop
    
    approximate: coefficient for approximating the polygon with Douglas-Peucker algorithm
    https://scikit-image.org/docs/stable/auto_examples/edges/plot_polygon.html
    approximate = 0 returns the original polygon
    
    filename: output filename for the generated mesh
    """
    
    # arrange the tools
    size = np.sqrt(3)*2
    c1, theta1 = arrange_array(3, 2, np.sqrt(3)/size, 1/size, freq = 0, phase = -np.pi/2, twist = twist)
    c2, theta2 = arrange_array(2, 3, np.sqrt(3)/size, 1/size, freq = 0, phase = -np.pi/2, twist = twist)

    # define tools
    phi = np.arange(n_phi+1)/n_phi*2*np.pi
    scale = min(Lx,Ly)*LISTPARAM[0]

    r = np.ones((n_phi+1))
    v = np.zeros((2,n_phi+1))
    for i in range(len(LISTPARAM)-1):
        ri = np.cos(3*(i+1)*phi)*LISTPARAM[i+1]
        r = r+ri
        
#     r_square_mean = np.mean(r**2)
#     r_mean = np.mean(r)
#     r = r/np.sqrt(r_square_mean)*r_mean
    
    r = (1+r.copy()/np.sum(np.abs(LISTPARAM[1:])))/2
    r = r/np.max(r)
    
    v = np.array([np.cos(phi), np.sin(phi)])*r
    v = v*scale

    meshsize_min = 0.07*min(Lx,Ly)#0.015*min(Lx,Ly)
    meshsize_max = 0.2*min(Lx,Ly)
#     print(meshsize_min)
    
    pygmsh_mesh_functions.reset_geometry_parameters(meshsize_min,meshsize_max)

    domain = add_polygon([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    X0 = np.array([0, 0])

    #OPERATIONS ON DOMAIN
    for i in range(len(theta1)):
        co, so = np.cos(theta1[i]), np.sin(theta1[i])
        Rotation = np.array(((co, -so), (so, co)))

        polygon = (Rotation@v).T
        polygon = polygon[:-1]

#         polygon_subdivide = subdivide_polygon(polygon, degree=2, preserve_ends=True)
#         polygon_approx = approximate_polygon(polygon_subdivide,approximate*min(Lx,Ly))
        polygon_approx = approximate_polygon(polygon,approximate*min(Lx,Ly))
        polygon_shift = polygon_approx + (c1*min(Lx,Ly))[:,i]

        hole_polygon(domain, polygon_shift)

    for i in range(len(theta2)):
        co, so = np.cos(theta1[i]), np.sin(theta1[i])
        Rotation = np.array(((co, -so), (so, co)))

        polygon = (Rotation@v).T
        polygon = polygon[:-1]

        polygon_subdivide = subdivide_polygon(polygon, degree=2, preserve_ends=True)
        polygon_approx = approximate_polygon(polygon_subdivide,approximate*min(Lx,Ly))
#         polygon_approx = approximate_polygon(polygon,approximate*min(Lx,Ly))
        polygon_shift = polygon_approx + (c2*min(Lx,Ly))[:,i]

        hole_polygon(domain, polygon_shift)

    #CREATE AND STORE MESH
    sample_name = filename
    out_dir = pathlib.Path("output_files")
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = create_fenics_mesh(mesh_path = out_dir / sample_name, verbose=Verbose)
    cell_type = mesh.cell_name()
    with create_XDMFFile(out_dir / sample_name) as xf:
        xf.write(mesh)

#     plot(mesh)
    len(mesh.coordinates())
    
    return mesh

##################### HARMONIC HOLES

def cosdecomphole(LISTPARAM,nbpoints=50,w=0):
    """
    returns the list of coordinates of the hole contour
    
    the w parameter enables to dilatate the polygon by a width W
    
    formula for the contour (LaTeX): 
    
    R(\theta_j)=\frac{R_{max}}{2}\left(\frac{
    \sum_{k=1}^{N_h}c_k\cos(k(\theta_j+2\pi\phi_k))
    }{
    \max_{i\in[\![1,n]\!]}|\sum_{k=1}^{N_h}c_k\cos(k(\theta_i+2\pi\phi_k))|
    }
    +1\right)
    
    
    Parameters:
        LISTPARAM=[x,y,maxR,coef1,rot1,coef2,rot2,coef3,rot3,coef4...]
        rotations are rescaled from [0,2*np.pi] to [0,1]
    """
    x=LISTPARAM[0]
    y=LISTPARAM[1]
    maxR=LISTPARAM[2]
    coeffs=LISTPARAM[3:]
    
    PHI=np.linspace(0,2*np.pi,nbpoints)
    
    RR=np.zeros(nbpoints)
    
    print(len(coeffs))
    for i in range(int(len(coeffs)/2)):
        c,rot=coeffs[2*i],2*np.pi*coeffs[2*i+1]
        RR+=c*np.cos(i*(PHI+rot))
    RR/=max(abs(RR))
    RR=w+maxR*(RR+1)/2
    
    X=[RR[i]*np.cos(PHI[i])+x for i in range(nbpoints-1)]
    Y=[RR[i]*np.sin(PHI[i])+y for i in range(nbpoints-1)]
#     plt.plot(X,Y)
#     plt.show()
    
    return list(zip(X,Y))

def HHgeneralized_meshwithholes(LISTHOLES,Lx=1e-2,Ly=1e-2,nbpoints=50,filename="test_complex-harmonic-holes.xdmf",Verbose=True):
    """
    return the holed mesh
    with LISTHOLES = [nb_holes,x1,y1,maxR1,coef11,rot11,...,x2,y2,maxR2,...,x3,...,x(nb_holes),y,maxR,coef1,rot1,...]
    """
    #BASE DOMAIN___________________________________________________________________________________________________________
#     pygmsh_mesh_functions.reset_geometry()

    meshsize_min = 0.07*min(Lx,Ly)#0.015*min(Lx,Ly)
    meshsize_max = 0.2*min(Lx,Ly)
    print(meshsize_min)
    
    pygmsh_mesh_functions.reset_geometry_parameters(meshsize_min,meshsize_max)

    domain = pygmsh_mesh_functions.add_polygon([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    X0 = np.array([0, 0])

    #OPERATIONS ON DOMAIN__________________________________________________________________________________________________
    nb_holes=int(LISTHOLES[0])
    lenhole=(len(LISTHOLES)-1)/nb_holes
    for i in range(nb_holes):
        holeparam_i = LISTHOLES[int(1+i*lenhole):int(1+(i+1)*lenhole)]
        hole=pygmsh_mesh_functions.add_polygon(cosdecomphole(holeparam_i,nbpoints=nbpoints))
        domain = pygmsh_mesh_functions.subtract(domain,hole)
    
    #CREATE AND STORE MESH
    sample_name = filename
    out_dir = pathlib.Path("output_files")
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = create_fenics_mesh(mesh_path = out_dir / sample_name, verbose=Verbose)
    cell_type = mesh.cell_name()
    with create_XDMFFile(out_dir / sample_name) as xf:
        xf.write(mesh)

#     plot(mesh)
    len(mesh.coordinates())
    
    return mesh

#### Free harmonic holes with wall to avoid very small material strips
def freeHH_withholewall(LISTPARAM,Lx=1e-2,Ly=1e-2,nbpoints=50,filename="test_complex-harmonic-holes.xdmf",Verbose=True):
    """
    return the holed mesh
    with LISTHOLES = [W,nb_holes,x1,y1,maxR1,coef11,rot11,...,x2,y2,maxR2,...,x3,...,x(nb_holes),y,maxR,coef1,rot1,...]
    W is the minimal strip width
    """
    #BASE DOMAIN___________________________________________________________________________________________________________
#     pygmsh_mesh_functions.reset_geometry()

    meshsize_min = 0.07*min(Lx,Ly)#0.015*min(Lx,Ly)
    meshsize_max = 0.2*min(Lx,Ly)
    print(meshsize_min)
    
    pygmsh_mesh_functions.reset_geometry_parameters(meshsize_min,meshsize_max)

    domain = pygmsh_mesh_functions.add_polygon([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    X0 = np.array([0, 0])

    #OPERATIONS ON DOMAIN__________________________________________________________________________________________________
    
    W=LISTPARAM[0]
    LISTHOLES=LISTPARAM[1:]
    nb_holes=int(LISTHOLES[0])
    lenhole=(len(LISTHOLES)-1)/nb_holes
    
    for i in range(nb_holes):
        holeparam_i = LISTHOLES[int(1+i*lenhole):int(1+(i+1)*lenhole)]
        
        outsidepolygon=pygmsh_mesh_functions.add_polygon(cosdecomphole(holeparam_i,nbpoints=nbpoints,w=W))
        domain = pygmsh_mesh_functions.union(domain,outsidepolygon)
        
        hole=pygmsh_mesh_functions.add_polygon(cosdecomphole(holeparam_i,nbpoints=nbpoints))
        domain = pygmsh_mesh_functions.subtract(domain,hole)
    
    frame = pygmsh_mesh_functions.add_polygon([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    domain = pygmsh_mesh_functions.intersect(domain,frame)
    
    #CREATE AND STORE MESH
    sample_name = filename
    out_dir = pathlib.Path("output_files")
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = create_fenics_mesh(mesh_path = out_dir / sample_name, verbose=Verbose)
    cell_type = mesh.cell_name()
    with create_XDMFFile(out_dir / sample_name) as xf:
        xf.write(mesh)

#     plot(mesh)
    len(mesh.coordinates())
    
    return mesh


#### Harmonic holes on a grid

def HolesWithPhaseOnGrid(LISTHOLES,Lx=1e-2,Ly=1e-2,nbpoints=50,filename="test_grid-harmonic-holes.xdmf",Verbose=True):
    """
    return the holed mesh
    with LISTHOLES = [Nx,Ny,Rmax,coef11,rot11,...,coef21,rot21,...,coef(Nx*Ny)1,(Nx*Ny)rot1,...,coef(Nx*Ny)(Nharmonics),(Nx*Ny)(Nharmonics)]
    """
    #Extract parameters
    Nx=int(LISTHOLES[0])
    Ny=int(LISTHOLES[1])
    Rmax=float(LISTHOLES[2])
    LenHole=int((len(LISTHOLES)-3)/(Nx*Ny))
#     print("LenHole : {}".format(LenHole))
    
    #Centers
    
    c, theta = arrange_array(Nx, Ny, 1/(Nx+1), 1/(Ny+1), freq = 1/2, phase = np.pi/4, twist = 0)
    
    #BASE DOMAIN___________________________________________________________________________________________________________
#     pygmsh_mesh_functions.reset_geometry()

    meshsize_min = 0.07*min(Lx,Ly)#0.015*min(Lx,Ly)
    meshsize_max = 0.2*min(Lx,Ly)
    print(meshsize_min)
    
    pygmsh_mesh_functions.reset_geometry_parameters(meshsize_min,meshsize_max)

    domain = pygmsh_mesh_functions.add_polygon([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    X0 = np.array([0, 0])

    #OPERATIONS ON DOMAIN__________________________________________________________________________________________________
    for i in range(Nx*Ny):
#         print("First list :",[c[0][i],c[1][i],Rmax],"type :{}".format(type([c[0][i],c[1][i],Rmax])))
#         print("First list :",LISTHOLES[int(3+i*LenHole):int(3+(i+1)*LenHole)],"type :{}".format(type(LISTHOLES[int(3+i*LenHole):int(3+(i+1)*LenHole)])))
        holeparam_i = [Lx*c[0][i],Ly*c[1][i],Rmax]+list(LISTHOLES[int(3+i*LenHole):int(3+(i+1)*LenHole)])
        polygon=np.array(cosdecomphole(holeparam_i,nbpoints=nbpoints))
        polygon_subdivide = subdivide_polygon(polygon, degree=2, preserve_ends=True)
        polygon_approx = approximate_polygon(polygon_subdivide,0.005*min(Lx,Ly))
        hole=pygmsh_mesh_functions.add_polygon(polygon_approx)
        domain = pygmsh_mesh_functions.subtract(domain,hole)
    
    #CREATE AND STORE MESH
    sample_name = filename
    out_dir = pathlib.Path("output_files")
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = create_fenics_mesh(mesh_path = out_dir / sample_name, verbose=Verbose)
    cell_type = mesh.cell_name()
    with create_XDMFFile(out_dir / sample_name) as xf:
        xf.write(mesh)

#     plot(mesh)
    len(mesh.coordinates())
    
    return mesh

#### Harmonic holes on a grid with WALLS

def gridHH_withholewall(LISTPARAM,Lx=1e-2,Ly=1e-2,nbpoints=50,filename="test_grid-harmonic-holes.xdmf",Verbose=True):
    """
    return the holed mesh
    with LISTPARAM = [W,Nx,Ny,Rmax,coef11,rot11,...,coef21,rot21,...,coef(Nx*Ny)1,(Nx*Ny)rot1,...,coef(Nx*Ny)(Nharmonics),(Nx*Ny)(Nharmonics)]
    where W is the "wall width"
    """
    W=LISTPARAM[0]
    LISTHOLES=LISTPARAM[1:]
    #Extract parameters
    Nx=int(LISTHOLES[0])
    Ny=int(LISTHOLES[1])
    Rmax=float(LISTHOLES[2])
    LenHole=int((len(LISTHOLES)-3)/(Nx*Ny))
#     print("LenHole : {}".format(LenHole))
    
    #Centers
    
    c, theta = arrange_array(Nx, Ny, 1/(Nx+1), 1/(Ny+1), freq = 1/2, phase = np.pi/4, twist = 0)
    
    #BASE DOMAIN___________________________________________________________________________________________________________
#     pygmsh_mesh_functions.reset_geometry()

    meshsize_min = 0.07*min(Lx,Ly)#0.015*min(Lx,Ly)
    meshsize_max = 0.2*min(Lx,Ly)
    print(meshsize_min)
    
    pygmsh_mesh_functions.reset_geometry_parameters(meshsize_min,meshsize_max)

    domain = pygmsh_mesh_functions.add_polygon([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    X0 = np.array([0, 0])

    #OPERATIONS ON DOMAIN__________________________________________________________________________________________________
    for i in range(Nx*Ny):
        
        holeparam_i = [Lx*c[0][i],Ly*c[1][i],Rmax]+list(LISTHOLES[int(3+i*LenHole):int(3+(i+1)*LenHole)])
        
        polygon=np.array(cosdecomphole(holeparam_i,nbpoints=nbpoints,w=W))
        polygon_subdivide = subdivide_polygon(polygon, degree=2, preserve_ends=True)
        polygon_approx = approximate_polygon(polygon_subdivide,0.005*min(Lx,Ly))
        holeandwall=pygmsh_mesh_functions.add_polygon(polygon_approx)
        
        domain = pygmsh_mesh_functions.union(domain,holeandwall)
        
        polygon=np.array(cosdecomphole(holeparam_i,nbpoints=nbpoints))
        polygon_subdivide = subdivide_polygon(polygon, degree=2, preserve_ends=True)
        polygon_approx = approximate_polygon(polygon_subdivide,0.005*min(Lx,Ly))
        hole=pygmsh_mesh_functions.add_polygon(polygon_approx)
        
        domain = pygmsh_mesh_functions.subtract(domain,hole)
    
    frame = pygmsh_mesh_functions.add_polygon([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    domain = pygmsh_mesh_functions.intersect(domain,frame)
    
    #CREATE AND STORE MESH
    sample_name = filename
    out_dir = pathlib.Path("output_files")
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = create_fenics_mesh(mesh_path = out_dir / sample_name, verbose=Verbose)
    cell_type = mesh.cell_name()
    with create_XDMFFile(out_dir / sample_name) as xf:
        xf.write(mesh)

#     plot(mesh)
    len(mesh.coordinates())
    
    return mesh


#### Network model --------------------------------------------------------------------
def arch(parameters,size=4,height=0.25,n_arch = 64):
    x_arch = (np.arange(n_arch+1)/n_arch-0.5)
    y_arch = np.zeros_like(x_arch)
    
    if np.sum(np.abs(parameters))>0:
        for i in range(len(parameters)):
            y_arch = y_arch + np.cos((x_arch)*np.pi*(2*i+1))*parameters[i]
        y_arch = y_arch/np.max(np.abs(y_arch))/size*height

    x_arch = x_arch/size
    v_arch = np.array([x_arch,y_arch])
    
    return v_arch

def genmesh_network_2fold(LISTPARAM, Lx = 1.0, Ly = 1.0,
                             twist = 0, height = 0.25,
                             n_arch=64, approximate = 0.5, 
                             filename = "test_network_2fold.xdmf",Verbose=True):
    """
    Returns the mesh with p4gm symmetrical pattern by replacing the unit cell boundary with curves 
    generated by Fourier series.
    
    LISTPARAM[0]: the edge width
    LISTPARAM[1]: max curve height
    LISTPARAM[2:]: coefficients of fourier series 
                   sum(A_i*cos((2*i+1)*pi*theta))/sum(abs(A_i))]/2 for n = 0 to N-1
    
    """
    
    # arrange the tools
    size = 4
    c_h, theta_h = arrange_array(5, 4, 1/size, 1/size, freq = 1, phase = np.pi/2)
    c_v, theta_v = arrange_array(4, 3, 1/size, 1/size, freq = 1, phase = 0)
    
    v_arch = arch(LISTPARAM[2:],size,LISTPARAM[1],n_arch=64)
    d_fuzz = LISTPARAM[0]

    n_x = 200
    n_y = 200

    x = np.arange(n_x)/n_x
    y = np.arange(n_y)/n_y

    xx, yy = np.meshgrid(x,y)
    phi = np.zeros_like(xx)

    for i in range(len(theta_h)):
        co, so = np.cos(theta_h[i]), np.sin(theta_h[i])
        Rotation = np.array(((co, -so), (so, co)))
        vR = (Rotation@v_arch).T+c_h[:,i]

        for j in range(len(vR)):
            dx = xx-vR[j,0];
            dy = yy-vR[j,1];
    #         phi = phi + np.exp(-(dx**2+dy**2)/2/d_fuzz**2)
            phi = phi + (np.sign(d_fuzz-np.sqrt(dx**2+dy**2))+1)/2

    for i in range(len(theta_v)):
        co, so = np.cos(theta_v[i]), np.sin(theta_v[i])
        Rotation = np.array(((co, -so), (so, co)))
        vR = (Rotation@v_arch).T+c_v[:,i]

        for j in range(len(vR)):
            dx = xx-vR[j,0];
            dy = yy-vR[j,1];
    #         phi = phi + np.exp(-(dx**2+dy**2)/2/d_fuzz**2)
            phi = phi + (np.sign(d_fuzz-np.sqrt(dx**2+dy**2))+1)/2
            
    imsize = phi.shape
    phi_gaussian = gaussian(phi,sigma=1)
    contours_list = find_contours(phi_gaussian, 0.5)
    polygon_list = []
    for i in range(len(contours_list)):
        polygon = np.flip(contours_list[i],axis=1)

        # exclude the invalid polygons
        if not np.array_equal(polygon[-1],polygon[0]):
            d_ends = polygon[-1]-polygon[0]
            # bottom
            cond_btm_y = (polygon[-1][1]==0) & (polygon[0][1]==0)
            cond_btm_x =  np.abs((polygon[-1][0]+polygon[0][0])/2-n_x/2) > 2
            if cond_btm_y & cond_btm_x:
                continue
            # top
            cond_top_y = (polygon[-1][1]==imsize[0]-1) & (polygon[0][1]==imsize[0]-1)
            cond_top_x =  np.abs((polygon[-1][0]+polygon[0][0])/2-n_x/2) > 2
            if cond_top_y & cond_top_x:
                continue
            # left and right
            if (polygon[-1][0]==0) & (polygon[0][0]==imsize[0]-1):
                continue
            if (polygon[0][0]==0) & (polygon[-1][0]==imsize[0]-1):
                continue
            # corners
            if d_ends[0]*d_ends[1] != 0:
                continue
                
        polygon_size_x = max(polygon[:,0])-min(polygon[:,0])
        polygon_size_y = max(polygon[:,1])-min(polygon[:,1])
        polygon_size = max(polygon_size_x,polygon_size_x)
        if polygon_size<4:
            continue
            
        if len(polygon)<3:
            continue

        # simplify the polygons
        polygon_subdivide = subdivide_polygon(polygon, degree=2, preserve_ends=True)
        polygon_approx = approximate_polygon(polygon_subdivide,approximate)

        polygon_list.append(polygon_approx)
        
    meshsize_min = 0.07*min(Lx,Ly)#0.015*min(Lx,Ly)
    meshsize_max = 0.2*min(Lx,Ly)
#     print(meshsize_min)

    pygmsh_mesh_functions.reset_geometry_parameters(meshsize_min,meshsize_max)

    domain = add_polygon([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    X0 = np.array([0, 0])

    #OPERATIONS ON DOMAIN
    for i in range(len(polygon_list)):
        hole_polygon(domain, polygon_list[i]/(imsize[0]-1)*min(Lx,Ly))


    #CREATE AND STORE MESH
    sample_name = filename
    out_dir = pathlib.Path("output_files")
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = create_fenics_mesh(mesh_path = out_dir / sample_name, verbose=Verbose)
    cell_type = mesh.cell_name()
    with create_XDMFFile(out_dir / sample_name) as xf:
        xf.write(mesh)

#     plot(mesh)
    len(mesh.coordinates())
    
    return mesh
    
def genmesh_network_3fold(LISTPARAM, Lx = 1.0, Ly = 1.0,
                             twist = 0, height = 0.25,
                             n_arch=64, approximate = 0.5, 
                             filename = "test_network_3fold.xdmf",Verbose=True):
    """
    Returns the mesh with p31m symmetrical pattern by replacing the unit cell boundary with curves 
    generated by Fourier series.
    
    LISTPARAM[0]: the edge width
    LISTPARAM[1]: max curve height
    LISTPARAM[2:]: coefficients of fourier series 
                   sum(A_i*cos((2*i+1)*pi*theta))/sum(abs(A_i))]/2 for n = 0 to N-1
    
    """
    
    # arrange the tools
    size = np.sqrt(3)*2
    c_h1, theta_h1 = arrange_array(3, 3, np.sqrt(3)/size, 1/size, freq = 0, phase = np.pi)
    c_h2, theta_h2 = arrange_array(2, 4, np.sqrt(3)/size, 1/size, freq = 0, phase = np.pi)
    c_v, theta_v = arrange_array(4, 6, np.sqrt(3)/size/2, 1/size/2, freq = 4/3, phase = np.pi)
    
    v_arch = arch(LISTPARAM[2:],size,LISTPARAM[1],n_arch=64)
    v_arch = v_arch/np.sqrt(3)
    d_fuzz = LISTPARAM[0]
    
    n_x = 200
    n_y = 200

    x = np.arange(n_x)/n_x
    y = np.arange(n_y)/n_y

    xx, yy = np.meshgrid(x,y)
    phi = np.zeros_like(xx)
    
    for i in range(len(theta_h1)):
        co, so = np.cos(theta_h1[i]), np.sin(theta_h1[i])
        Rotation = np.array(((co, -so), (so, co)))
        vR = (Rotation@v_arch).T+c_h1[:,i]

        for j in range(len(vR)):
            dx = xx-vR[j,0];
            dy = yy-vR[j,1];
    #         phi = phi + np.exp(-(dx**2+dy**2)/2/d_fuzz**2)
            phi = phi + (np.sign(d_fuzz-np.sqrt(dx**2+dy**2))+1)/2
        
    for i in range(len(theta_h2)):
        co, so = np.cos(theta_h2[i]), np.sin(theta_h2[i])
        Rotation = np.array(((co, -so), (so, co)))
        vR = (Rotation@v_arch).T+c_h2[:,i]

        for j in range(len(vR)):
            dx = xx-vR[j,0];
            dy = yy-vR[j,1];
    #         phi = phi + np.exp(-(dx**2+dy**2)/2/d_fuzz**2)
            phi = phi + (np.sign(d_fuzz-np.sqrt(dx**2+dy**2))+1)/2

    for i in range(len(theta_v)):
        co, so = np.cos(theta_v[i]), np.sin(theta_v[i])
        Rotation = np.array(((co, -so), (so, co)))
        vR = (Rotation@v_arch).T+c_v[:,i]

        for j in range(len(vR)):
            dx = xx-vR[j,0];
            dy = yy-vR[j,1];
    #         phi = phi + np.exp(-(dx**2+dy**2)/2/d_fuzz**2)
            phi = phi + (np.sign(d_fuzz-np.sqrt(dx**2+dy**2))+1)/2
        
    imsize = phi.shape
    phi_gaussian = gaussian(phi,sigma=1)
    contours_list = find_contours(phi_gaussian, 0.5)
    polygon_list = []
    for i in range(len(contours_list)):
        polygon = np.flip(contours_list[i],axis=1)

        # exclude the invalid polygons
        if not np.array_equal(polygon[-1],polygon[0]):
            d_ends = polygon[-1]-polygon[0]
            # bottom
            cond_btm_y = (polygon[-1][1]==0) & (polygon[0][1]==0)
            cond_btm_x =  np.abs((polygon[-1][0]+polygon[0][0])/2-n_x/2) < 2
            if cond_btm_y & cond_btm_x:
                continue
            # top
            cond_top_y = (polygon[-1][1]==imsize[0]-1) & (polygon[0][1]==imsize[0]-1)
            if cond_top_y:
                continue
            # left and right
            if (polygon[-1][0]==0) & (polygon[0][0]==imsize[0]-1):
                continue
            if (polygon[0][0]==0) & (polygon[-1][0]==imsize[0]-1):
                continue
            # corners
            if d_ends[0]*d_ends[1] != 0:
                continue
                
        polygon_size_x = max(polygon[:,0])-min(polygon[:,0])
        polygon_size_y = max(polygon[:,1])-min(polygon[:,1])
        polygon_size = max(polygon_size_x,polygon_size_x)
        if polygon_size<4:
            continue
            
        if len(polygon)<3:
            continue

        # simplify the polygons
        polygon_subdivide = subdivide_polygon(polygon, degree=2, preserve_ends=True)
        polygon_approx = approximate_polygon(polygon_subdivide,approximate)

        polygon_list.append(polygon_approx)
        
    meshsize_min = 0.07*min(Lx,Ly)#0.015*min(Lx,Ly)
    meshsize_max = 0.2*min(Lx,Ly)
#     print(meshsize_min)

    pygmsh_mesh_functions.reset_geometry_parameters(meshsize_min,meshsize_max)

    domain = add_polygon([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    X0 = np.array([0, 0])

    #OPERATIONS ON DOMAIN
    for i in range(len(polygon_list)):
        hole_polygon(domain, polygon_list[i]/(imsize[0]-1)*min(Lx,Ly))

    #CREATE AND STORE MESH
    sample_name = filename
    out_dir = pathlib.Path("output_files")
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = create_fenics_mesh(mesh_path = out_dir / sample_name, verbose=Verbose)
    cell_type = mesh.cell_name()
    with create_XDMFFile(out_dir / sample_name) as xf:
        xf.write(mesh)

#     plot(mesh)
    len(mesh.coordinates())
    
    return mesh

#### Bezier ####---------------------------------------------------------------------------------------------
def arch_nBezier_3fold(parameters, size=6, n_arch = 64):
    t = np.arange(n_arch+1)/n_arch
    
    N = len(parameters)
    n = 2*N+1
    # position of control points
    Ci = []
    if n>3: 
        delta_x = np.pi/3/(n-3)
        for i in range(N-1):
            Cix = delta_x*(i+1)
            Ciy = parameters[i+1]*np.pi/6
            t1 = np.tan(Cix)
            t2 = np.tan(Ciy)
            Cx = (3*t1*t2+np.sqrt(3)*t1)/(2*t1*t2+2)
            Cy = -(np.sqrt(3)*t1-3)*t2/(2*t1*t2+2)
            Ci.append([Cx,Cy])
        
    Ci_rev = [[-x[0],x[1]] for x in reversed(Ci)]
    
    C = np.array([[-0.5,0]]+Ci_rev+[[0,parameters[0]*np.sqrt(3)/2]]+Ci+[[0.5,0]])
        
    v = np.zeros((2,len(t)))
    for j in range(n):
        basis = (t**j*(1-t)**(n-1-j)*
                np.math.factorial(n-1)/np.math.factorial(n-1-j)/np.math.factorial(j))
   
        v = v + np.array([x*basis for x in C[j]])
    
    V = v/size
    
    return V

def arch_nBezier_2fold(parameters, size=4, n_arch = 64):
    t = np.arange(n_arch+1)/n_arch
    
    N = len(parameters)
    n = 2*N+1
    # position of control points
    Ci = []
    if n>3: 
        delta_x = 1/(n-3)
        for i in range(N-1):
            Cix = delta_x*(i+1)
            Ciy = parameters[i+1]/2
            Ci.append([Cix,Ciy])
        
    Ci_rev = [[-x[0],x[1]] for x in reversed(Ci)]
    
    C = np.array([[-0.5,0]]+Ci_rev+[[0,parameters[0]/2]]+Ci+[[0.5,0]])
        
    v = np.zeros((2,len(t)))
    for j in range(n):
        basis = (t**j*(1-t)**(n-1-j)*
                np.math.factorial(n-1)/np.math.factorial(n-1-j)/np.math.factorial(j))
   
        v = v + np.array([x*basis for x in C[j]])
    
    V = v/size
    
    return V

def genmesh_Bezier_3fold(LISTPARAM, Lx = 1.0, Ly = 1.0,
                             twist = 0, height = 0.25,
                             n_arch=64, approximate = 0.5, 
                             filename = "test_network_3fold.xdmf",Verbose=True):
    """
    Returns the mesh with p31m symmetrical pattern by replacing the unit cell boundary with Bezier curves.
    The Bezier curves are determined by 2*N+1 control points where N = len(LISTPARAM)-1.
    
    LISTPARAM[0]: the edge width
    LISTPARAM[1:]: defining the position of Bezier curve control points in two-center bipolar coordinates
    """
    
    # arrange the tools
    size = np.sqrt(3)*2
    c_h1, theta_h1 = arrange_array(3, 3, np.sqrt(3)/size, 1/size, freq = 0, phase = np.pi)
    c_h2, theta_h2 = arrange_array(2, 4, np.sqrt(3)/size, 1/size, freq = 0, phase = np.pi)
    c_v, theta_v = arrange_array(4, 6, np.sqrt(3)/size/2, 1/size/2, freq = 4/3, phase = np.pi)
    
    v_arch = arch_nBezier_3fold(LISTPARAM[1:],n_arch=64)
    v_arch = v_arch
    d_fuzz = LISTPARAM[0]
    
    n_x = 200
    n_y = 200

    x = np.arange(n_x)/n_x
    y = np.arange(n_y)/n_y

    xx, yy = np.meshgrid(x,y)
    phi = np.zeros_like(xx)
    
    for i in range(len(theta_h1)):
        co, so = np.cos(theta_h1[i]), np.sin(theta_h1[i])
        Rotation = np.array(((co, -so), (so, co)))
        vR = (Rotation@v_arch).T+c_h1[:,i]

        for j in range(len(vR)):
            dx = xx-vR[j,0];
            dy = yy-vR[j,1];
    #         phi = phi + np.exp(-(dx**2+dy**2)/2/d_fuzz**2)
            phi = phi + (np.sign(d_fuzz-np.sqrt(dx**2+dy**2))+1)/2
        
    for i in range(len(theta_h2)):
        co, so = np.cos(theta_h2[i]), np.sin(theta_h2[i])
        Rotation = np.array(((co, -so), (so, co)))
        vR = (Rotation@v_arch).T+c_h2[:,i]

        for j in range(len(vR)):
            dx = xx-vR[j,0];
            dy = yy-vR[j,1];
    #         phi = phi + np.exp(-(dx**2+dy**2)/2/d_fuzz**2)
            phi = phi + (np.sign(d_fuzz-np.sqrt(dx**2+dy**2))+1)/2

    for i in range(len(theta_v)):
        co, so = np.cos(theta_v[i]), np.sin(theta_v[i])
        Rotation = np.array(((co, -so), (so, co)))
        vR = (Rotation@v_arch).T+c_v[:,i]

        for j in range(len(vR)):
            dx = xx-vR[j,0];
            dy = yy-vR[j,1];
    #         phi = phi + np.exp(-(dx**2+dy**2)/2/d_fuzz**2)
            phi = phi + (np.sign(d_fuzz-np.sqrt(dx**2+dy**2))+1)/2
        
    imsize = phi.shape
    phi_gaussian = gaussian(phi,sigma=1)
    contours_list = find_contours(phi_gaussian, 0.5)
    polygon_list = []
    for i in range(len(contours_list)):
        polygon = np.flip(contours_list[i],axis=1)

        # exclude the invalid polygons
        if not np.array_equal(polygon[-1],polygon[0]):
            d_ends = polygon[-1]-polygon[0]
            # bottom
            cond_btm_y = (polygon[-1][1]==0) & (polygon[0][1]==0)
            cond_btm_x =  np.abs((polygon[-1][0]+polygon[0][0])/2-n_x/2) < 2
            if cond_btm_y & cond_btm_x:
                continue
            # top
            cond_top_y = (polygon[-1][1]==imsize[0]-1) & (polygon[0][1]==imsize[0]-1)
            if cond_top_y:
                continue
            # left and right
            if (polygon[-1][0]==0) & (polygon[0][0]==imsize[0]-1):
                continue
            if (polygon[0][0]==0) & (polygon[-1][0]==imsize[0]-1):
                continue
            # corners
            if d_ends[0]*d_ends[1] != 0:
                continue
                
        polygon_size_x = max(polygon[:,0])-min(polygon[:,0])
        polygon_size_y = max(polygon[:,1])-min(polygon[:,1])
        polygon_size = max(polygon_size_x,polygon_size_x)
        if polygon_size<4:
            continue
            
        if len(polygon)<3:
            continue

        # simplify the polygons
        polygon_subdivide = subdivide_polygon(polygon, degree=2, preserve_ends=True)
        polygon_approx = approximate_polygon(polygon_subdivide,approximate)

        polygon_list.append(polygon_approx)
        
    meshsize_min = 0.07*min(Lx,Ly)#0.015*min(Lx,Ly)
    meshsize_max = 0.2*min(Lx,Ly)
#     print(meshsize_min)

    pygmsh_mesh_functions.reset_geometry_parameters(meshsize_min,meshsize_max)

    domain = add_polygon([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    X0 = np.array([0, 0])

    #OPERATIONS ON DOMAIN
    for i in range(len(polygon_list)):
        hole_polygon(domain, polygon_list[i]/(imsize[0]-1)*min(Lx,Ly))

    #CREATE AND STORE MESH
    sample_name = filename
    out_dir = pathlib.Path("output_files")
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = create_fenics_mesh(mesh_path = out_dir / sample_name, verbose=Verbose)
    cell_type = mesh.cell_name()
    with create_XDMFFile(out_dir / sample_name) as xf:
        xf.write(mesh)

#     plot(mesh)
    len(mesh.coordinates())
    
    return mesh
    
def genmesh_Bezier_2fold(LISTPARAM, Lx = 1.0, Ly = 1.0,
                             twist = 0, height = 0.25,
                             n_arch=64, approximate = 0.5, 
                             filename = "test_network_2fold.xdmf",Verbose=True):
    """
    Returns the mesh with p4gm symmetrical pattern by replacing the unit cell boundary with Bezier curves.
    The Bezier curves are determined by 2*N+1 control points where N = len(LISTPARAM)-1.
    
    LISTPARAM[0]: the edge width
    LISTPARAM[1:]: defining the position of Bezier curve control points
    """
    
    # arrange the tools
    size = 4
    c_h, theta_h = arrange_array(5, 4, 1/size, 1/size, freq = 1, phase = np.pi/2)
    c_v, theta_v = arrange_array(4, 3, 1/size, 1/size, freq = 1, phase = 0)
    
    v_arch = arch_nBezier_2fold(LISTPARAM[1:],n_arch=64)
    d_fuzz = LISTPARAM[0]

    n_x = 200
    n_y = 200

    x = np.arange(n_x)/n_x
    y = np.arange(n_y)/n_y

    xx, yy = np.meshgrid(x,y)
    phi = np.zeros_like(xx)

    for i in range(len(theta_h)):
        co, so = np.cos(theta_h[i]), np.sin(theta_h[i])
        Rotation = np.array(((co, -so), (so, co)))
        vR = (Rotation@v_arch).T+c_h[:,i]

        for j in range(len(vR)):
            dx = xx-vR[j,0];
            dy = yy-vR[j,1];
    #         phi = phi + np.exp(-(dx**2+dy**2)/2/d_fuzz**2)
            phi = phi + (np.sign(d_fuzz-np.sqrt(dx**2+dy**2))+1)/2

    for i in range(len(theta_v)):
        co, so = np.cos(theta_v[i]), np.sin(theta_v[i])
        Rotation = np.array(((co, -so), (so, co)))
        vR = (Rotation@v_arch).T+c_v[:,i]

        for j in range(len(vR)):
            dx = xx-vR[j,0];
            dy = yy-vR[j,1];
    #         phi = phi + np.exp(-(dx**2+dy**2)/2/d_fuzz**2)
            phi = phi + (np.sign(d_fuzz-np.sqrt(dx**2+dy**2))+1)/2
            
    imsize = phi.shape
    phi_gaussian = gaussian(phi,sigma=1)
    contours_list = find_contours(phi_gaussian, 0.5)
    polygon_list = []
    for i in range(len(contours_list)):
        polygon = np.flip(contours_list[i],axis=1)

        # exclude the invalid polygons
        if not np.array_equal(polygon[-1],polygon[0]):
            d_ends = polygon[-1]-polygon[0]
            # bottom
            cond_btm_y = (polygon[-1][1]==0) & (polygon[0][1]==0)
            cond_btm_x =  np.abs((polygon[-1][0]+polygon[0][0])/2-n_x/2) > 2
            if cond_btm_y & cond_btm_x:
                continue
            # top
            cond_top_y = (polygon[-1][1]==imsize[0]-1) & (polygon[0][1]==imsize[0]-1)
            cond_top_x =  np.abs((polygon[-1][0]+polygon[0][0])/2-n_x/2) > 2
            if cond_top_y & cond_top_x:
                continue
            # left and right
            if (polygon[-1][0]==0) & (polygon[0][0]==imsize[0]-1):
                continue
            if (polygon[0][0]==0) & (polygon[-1][0]==imsize[0]-1):
                continue
            # corners
            if d_ends[0]*d_ends[1] != 0:
                continue
                
        polygon_size_x = max(polygon[:,0])-min(polygon[:,0])
        polygon_size_y = max(polygon[:,1])-min(polygon[:,1])
        polygon_size = max(polygon_size_x,polygon_size_x)
        if polygon_size<4:
            continue
            
        if len(polygon)<3:
            continue

        # simplify the polygons
        polygon_subdivide = subdivide_polygon(polygon, degree=2, preserve_ends=True)
        polygon_approx = approximate_polygon(polygon_subdivide,approximate)

        polygon_list.append(polygon_approx)
        
    meshsize_min = 0.07*min(Lx,Ly)#0.015*min(Lx,Ly)
    meshsize_max = 0.2*min(Lx,Ly)
#     print(meshsize_min)

    pygmsh_mesh_functions.reset_geometry_parameters(meshsize_min,meshsize_max)

    domain = add_polygon([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    X0 = np.array([0, 0])

    #OPERATIONS ON DOMAIN
    for i in range(len(polygon_list)):
        hole_polygon(domain, polygon_list[i]/(imsize[0]-1)*min(Lx,Ly))


    #CREATE AND STORE MESH
    sample_name = filename
    out_dir = pathlib.Path("output_files")
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = create_fenics_mesh(mesh_path = out_dir / sample_name, verbose=Verbose)
    cell_type = mesh.cell_name()
    with create_XDMFFile(out_dir / sample_name) as xf:
        xf.write(mesh)

#     plot(mesh)
    len(mesh.coordinates())
    
    return mesh

# def arch_Bezier_3fold(parameters, size=6, n_arch = 64):
#     t = np.arange(n_arch+1)/n_arch
    
#     # position of control points
#     C2 = [0,parameters[0]*np.sqrt(3)/2]
#     t1 = np.tan(parameters[1]*np.pi/6)
#     t2 = np.tan(parameters[2]*np.pi/6)
#     Cx = (3*t1*t2+np.sqrt(3)*t1)/(2*t1*t2+2)
#     Cy = -(np.sqrt(3)*t1-3)*t2/(2*t1*t2+2)
#     C1 = [-Cx,Cy]
#     C3 = [Cx,Cy]
#     C = np.array([[-0.5,0],C1,C2,C3,[0.5,0]])
    
#     v = []
#     for i in range(len(t)):
#         vi = ((1-t[i])**4*C[0] + 
#              4*(1-t[i])**3*t[i]*C[1] + 
#              6*(1-t[i])**2*t[i]**2*C[2] + 
#              4*(1-t[i])*t[i]**3*C[3] + 
#              t[i]**4*C[4]
#             )
#         v.append(vi)
#     V = np.array(v).T/size
    
#     return V

# def arch_Bezier_2fold(parameters, size=4, n_arch = 64):
#     t = np.arange(n_arch+1)/n_arch
    
#     # position of control points
#     C2 = [0,parameters[0]/2]
#     Cx = parameters[1]/2
#     Cy = parameters[2]/2
#     C1 = [-Cx,Cy]
#     C3 = [Cx,Cy]
#     C = np.array([[-0.5,0],C1,C2,C3,[0.5,0]])
    
#     v = []
#     for i in range(len(t)):
#         vi = ((1-t[i])**4*C[0] + 
#              4*(1-t[i])**3*t[i]*C[1] + 
#              6*(1-t[i])**2*t[i]**2*C[2] + 
#              4*(1-t[i])*t[i]**3*C[3] + 
#              t[i]**4*C[4]
#             )
#         v.append(vi)
#     V = np.array(v).T/size
    
#     return V