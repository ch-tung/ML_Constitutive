"""
Module for project part II of MEC568/2021

This module contains the basic geometry for the project task and a set of useful tools for
mesh manipulation, mesh creation and data processing.

Usage
=====


Meshing
-------

The module has a propert `geo` which is pygmsh geometry object. Right from the start, 
this geometry object is empty and can be populated with several `add_*` functions
provided by this module. The function `substract(...)` allows to introduce holes.
Translation and rotation is supported throught `translate()...` and `rotate(...)`.

The geometric domain can be transformed into a FEniCS mesh via `create_fenics_mesh`. In addition,
there is `write_mesh` for explicitly create a mesh and write it to a file. Furthermore,
`check_porosity` checks whether the current geometry meets the project requirements and provides
correspoding information.

Each of the routines mentioned above has its own brief documentation that can be accessed via 
`routine_name?` from a notebook session.
"""


import numpy as np
import pathlib
import os
import typing
import pygmsh
import meshio
import dolfin


# General definitions. Not to be modified by students.
# ====================================================


# Meshing
min_feature_size = 1 / 20000 # minimal feature size, ie. hole diameter...
feature_mesh_ratio = 30 # ratio "characterisitc feature size / mesh size"
hmeshmin = min_feature_size / feature_mesh_ratio # minimal mesh size
hmeshmax = 0.1  # maximal mesh size


# Type aliases
Surface = pygmsh.opencascade.surface_base.SurfaceBase
Geometry = pygmsh.opencascade.Geometry


# Geometry object (stateful!)
geo = None


# "User" routines
# ===============


def reset_geometry() -> None:
    """
    Reset the module-global geometry object to an empty pygmsh.opencascade.Geometry object.
    The other user internally routines operate on this object. This routine is interally called
    by `precracked_domain`.
    """
    
    # Setup the geometry kernel. I recommend to stay with the opencascade thing...
    global geo
    geo = pygmsh.opencascade.Geometry(
        # these parameters set some bounds on cell/element sizes
        characteristic_length_min=hmeshmin,
        characteristic_length_max=hmeshmax
    )


def subtract(domain: Surface, tools: typing.Iterable[Surface]) -> Surface:
    """
    Subtract the geometric features in `tools` from domain.
    
    :param domain: the domain from which the tools should be subtracted
    :param tools: list of domains (surface objects) that should be removed from domain
    :return: the modified domain
    """
    try:
        tools = list(tools)
    except TypeError:
        tools = [tools]
    return geo.boolean_difference([domain], tools)


def intersect(domain: Surface, tools: Surface) -> Surface:
    """
    Subtract the geometric features in `tools` from domain.
    
    :param domain: the domain from which the tools should be intersected
    :param tools: list of domains (surface objects) that should be removed from domain
    :return: the modified domain
    """
    return geo.boolean_intersection([domain, tools])



def rotate(obj: Surface, base_point: typing.Iterable, angle: float):
    """
    Rotating an object around base_point by a given angle
    
    :param obj: the object to be rotated
    :param base_point: base point of rotation
    """
    if len(base_point) == 2:
        base_point = [base_point[ii] for ii in (0, 1)] + [0]
    axis = [0, 0, 1]
    geo.rotate(obj, base_point, angle, axis)
    
    
def translate(obj: Surface, disp: typing.Iterable):
    """
    Tranlate an object by the given displacement vector
    
    :param obj: the object to be rotated
    :param disl: displacement vector
    """
    if len(disp) == 2:
        disp = [disp[ii] for ii in (0, 1)] + [0]
    geo.translate(obj, disp)
    

def add_ellipse(center: typing.Iterable, a: float,  b: float, 
                angle: float=0, mesh_size: float=None) -> Surface:
    """
    Create an ellipse. This implemetation accepts values b > a.
    
    :param center: array-like; center of the ellipse
    :param a: horizontal (x_1) axis length
    :param b: vertical (x_2) axis length
    :param angle: rotation angle (around center)
    :param mesh_size: size of the mesh
    :return: ellipse created
    """
    min_length = min_feature_size/2
    tol = 1e-3 * min_feature_size
    
    if a < b:
        _tmp = a
        a = b
        b = _tmp
        angle += np.pi/2

    if b < min_length - tol:
        raise ValueError("Minimal half-axis length is {!s} (detected: {!s})"
                         .format(min_length, b))
        
    mesh_size = b / feature_mesh_ratio if mesh_size is None else mesh_size

    if len(center) == 2:
        center = [center[ii] for ii in (0, 1)] + [0]
        
    ellipse = geo.add_disk(
        center,
        radius0=a,
        radius1=b,
        char_length=mesh_size
    )
    
    if abs(angle) > 1e-3:
        rotate(ellipse, center, angle)
        
    return ellipse


def add_rectangle(center: typing.Iterable, width: float, height:float, 
                  angle: float=0, corner_radius: float=0, mesh_size: float=None) -> Surface:
    """
    Create a rectangle.
    
    :param bottom_left_corner: array-like; center of the ellipse
    :param width: width of the (unrotated) rectangle
    :param height: height of the (unrotated) rectangle
    :param angle: rotation angle (around center)
    :param corner_radius: radius of corners; defaults to sharp corners
    :param mesh_size: size of the mesh
    :return: rectangle created
    """
    min_length = min_feature_size
    tol = 1e-3 * min_feature_size
    
    if min(width, height) < min_length - tol:
        raise ValueError("Minimal side length is {!s} (detected: {!s})".format(min_length, min(width, height)))
        
    mesh_size = min(width, height) / feature_mesh_ratio if mesh_size is None else mesh_size

    if len(center) == 2:
        center = np.array([center[ii] for ii in (0, 1)] + [0])
        
    bottom_left_corner = center.copy()
    bottom_left_corner[0] -= width/2
    bottom_left_corner[1] -= height/2
    rectangle = geo.add_rectangle(
        bottom_left_corner, width, height,
        corner_radius=corner_radius,
        char_length=mesh_size
    )
    
    if abs(angle) > 1e-3:
        rotate(rectangle, center, angle)
    
    return rectangle


def add_polygon(vertices: typing.List[typing.Iterable], mesh_size: float=None) -> Surface:
    """
    Create a polygon.
    
    :param vertices: list of array-like; vertices of the polygon
    :param mesh_size: size of the mesh
    :return: polygon created
    """
    min_dist = min_feature_size
    tol = 1e-3 * min_feature_size
    
    min_detected = 1
    vertices = list(vertices)
    
    for vi, vertex in enumerate(vertices):
        if len(vertex) == 2:
            vertex = [vertex[ii] for ii in (0, 1)] + [0]
        vertices[vi] = np.array(vertex)
        if vi > 0:
            for other in vertices[:vi]:
                dist = np.linalg.norm(vertex - other)
                min_detected = min(min_detected, dist)
                if dist < min_dist - tol:
                    raise ValueError("Minimal point distance is {!s} (detected: {!s})"
                                     .format(min_dist, min_detected))
        
    mesh_size = min_detected / feature_mesh_ratio if mesh_size is None else mesh_size
        
    polygon = geo.add_polygon(
        vertices,
        lcar=mesh_size
    )
    
    return polygon


def add_crack(vertices: typing.List[typing.Iterable], mesh_size) -> Surface:
    """
    Create a crack. This is the same as add_polygon but without the feature size limit.
    Also, the mesh size has to be given explicitly.
    
    :param vertices: list of array-like; vertices of the polygon
    :param mesh_size: size of the mesh
    :return: polygon created
    """
    for vi, vertex in enumerate(vertices):
        if len(vertex) == 2:
            vertex = [vertex[ii] for ii in (0, 1)] + [0]
        vertices[vi] = np.array(vertex)
        
    polygon = geo.add_polygon(
        vertices,
        lcar=mesh_size
    )
    return polygon


def write_mesh(mesh_path: pathlib.Path=pathlib.Path("mesh.xdmf"), verbose=True) -> None:
    """
    Writes a FEniCS-compatible mesh file. Given path must have ".xdmf" suffix!
    
    :param mesh_path: path/name of the mesh to be created
    :param verbose: whether gmsh output should be shown
    """
    if mesh_path.suffix != ".xdmf":
        raise ValueError("Given path must have \".xdmf\" suffix!")
        
    mdir = mesh_path.parent
    mname = mesh_path.name
    pygmsh_mesh = pygmsh.generate_mesh(geo, verbose=verbose)
    pre_path = mdir / "_pre_{!s}.vtu".format(mname)
    if pre_path.exists():
        os.remove(pre_path)
    if mesh_path.exists():
        os.remove(mesh_path)
    meshio.write(str(pre_path), pygmsh_mesh)
#     meshio.write(str(mesh_path), pygmsh_mesh)
    os.system("meshio-convert -pz {!s} {!s}".format(pre_path, mesh_path))
    
##    os.system("meshio-convert --remove_lower_dimensional_cells --remove_orphaned_nodes {!s} {!s}".format(pre_path, mesh_path))



def create_fenics_mesh(mesh_path: pathlib.Path=pathlib.Path("mesh.xdmf"), verbose=True) -> dolfin.Mesh:
    """
    Directly create a FEniCS mesh. Along the way a FEniCS-compatible mesh file is written.
    
    :param mesh_path: path/name of the mesh to be created
    :param verbose: whether gmsh output should be shown
    :return: the FEniCS mesh
    """
    write_mesh(mesh_path=mesh_path, verbose=verbose)
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile(str(mesh_path)) as xf:
        xf.read(mesh)
    return mesh


def create_XDMFFile(fname: pathlib.Path) -> dolfin.XDMFFile:
    """
    Create XDMF File for storing FEniCS data (mesh/results) readable by Paraview.
    This function is a wrapper around dolfin.XDMFFile constructor that sets some
    properties to more reasonable values.
    
    :param fname: path/name of the file to be create
    :return: the created file
    """
    
    xf = dolfin.XDMFFile(str(fname))
    # Functions are defined over same mesh. this is important for Paraview
    xf.parameters["functions_share_mesh"] = True 
    
    # write immediatly; for reading before simulation is done
    xf.parameters["flush_output"] = True 
    
    # only write the mesh once
    xf.parameters["rewrite_function_mesh"] = False
    return xf


# "reset" (initialize) the geometry object such that the `add_*` routines can be used from the start
reset_geometry()
