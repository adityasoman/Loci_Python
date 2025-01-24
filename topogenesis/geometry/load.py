

"""
Geometry loading modules of topoGenesis
"""

import numpy as np
import pyvista as pv
import warnings
import os
import rhino3dm as rg

__author__ = "Shervin Azadi, and Pirouz Nourian"
__copyright__ = "???"
__credits__ = ["Shervin Azadi", "Pirouz Nourian"]
__license__ = "???"
__version__ = "0.0.2"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"

file_directory = os.path.dirname(os.path.abspath(__file__))


def load_mesh(mesh_path, check_is_triangle=True):
    # load the mesh using pyvista
    pv_mesh = pv.read(mesh_path)

    # check if all of the faces are triangles
    if check_is_triangle and not pv_mesh.is_all_triangles():
        raise ValueError('All faces need to be triangles!')

    # extract faces and vertices
    v = np.array(pv_mesh.points)
    f = pv_mesh.faces.reshape(-1, 4)[:, 1:]

    # return them as 3d numpy arrays
    return np.array(v).astype(np.float64), np.array(f).astype(np.int64)

def load_rhino3dmmesh(mesh):
        # load the mesh using pyvista
        rhino_mesh = mesh
        ##rhino_mesh.Faces.ConvertTrianglesToQuads(0.1,0.5)
        # Extract vertices
        vertices = np.array([[v.X, v.Y, v.Z] for v in rhino_mesh.Vertices])
        # Extract faces
        faces = np.array(rhino_mesh.Faces)

        # Create a PyVista mesh
        pyvista_mesh = pv.PolyData(vertices, faces)
        # extract faces and vertices
        v = vertices
        f = faces.reshape(-1, 4)[:, 1:]

        # return them as 3d numpy arrays
        return np.array(v).astype(np.float64), np.array(f).astype(np.int64)
    
