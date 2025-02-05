

"""
Geometry loading modules of topoGenesis
"""

import numpy as np
import pyvista as pv
import warnings
import os
import trimesh as tm

__author__ = "Shervin Azadi, and Pirouz Nourian"
__copyright__ = "???"
__credits__ = ["Shervin Azadi", "Pirouz Nourian"]
__license__ = "???"
__version__ = "0.0.2"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"

file_directory = os.path.dirname(os.path.abspath(__file__))


def load_mesh(mesh_path):
    # load the mesh using pyvista
    pv_mesh = pv.read(mesh_path)

    # extract faces and vertices
    v = np.array(pv_mesh.points)
    f = pv_mesh.faces.reshape(-1, 4)[:, 1:]

    # return them as 3d numpy arrays
    return np.array(v).astype(np.float64), np.array(f).astype(np.int64)


def load_rhino3dmmesh(rhino_mesh):

    # Extract vertices and faces
    vertices = np.array([[rhino_mesh.Vertices[i].X, rhino_mesh.Vertices[i].Y, rhino_mesh.Vertices[i].Z] for i in range(len(rhino_mesh.Vertices))])
    faces = []
    # Extract faces correctly
    for i in range(rhino_mesh.Faces.Count):  # Loop through the number of faces
        face = rhino_mesh.Faces[i]  # Get face data (an array of vertex indices)
        faces.append([face[0], face[1], face[2]])

    np_faces = np.array(faces)
    # Create trimesh object
    Trimesh_mesh = tm.Trimesh(vertices=vertices, faces=np_faces)
    faces = np.pad(Trimesh_mesh.faces, ((0, 0),(1,0)), 'constant', constant_values=3)
    pv_mesh = pv.PolyData(Trimesh_mesh.vertices, faces)

    # extract faces and vertices
    v = np.array(pv_mesh.points)
    f = pv_mesh.faces.reshape(-1, 4)[:, 1:]

    # return them as 3d numpy arrays
    return np.array(v).astype(np.float64), np.array(f).astype(np.int64)