# Import Statements
import os
import topogenesis as tg
import trimesh as tm
import numpy as np
import pandas as pd
import rhino3dm as rg
import random

#Display full array without truncation
np.set_printoptions(threshold=np.inf)

class meshvoxelation ():
        
        def __init__(self,basemesh,vx, vy, vz):
            # Define the x,y,z size of the voxels
            self.basemesh = basemesh
            self.vx = vx
            self.vy = vy
            self.vz = vz


        def rhino3dmtotglattice(self):
            unit = [self.vx, self.vy, self.vz]
            tol = 1e-09
            mesh = tg.geometry.load_rhino3dmmesh(self.basemesh)
            sample_cloud, ray_origins = tg.geometry.mesh_sampling(mesh, unit, multi_core_process=False, return_ray_origin = True, tol=tol)
            tg_lattice = sample_cloud.voxelate(unit, closed=True)    
            return tg_lattice

        def tglattice_breps (self,lattice):
            voxel_points = lattice.centroids
            voxels = []
            for pt in voxel_points:
                # Define the corner points of the box (unit cube centered at pt)
                corner1 = rg.Point3d(pt[0] - self.vx / 2, pt[1] - self.vy / 2, pt[2] - self.vz / 2)
                corner2 = rg.Point3d(pt[0] + self.vx / 2, pt[1] + self.vy / 2, pt[2] + self.vz / 2)
                
                # Create a BoundingBox using the two corner points
                bounding_box = rg.BoundingBox(corner1, corner2)
                
                # Create the box using the BoundingBox
                box = rg.Box(bounding_box)
                brep_box = rg.Brep.CreateFromBox(box)
                # Append the mesh to the list
                voxels.append(brep_box)
            
            return voxels
        
        def tglattice_colouredmesh(self, lattice,colour_List):
            voxel_points = lattice.centroids
            voxel_meshes = []  # Store voxel meshes
            for number, pt in enumerate(voxel_points):
                # Create an empty mesh
                mesh = rg.Mesh()

                # Define 8 vertices for the voxel (cube)
                vertices = [
                    (pt[0] - self.vx / 2, pt[1] - self.vy / 2, pt[2] - self.vz / 2),  # 0
                    (pt[0] + self.vx / 2, pt[1] - self.vy / 2, pt[2] - self.vz / 2),  # 1
                    (pt[0] + self.vx / 2, pt[1] + self.vy / 2, pt[2] - self.vz / 2),  # 2
                    (pt[0] - self.vx / 2, pt[1] + self.vy / 2, pt[2] - self.vz / 2),  # 3
                    (pt[0] - self.vx / 2, pt[1] - self.vy / 2, pt[2] + self.vz / 2),  # 4
                    (pt[0] + self.vx / 2, pt[1] - self.vy / 2, pt[2] + self.vz / 2),  # 5
                    (pt[0] + self.vx / 2, pt[1] + self.vy / 2, pt[2] + self.vz / 2),  # 6
                    (pt[0] - self.vx / 2, pt[1] + self.vy / 2, pt[2] + self.vz / 2)   # 7
                ]

                # Add vertices to mesh
                for v in vertices:
                    mesh.Vertices.Add(v[0], v[1] ,v[2])

                # Define faces for the cube (each face is a quad split into 2 triangles)
                faces = [
                    (0, 1, 2, 3),  # Bottom
                    (4, 5, 6, 7),  # Top
                    (0, 1, 5, 4),  # Side
                    (1, 2, 6, 5),
                    (2, 3, 7, 6),
                    (3, 0, 4, 7)   # Side
                ]

                # Add faces to mesh
                for f in faces:
                    mesh.Faces.AddFace(*f)

                # Assign random colors to each voxel (optional)

                for i in range(len(mesh.Vertices)):
                    color = colour_List[number]
                    mesh.VertexColors.Add( red=color[0],green=color[1],blue=color[2])

                # Add voxel mesh to list
                voxel_meshes.append(mesh)

            return voxel_meshes
        
        def createbrepvoxels (self):
            tg_lattice = self.rhino3dmtotglattice()
            voxel_grid = self.tglattice_breps(tg_lattice)
            return(voxel_grid)
        
        def createcoloredmeshvoxels (self,colour_List):
            tg_lattice = self.rhino3dmtotglattice()
            voxel_grid = self.tglattice_colouredmesh(tg_lattice,colour_List)
            return(voxel_grid)
        