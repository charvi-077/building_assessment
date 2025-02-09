import open3d as o3d
import numpy as np

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("/home/charvi/Documents/ZED/Meshes/felicity_ground/mesh.ply")
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("/home/charvi/Documents/ZED/Meshes/felicity_ground/felicity_pointcloud.pcd", pcd)