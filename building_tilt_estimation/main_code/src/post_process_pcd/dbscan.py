# import open3d as o3d
# import numpy as np

# # Replace 'your_point_cloud.pcd' with the path to your PCD file
# pcd_path = 'tilted_demo_building.pcd'

# pcd = o3d.io.read_point_cloud(pcd_path)

# # Compute normals
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# # Segment the point cloud based on normals
# plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
# colors = np.asarray(pcd.colors)
# colors[inliers] = [1, 0, 0]  # Color of the plane segment (e.g., red)
# colors[~np.array(inliers)] = [0, 0, 1]  # Color of the non-plane segment (e.g., blue)
# pcd.colors = o3d.utility.Vector3dVector(colors)

# # Visualize the colored point cloud
# o3d.visualization.draw_geometries([pcd])

import open3d as o3d
import numpy as np
import array

# Replace 'your_point_cloud.pcd' with the path to your PCD file
pcd_path = 'tilted_demo_building.pcd'

# Read the point cloud from the specified file
pcd = o3d.io.read_point_cloud(pcd_path)

# Compute normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd = o3d.geometry.orient_normals_to_align_with_direction(pcd, orientation_reference=array([0., 0., 1.]))

o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# Segment the point cloud based on normals
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

# Create separate point clouds for the plane and non-plane segments
pcd_plane = pcd.select_by_index(np.where(inliers)[0])
pcd_non_plane = pcd.select_by_index(np.where(~np.array(inliers))[0])

# Color the points based on segmentation (red for plane, blue for non-plane)
pcd_plane.paint_uniform_color([1, 0, 0])  # Color of the plane segment (e.g., red)
pcd_non_plane.paint_uniform_color([0, 0, 1])  # Color of the non-plane segment (e.g., blue)

# Visualize the colored point clouds
o3d.visualization.draw_geometries([pcd_plane])