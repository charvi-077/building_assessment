import open3d as o3d
import numpy as np
# Replace 'your_point_cloud.pcd' with the path to your PCL file
pcd_path = 'roof_mode_data.ply'
pcd = o3d.io.read_point_cloud(pcd_path)

# Calculate the centroid of the point cloud
centroid = np.mean(np.asarray(pcd.points), axis=0)

# Create lines representing the coordinate axes with origin at the centroid
axis_lines = o3d.geometry.LineSet()
axis_lines.points = o3d.utility.Vector3dVector([centroid, centroid + [1, 0, 0], centroid, centroid + [0, 1, 0], centroid, centroid + [0, 0, 1]])
axis_lines.lines = o3d.utility.Vector2iVector([[0, 1], [0, 3], [0, 5]])
axis_lines.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Create a visualization window
o3d.visualization.draw_geometries([pcd, axis_lines])