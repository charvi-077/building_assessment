import open3d as o3d
import numpy as np

# Replace 'your_point_cloud.pcd' with the path to your PCD file
pcd_path = 'tilted_demo_building.pcd'

# Read the point cloud from the specified file
pcd = o3d.io.read_point_cloud(pcd_path)
# Compute normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))

# Initialize an empty array to store plane models
planes = []

# Iterate to find multiple planes
while True:
    # Segment the point cloud based on normals
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=50, num_iterations=1000)
    
    # Check if enough inliers are found
    if len(inliers) < 50:
        break  # Exit the loop if there are not enough inliers for a plane
    
    # Store the plane model
    planes.append(plane_model)

    # Remove the segmented plane from the point cloud
    pcd = pcd.select_by_index(np.where(~np.array(inliers))[0])

# Visualize the original point cloud with colored planes
for i, plane_model in enumerate(planes):
    # Create a new point cloud containing only the points belonging to the current plane
    pcd_plane = pcd.select_by_index(np.where(plane_model)[0])

    # Color the points based on the plane index
    color = np.random.rand(3) # Generate a random color for visualization
    pcd_plane.paint_uniform_color(color)

    # Visualize the colored point cloud for each plane
    o3d.visualization.draw_geometries([pcd_plane])

# Visualize the original point cloud without planes
o3d.visualization.draw_geometries([pcd])
