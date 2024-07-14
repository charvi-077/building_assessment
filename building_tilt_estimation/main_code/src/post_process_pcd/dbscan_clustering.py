import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt



# ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud("tilted_demo_building.pcd")
o3d.visualization.draw_geometries([pcd])

eps_values = [0.01, 0.005, 0.009, 0.007, 0.003]

for eps in eps_values:
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=10))
    num_clusters = len(np.unique(labels)) - 1  # Subtract 1 to exclude noise points
    print(f"For eps={eps}, number of clusters: {num_clusters}")

# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(
#         pcd.cluster_dbscan(eps=0.01, min_points=10, print_progress=True))

# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.455,
#                                   front=[-0.4999, -0.1659, -0.8499],
#                                   lookat=[2.1813, 2.0619, 2.0999],
#                                   up=[0.1204, -0.9852, 0.1215])
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=10, print_progress=True))

# Check the number of clusters
max_label = labels.max()
print(f"Point cloud has {max_label + 1} clusters")

# Assign colors to clusters
unique_labels = np.unique(labels)
num_clusters = len(unique_labels)
colors = plt.get_cmap("tab20")(unique_labels / (max_label if max_label > 0 else 1))

# Handle noise points separately
colors_with_noise = np.zeros((len(labels), 3))
for i, label in enumerate(unique_labels):
    if label >= 0:
        colors_with_noise[labels == label] = np.array(colors[i, :3])

# Create a new point cloud with colors
colored_pcd = o3d.geometry.PointCloud()
colored_pcd.points = pcd.points
colored_pcd.colors = o3d.utility.Vector3dVector(colors_with_noise)

# Visualize the colored point cloud
o3d.visualization.draw_geometries([colored_pcd]
                                )