import numpy as np
import open3d as o3d
from numpy import linalg
from scipy.spatial import KDTree
import datetime


def icp_core(points1, points2):
    """
    solve transformation from points1 to points2, points of the same index are well matched
    :param points1: numpy array of points1, size = nx3, n is num of point
    :param points2: numpy array of points2, size = nx3, n is num of point
    :return: transformation matrix T, size = 4x4
    """

    """
    Args:
        centroid1 : the center point of points1， size = 1x3   
        centroid2 : the center point of points2， size = 1x3
        N         : num of point
        points1C  : points1 minus the center point of points1 (centroid1), size = nx3, n is num of point
        points2C  : points2 minus the center point of points2 (centroid2), size = nx3, n is num of point
        H         : the covariance matrix between points1C and points2C, size = 3x3
        U         : rotation matrix decomposed from H by svd method, size = 3x3
        sigma     : eigen matrix of H matrix decomposed by svd method,  size = 3x3
        VT        : rotation matrix decomposed from H by svd method, size = 3x3
        R         : rotation matrix of point 1 to point 2, size = 3x3
        t         : translation matrix of point 1 to point 2, size = 3x1

    Returns:
        T         : transformation matrix T, size = 4x4

    Raises:
        IOError: points1.shape != points2.shape, 'point could size not match'
    """
    assert points1.shape == points2.shape, 'point could size not match'

    T = np.zeros(shape=(4, 4))
    T[0:3, 0:3] = np.eye(3)
    T[3, 3] = 1

    # Todo: step1: calculate centroid
    centroid1 = np.average(points1, axis=0)
    centroid2 = np.average(points2, axis=0)

    # Todo: step2: de-centroid of points1 and points2
    N = points1.shape[0]
    points1C = points1 - np.tile(centroid1, (N, 1))
    points2C = points2 - np.tile(centroid2, (N, 1))

    # Todo: step3: compute H, which is sum of p1i'*p2i'^T
    H = np.matmul(points1C.T, points2C)

    # Todo: step4: SVD of H (can use 3rd-part lib), solve R and t
    U, sigma, VT = linalg.svd(H)
    R = np.matmul(VT.T, U.T)
    t = -np.matmul(R, centroid1) + centroid2

    # Todo: step5, combine R and t into transformation matrix T
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


def svd_based_icp_unmatched(points1, points2, n_iter, threshold):
    # Copy the two input point clouds to 'points_1' and 'points_2'
    # initialize the accumulated transformation matrix 'T_accumulated' to a 4x4 identity matrix
    points_1 = points1.copy()  # Copy point cloud 1
    points_2 = points2.copy()  # Copy point cloud 2
    T_accumulated = np.eye(4)  # Initialize the accumulated transformation matrix as a 4x4 identity matrix

    # Create an o3d visualization window and visualize the second point cloud as blue
    # Create a coordinate frame triangle mesh
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    pcd2 = o3d.geometry.PointCloud()  # Create an o3d point cloud object for the second point cloud
    # Assign the coordinate data of the second point cloud to the o3d point cloud object
    pcd2.points = o3d.utility.Vector3dVector(points_2)
    pcd2.paint_uniform_color([0, 0, 1])  # Set the color of the second point cloud to blue
    vis = o3d.visualization.Visualizer()  # Create an o3d visualizer
    vis.create_window()  # Create an o3d visualization window
    vis.add_geometry(axis_pcd)  # Add the coordinate frame triangle mesh to the visualizer
    vis.add_geometry(pcd2)  # Add the second point cloud to the visualizer

    start_time = datetime.datetime.now()

    for i in range(n_iter):

        # Todo: for all point in points_1, find nearest point in points_2, and generate points_2_nearest
        kdtree = KDTree(points_2)  # Create a KDTree object from the second point cloud
        # Query the KDTree to find the nearest points in points_2 for each point in points_1 ，
        # The second output 'indices' contains the indices of the nearest neighbors in points_2
        _, indices = kdtree.query(points_1)
        points_2_nearest = points_2[indices]  # Use the indices to generate a new point cloud

        # solve icp
        T = icp_core(points_1, points_2_nearest)

        # Todo: update accumulated T
        # Update the accumulated transformation matrix by multiplying it with the current transformation matrix 'T'
        T_accumulated = np.matmul(T, T_accumulated)

        print('-----------------------------------------')
        print('iteration = ' + str(i + 1))
        print('T = ')
        print(T)
        print('accumulated T = ')
        print(T_accumulated)

        # Todo: update points_1
        # Update the position of the first point cloud using the accumulated transformation matrix 'T_accumulated'
        points_1 = np.matmul(points1, T_accumulated[0:3, 0:3].T) + T_accumulated[0:3, 3]
        # Calculate the mean distance between the updated 'points_1' and the second point cloud 'points_2'
        mean_distance = mean_dist(points_1, points_2)
        print('mean_error= ' + str(mean_distance))

        # visualization
        pcd1_transed = o3d.geometry.PointCloud()
        pcd1_transed.points = o3d.utility.Vector3dVector(points_1)
        pcd1_transed.paint_uniform_color([1, 0, 0])
        vis.add_geometry(pcd1_transed)
        vis.poll_events()
        vis.update_renderer()
        vis.remove_geometry(pcd1_transed)

        if mean_distance < 0.00001 or mean_distance < threshold:
            print('fully converged!')
            break
    # Record the end time and calculate the time difference
    end_time = datetime.datetime.now()
    time_difference = (end_time - start_time).total_seconds()
    print('time cost: ' + str(time_difference) + ' s')
    vis.destroy_window()
    o3d.visualization.draw_geometries([axis_pcd, pcd2, pcd1_transed])


def mean_dist(points1, points2):
    # Initialize an empty list to store the distances between corresponding points in the two point clouds
    dis_array = []
    # Loop over all points in the two point clouds and calculate the Euclidean distance between corresponding points
    for i in range(points1.shape[0]):
        dif = points1[i] - points2[i]
        dis = np.linalg.norm(dif)
        dis_array.append(dis)
    return np.mean(np.array(dis_array))


def main():
    pcd1 = o3d.io.read_point_cloud(r'bunny1.ply')
    pcd2 = o3d.io.read_point_cloud(r'bunny2.ply')
    points1 = np.array(pcd1.points)
    points2 = np.array(pcd2.points)

    # task 2:
    svd_based_icp_unmatched(points1, points2, n_iter=100, threshold=0.3)


if __name__ == '__main__':
    main()
