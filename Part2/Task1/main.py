import numpy as np
import open3d as o3d
from numpy import linalg


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


def svd_based_icp_matched(points1, points2):
    T = icp_core(points1, points2)
    print('------------transformation matrix------------')
    print(T)

    # Todo: calculate transformed point cloud 1 based on T solved above
    # points1_transformed is the points1 that reaches points2 after transformation
    points1_transformed = np.matmul(points1, T[0:3, 0:3].T) + T[0:3, 3]

    # visualization
    mean_distance = mean_dist(points1_transformed, points2)
    print('mean_error= ' + str(mean_distance))
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd1_tran = o3d.geometry.PointCloud()
    pcd1_tran.points = o3d.utility.Vector3dVector(points1_transformed)
    pcd1.paint_uniform_color([1, 0, 0])
    pcd2.paint_uniform_color([0, 1, 0])
    pcd1_tran.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcd1, pcd2, pcd1_tran, axis_pcd])


def mean_dist(points1, points2):
    dis_array = []
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

    # task 1:
    svd_based_icp_matched(points1, points2)


if __name__ == '__main__':
    main()
