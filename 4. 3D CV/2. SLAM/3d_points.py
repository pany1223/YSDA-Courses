import numpy as np 
import cv2


def main():
    # an image point was measured on these two images 
    # (the bottom left corner of the monitor)
    image_0_name = "tests/00_test_slam_input/rgb_with_poses/000000.png" 
    image_6_name = "tests/00_test_slam_input/rgb_with_poses/000006.png"

    # image coordinates (measured manually) 
    image_0_2d_point = np.array([305.0, 292.0], ndmin=2) 
    image_6_2d_point = np.array([363.0, 195.0], ndmin=2)

    # image translation and quaternion from the file:
    # "tests/00_test_slam_gt/ground_truth.txt"
    image_0_translation = np.array([1.3390429617085644,
                                    0.6235776860273197,
                                    1.6559810006210114], ndmin=2)
    image_0_quaternion = np.array([0.6574280826403377,
                                   0.612626168885718,
                                   -0.29491259746065657,
                                   -0.32481387472099443])

    # image translation and quaternion for the second image
    image_6_translation = np.array([1.121783231448374,
                                    0.6276849491056922,
                                    1.3913821856210593], ndmin=2)
    image_6_quaternion = np.array([0.6691820728523993,
                                   0.6388828845567809,
                                   -0.2857923437256659,
                                   -0.24969331080580404])

    # compute rotation matrix from the quaternion
    # here we invert the rotation matrix because we need an inverse transfromation 
    image_0_rotation = np.linalg.inv(quaternion_to_rotation_matrix(image_0_quaternion)) 
    # compute Rodrigues vector from rotation matrix (is needed for OpenCV) 
    image_0_rodrigues, _ = cv2.Rodrigues(image_0_rotation)
    # update translation as well for the inverse transformation
    image_0_translation = -1.0 * np.matmul(image_0_rotation, image_0_translation.T)


    print("\nCamera 0 position:\n", image_0_translation) 
    print("\nCamera 0 rotation matrix\n", image_0_rotation) 
    print("\nCamera 0 rodrigues vector\n", image_0_rodrigues)

    # the same for the second image
    image_6_rotation = np.linalg.inv(quaternion_to_rotation_matrix(image_6_quaternion))
    image_6_rodrigues, _ = cv2.Rodrigues(image_6_rotation)
    image_6_translation = -1.0 * np.matmul(image_6_rotation, image_6_translation.T) 
    print("\nCamera 6 position\n", image_6_translation)
    print("\nCamera 6 rotation matrix\n", image_6_rotation) 
    print("\nCamera 6 rodrigues vector\n", image_6_rodrigues)

    # define the intrinsic matrix
    K = np.array([[525.0, 0.0, 319.5], 
                  [0.0, 525.0, 239.5], 
                  [0.0, 0.0, 1.0]])
    print("\nCamera intrinsic matrix\n", K)

    # compute projection matrices
    image_0_projection_matrix = np.matmul(K, 
                                          np.concatenate((image_0_rotation, image_0_translation), axis=1)) 
    image_6_projection_matrix = np.matmul(K, 
                                          np.concatenate((image_6_rotation, image_6_translation), axis=1)) 
    print("\nCamera 0 projection matrix\n", image_0_projection_matrix)
    print("\nCamera 6 projection matrix\n", image_6_projection_matrix)

    # triangulate 3D point
    point_3d = cv2.triangulatePoints(image_0_projection_matrix, 
                                     image_6_projection_matrix,  
                                     image_0_2d_point.T, 
                                     image_6_2d_point.T)
    point_3d = point_3d / point_3d[3]
    print("\n3D point (homogeneous coordinates)\n", point_3d)

    # reproject 3D point back to the images
    image_0_2d_point_reprojection, _ = cv2.projectPoints(point_3d[0:3].T,
                                                         image_0_rodrigues, image_0_translation, K, None) 
    reprojection_error_0 = np.linalg.norm(image_0_2d_point - \
                                          image_0_2d_point_reprojection) 
    print("\nReprojection 2D point:\n", image_0_2d_point_reprojection) 
    print("\nReprojection error for point 0: %.2f pixels\n" % reprojection_error_0)

    image_6_2d_point_reprojection, _ = \
                cv2.projectPoints(point_3d[0:3].T,
                                  image_6_rodrigues, image_6_translation, K, None) 
    reprojection_error_6 = np.linalg.norm(image_6_2d_point - \
                                          image_6_2d_point_reprojection) 
    print("\nReprojection 2D point: \n", image_6_2d_point_reprojection) 
    print("\nReprojection error for point 6: %.2f pixels\n" % reprojection_error_6)


def quaternion_to_rotation_matrix(quaternion):
    """
    Generate rotation matrix 3x3  from the unit quaternion.
    Input:
    qQuaternion -- tuple consisting of (qx,qy,qz,qw) where
         (qx,qy,qz,qw) is the unit quaternion.
    Output:
    matrix -- 3x3 rotation matrix
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    eps = np.finfo(float).eps * 4.0
    assert nq > eps
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1])
    ), dtype=np.float64)

if __name__ == '__main__':
    main()