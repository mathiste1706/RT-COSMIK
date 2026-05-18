import subprocess
import numpy as np
import os  
import cv2 as cv
import yaml

def list_cameras():
    """
    Use v4l2-ctl to list all connected cameras and their device paths.
    Return:
        cameras: dictionary of camera indices and associated device names.
    """
    cameras = {}
    try:
        # Get list of video devices
        output = subprocess.check_output("v4l2-ctl --list-devices", shell=True).decode("utf-8")
        devices = output.split("\n\n")  # Separate different devices
        for device in devices:
            lines = device.split("\n")
            if len(lines) > 1:
                device_name = lines[0].strip()
                video_path = lines[1].strip()
                if "/dev/video" in video_path:
                    index = int(video_path.split("video")[-1])
                    cameras[index] = device_name
    except Exception as e:
        print("Error using v4l2-ctl:", e)
    return cameras

def rt_to_homogeneous(R, translation_matrix):
    """
    Convert (R, translation_matrix) to a 4x4 homogeneous transformation matrix.
    Parameters: 
        R (numpy.ndarray): rotation matrix (3x3)
        translation_matrix (numpy.ndarray): translation matrix (3x1)
    Returns:
        T (numpy.ndarray): homogeneous translation matrix (4x4)
    """
    
    translation_matrix = translation_matrix.reshape(3,)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation_matrix
    return T

def invert_homogeneous(T):
    """
    Invert a 4x4 homogeneous transformation matrix.
    Parameter: 
        T (numpy.ndarray): homogeneous translation matrix (4x4)
    Returns:
        T_inv (numpy.ndarray): the inverse matrix of T
    """
    
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def decompose_homogeneous(T):
    """
    Extract (R, translation_matrix) from T a 4x4 homogeneous matrix.
    Parameter:
        T (numpy.ndarray): a 4x4 homogeneous matrix
    Returns:
        R (numpy.ndarray): Rotation matrix (3x3) from transformation matrix T
        translation_matrix (numpy.ndarray): translation_matrix (3x1) from transformation matrix T
    """
    
    R = T[:3, :3]
    translation_matrix = T[:3, 3]
    return R, translation_matrix


def load_cam_params(path):
    """
    Loads camera parameters from a given file.
    Parameter:
        path (str): The path to the file containing the camera parameters.
    Returns:
        tuple: A tuple containing the camera matrix and distortion matrix.
            - camera_matrix (numpy.ndarray): The camera matrix.
            - distortion_matrix (numpy.ndarray): The distortion matrix.
    """
    
    # FILE_STORAGE_READ
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve otherwise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    distortion_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return camera_matrix, distortion_matrix


def load_cam_to_cam_params(path):
    """
    Loads camera-to-camera calibration parameters from a given file.
    This function reads the rotation matrix R (3x3) and translation matrix (3x1) from a 
    specified file using OpenCV's FileStorage. The file should contain these parameters 
    stored under the keys 'R' and 'T'.
    Parameter:
        path (str): The file path to the calibration parameters.
    Returns:
        tuple: A tuple containing:
            - R (numpy.ndarray): The rotation matrix.
            - translation_matrix (numpy.ndarray): The translation matrix.
    """
    
    # FILE_STORAGE_READ
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve otherwise we only get a
    # FileNode object back instead of a matrix
    R = cv_file.getNode('R').mat()
    translation_matrix = cv_file.getNode('T').mat()

    cv_file.release()
    return R, translation_matrix

def load_global_cam_params(path, cam_index):
    """
    Loads the global camera transformation parameters for a specified camera
    from a YAML file. This function reads the rotation matrix R (3x3) and translation
    matrix (3x1) stored under the keys 'camera_{cam_index}_R' and 'camera_{cam_index}_T'.
    
    Parameters:
        path (str): The file path to the YAML file.
        cam_index (int): The camera index to load.
        
    Returns:
        tuple: A tuple containing:
            - R (numpy.ndarray): The rotation matrix.
            - translation_matrix (numpy.ndarray): The translation vector.
    """
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    R = cv_file.getNode(f'camera_{cam_index}_R').mat()
    translation_matrix = cv_file.getNode(f'camera_{cam_index}_T').mat()
    cv_file.release()
    return R, translation_matrix


def load_cam_pose(filename):
    """
        Load the rotation matrix (3x3) and translation matrix (3x1) from a YAML file.
        Parameters:
            filename (str): The path to the YAML file.
        Returns:
            rotation_matrix (np.ndarray): The 3x3 rotation matrix.
            translation_matrix (np.ndarray): The 3x1 translation matrix.
    """

    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    rotation_matrix = np.array(data['rotation_matrix']['data']).reshape((3, 3))
    translation_matrix = np.array(data['translation_vector']['data']).reshape((3, 1))
    
    return rotation_matrix, translation_matrix


def load_camera_parameters(config_path):
    """
    Load intrinsic and extrinsic camera parameters from a file.
    Parameters:
        config_path (str): The path to the configuration file.
    Returns:
        K1: the camera intrinsic matrix of camera 0.
        D1: the camera distortion matrix of camera 0.
        K2: the camera extrinsic matrix of camera 2.
        D2: the camera distortion matrix of camera 2.
        R: the rotation matrix between camera 0 and camera 2.
        translation_matrix: the translation matrix between camera 0 and camera 2.
    """
    K1, D1 = load_cam_params(os.path.join(config_path, "c0_params_color.yaml"))
    K2, D2 = load_cam_params(os.path.join(config_path, "c2_params_color.yaml"))
    R, translation_matrix = load_cam_to_cam_params(os.path.join(config_path, "c0_to_c2_params_color.yaml"))

    K_matrix_list=[np.array(K1), np.array(K2)]
    Distortion_matrix_list=[D1,D2]

    rotation_matrix_list=[np.eye(3), np.array(R)]
    translation_matrix_list=[np.zeros((3,1)), np.array(translation_matrix)]

    proj_camera1 = np.concatenate([rotation_matrix_list[0], translation_matrix_list[0]], axis=-1)
    proj_camera2 = np.concatenate([rotation_matrix_list[1], translation_matrix_list[1]], axis=-1)
    proj_camera_list=[proj_camera1, proj_camera2]

    return K_matrix_list, Distortion_matrix_list, proj_camera_list, rotation_matrix_list, translation_matrix_list

def load_world_transformation(config_path):
    """
    Load world transformation matrix from a file.
    Parameters:
        config_path (str): The path to the configuration file.
    Returns:
        world_R1_cam (np.ndarray): The rotation matrix from camera 0 to the world.
        world_translation1_cam (np.ndarray): The translation matrix from camera 0 to the world.
    """
    world_R1_cam, world_translation1_cam = load_cam_pose(os.path.join(config_path, "camera0_pose.yaml"))
    return world_R1_cam, world_translation1_cam.reshape((3,))