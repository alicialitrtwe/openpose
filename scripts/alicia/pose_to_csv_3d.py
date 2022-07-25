##############################
# Imports
import os, re, json, sys

import numpy as np
import pandas as pd

##############################
# basic pose vars
DEFAULT_KEYPOINTS = {
    'body': 'pose_keypoints_2d',
    'face': 'face_keypoints_2d',
    'left_hand': 'hand_left_keypoints_2d',
    'right_hand': 'hand_right_keypoints_2d',
    'body_3d': 'pose_keypoints_3d',
    'face_3d': 'face_keypoints_3d',
    'left_hand_3d': 'hand_left_keypoints_3d',
    'right_hand_3d': 'hand_right_keypoints_3d'
}

N_POINTS = 138
N_DIMS = 3  # Default dimension of pose estimates, 2 or 3
N_COORDS = N_DIMS + 1  # Include confidence
BODY_RANGE = slice(0, 25*N_COORDS)      # Assume BODY_25 output from OpenPose
L_HAND_RANGE = slice(26*N_COORDS, 47*N_COORDS)
R_HAND_RANGE = slice(47*N_COORDS, 68*N_COORDS)
FACE_RANGE = slice(68*N_COORDS, 138*N_COORDS)

# Prepare the names of all the individual pose points
BODY_POINT_NAMES = [
    'Nose',
    'Neck',
    'RShoulder',
    'RElbow',
    'RWrist',
    'LShoulder',
    'LElbow',
    'LWrist',
    'MidHip',
    'RHip',
    'RKnee',
    'RAnkle',
    'LHip',
    'LKnee',
    'LAnkle',
    'REye',
    'LEye',
    'REar',
    'LEar',
    'LBigToe',
    'LSmallToe',
    'LHeel',
    'RBigToe',
    'RSmallToe',
    'RHeel',
    'Background'
]
HAND_POINT_NAMES = [
    'Wrist',
    'Base',
    'Proximal',
    'Distal',
    'Tip',
    'Base',
    'Proximal',
    'Distal',
    'Tip',
    'Base',
    'Proximal',
    'Distal',
    'Tip',
    'Base',
    'Proximal',
    'Distal',
    'Tip',
    'Base',
    'Proximal',
    'Distal',
    'Tip',
]
FACE_POINT_NAMES = ['placeholder'] * 70
ALL_POINT_NAMES = [
    *BODY_POINT_NAMES,
    *HAND_POINT_NAMES,
    *HAND_POINT_NAMES,
    *FACE_POINT_NAMES
]

# Prepare the names of body components each point belongs to
BODY_COMPONENTS = [
    'Head',
    'Torso',
    'Torso',
    'RArm',
    'RArm',
    'Torso',
    'LArm',
    'LArm',
    'Torso',
    'Torso',
    'RLeg',
    'RLeg',
    'Torso',
    'LLeg',
    'LLEg',
    'Head',
    'Head',
    'Head',
    'Head',
    'LLeg',
    'LLeg',
    'LLeg',
    'RLeg',
    'RLeg',
    'RLeg',
    'Background'
]
HAND_COMPONENTS = [
    'Palm',
    'Palm',
    'Thumb',
    'Thumb',
    'Thumb',
    'Palm',
    'Pointer',
    'Pointer',
    'Pointer',
    'Palm',
    'Middle',
    'Middle',
    'Middle',
    'Palm',
    'Ring',
    'Ring',
    'Ring',
    'Palm',
    'Pinky',
    'Pinky',
    'Pinky',
]
FACE_COMPONENTS = ['placeholder'] * 70
ALL_COMPONENTS = [
    *BODY_COMPONENTS,
    *HAND_COMPONENTS,
    *HAND_COMPONENTS,
    *FACE_COMPONENTS
]

# Prepare the names of the pose estimation set each point belongs to
POSE_SETS = [*(['Body']*26), *(['L Hand']*21), *(['R Hand']*21), *(['Face']*70)]

BODY_CONNECTORS = [
    [(), ()]
]

##############################
# function sort list in alpha-numeric
# order
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


def expand_names(base_names, n_coords=N_COORDS):
    """Expand lists of names to cover the x y (z) c """
    expanded_names = []
    for name in base_names:
        expanded_names.extend([name]*n_coords)
    return expanded_names

def json_to_csv(source_dir, save_path):
    """
    Covert a collection of JSON pose estimate keyframe files to a DataFrame representation
    Current implementation assumes the BODY_25 output from OpenPose
    """
    print("entered json_to_csv")
    video_name = os.path.basename(os.path.normpath(save_path)) + "_" + source_dir.split("/")[-1]
    out_name = os.path.join(os.path.dirname(save_path), f'pose_{video_name}.csv')

    files = [f for f in os.listdir(source_dir) if '.json' in f]
    print("files: ", files)
    files.sort(key=natural_keys)
    n_frames = len(files)
    all_pose = np.zeros((n_frames, N_POINTS*N_COORDS))
    all_ids = []

    for i, f in enumerate(files):
        with open(os.path.join(source_dir, f)) as datafile:
            json_data = json.load(datafile)

        # Extract the frame id from the file name
        print("f: ", f)
        # id = int(re.search('_([0-9]{1,3})_keypoints', f).group(1))
        # all_ids.append(id)

        # Collect all keypoints into a single long list
        if json_data['people']:
            pose_data = json_data['people'][0]
            if pose_data[DEFAULT_KEYPOINTS['body_3d']]:
                all_pose[i, BODY_RANGE] = pose_data[DEFAULT_KEYPOINTS['body_3d']]

            if pose_data[DEFAULT_KEYPOINTS['left_hand_3d']]:
                all_pose[i, L_HAND_RANGE] = pose_data[DEFAULT_KEYPOINTS['left_hand_3d']]

            if pose_data[DEFAULT_KEYPOINTS['right_hand_3d']]:
                all_pose[i, R_HAND_RANGE] = pose_data[DEFAULT_KEYPOINTS['right_hand_3d']]

            if pose_data[DEFAULT_KEYPOINTS['face_3d']]:
                all_pose[i, FACE_RANGE] = pose_data[DEFAULT_KEYPOINTS['face_3d']]

    # Wrap the numpy array into a dataframe with a multiindex
    pose_df = pd.DataFrame.from_records(all_pose)

    coord_names = ['x', 'y', 'z', 'c'] * N_POINTS
    all_point_names = expand_names(ALL_POINT_NAMES)
    body_part_names = expand_names(ALL_COMPONENTS)
    pose_sets = expand_names(POSE_SETS)
    big_index = pd.MultiIndex.from_tuples(
        zip(pose_sets, body_part_names, all_point_names, coord_names),
        names=['Pose Set', 'Body Part', 'Point', 'Coordinate']
    )
    pose_df.columns = big_index

    # print("os.path.dirname(save_path): ", os.path.dirname(save_path))
    # out_name = os.path.join(os.path.dirname(save_path), f'pose_{video_name}.csv')
    # print("csv name:", out_name)
    # out_name = os.path.dirname(save_path) + '/json_to_csv_3d/pose_3d.csv'
    #out_name = os.path.dirname(save_path) + 'C:/Users/User/CSE600/reconstruction-3d/animation/3d_keypoints/3-11-22_3d.csv'
    # out_name = 'C:/Users/User/CSE600/wasabi_videos/11-17-21_videos/11-17-21_3d-keypoints'
    return pose_df


if __name__ == '__main__':
    print("entered main")
    args = sys.argv[1:]
    source = args[0]
    save = args[1]

    print("source: ", source)
    print("save: ", save)
    print("os.listdir(source): ", os.listdir(source))

    folders = [os.path.join(source, f) for f in os.listdir(source) if os.path.isdir(os.path.join(source, f))]
    print("folders: ", folders)
    folders.sort(key=natural_keys)
    for folder in folders:
        print("about to enter json_to_csv")
        pose_df = json_to_csv(folder, save)
        folder_name = os.path.normpath(folder).split(os.path.sep)[-1]
        pose_df.to_csv(os.path.join(save, folder_name + '.csv'))