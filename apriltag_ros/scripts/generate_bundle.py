
import numpy as np
import tf.transformations as tfs

def generate_bundle_yaml():
    cube_side_length = 0.07
    L = cube_side_length / 2.0
    tag_dist_from_center = 0.0169418 # measured from sticker sheet
    tag_size = 0.0204 # From tags.yaml

    corners = {
        'tl': [-tag_dist_from_center,  tag_dist_from_center, 0], # top left
        'tr': [ tag_dist_from_center,  tag_dist_from_center, 0], # top right
        'bl': [-tag_dist_from_center, -tag_dist_from_center, 0], # bottom left
        'br': [ tag_dist_from_center, -tag_dist_from_center, 0], # bottom right
    }

    faces = {
        'front': {
            'translation': [0, 0, -L],
            'rotation': tfs.quaternion_from_euler(0, np.pi, 0),
            'tag_ids': [1, 2, 3, 4]
        },
        'left': {
            'translation': [0, 0, -L],
            'rotation': tfs.quaternion_from_euler(0, -np.pi/2, 0),
            'tag_ids': [5, 6, 7, 8]
        },
        'bottom': {
            'translation': [0, 0, -L],
            'rotation': tfs.quaternion_from_euler(-np.pi/2, 0, np.pi/2),
            'tag_ids': [9, 10, 11, 12]
        },
        'top': {
            'translation': [0, 0, -L],
            'rotation': tfs.quaternion_from_euler(np.pi/2, 0, -np.pi/2),
            'tag_ids': [13, 14, 15, 16]
        },
        'right': {
            'translation': [0, 0, -L],
            'rotation': tfs.quaternion_from_euler(0, np.pi/2, 0),
            'tag_ids': [17, 18, 19, 20]
        },
        'back': {
            'translation': [0, 0, -L],
            'rotation': tfs.quaternion_from_euler(0, 0, 0),
            'tag_ids': [21, 22, 23, 24]
        }
    }

    print("tag_bundles:")
    print("  [")
    print("    {")
    print("      name: 'cube',")
    print("      layout:")
    print("        [")

    for face_name, face_data in faces.items():
        T_face_cube = tfs.quaternion_matrix(face_data['rotation'])
        T_face_cube[:3, 3] = face_data['translation']
        T_cube_face = np.linalg.inv(T_face_cube)

        for i, tag_id in enumerate(face_data['tag_ids']):
            corner_pos = corners[list(corners.keys())[i]]
            T_face_tag = np.eye(4)
            T_face_tag[:3, 3] = corner_pos

            T_cube_tag = T_cube_face @ T_face_tag
            
            trans = T_cube_tag[:3, 3]
            quat = tfs.quaternion_from_matrix(T_cube_tag)

            print(f"          {{id: {tag_id}, size: {tag_size}, x: {trans[0]:.6f}, y: {trans[1]:.6f}, z: {trans[2]:.6f}, qw: {quat[3]:.6f}, qx: {quat[0]:.6f}, qy: {quat[1]:.6f}, qz: {quat[2]:.6f}}},")

    print("        ]")
    print("    }")
    print("  ]")

if __name__ == "__main__":
    generate_bundle_yaml()
