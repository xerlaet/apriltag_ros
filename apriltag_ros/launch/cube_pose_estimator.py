#!/usr/bin/env python

import rospy
import numpy as np
import tf.transformations as tfs
import tf2_ros
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseStamped, TransformStamped

class CubePoseEstimator:
    def __init__(self):
        rospy.init_node('cube_pose_estimator', anonymous=True)
        self.tag_cube_transforms = self.initialize_transforms()
        self.camera_frame = None

        # tf broadcaster and pose publisher
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.pose_publisher = rospy.Publisher('/cube_pose', PoseStamped, queue_size=10)

        # subscribe to apriltag detections
        rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.detections_callback)

        # logs
        rospy.loginfo("cube pose estimator node initialized")
        rospy.loginfo("waiting for tag detections on /tag_detections...")

    def initialize_transforms(self):
        self.cube_side_length = 0.06
        L = self.cube_side_length / 2.0

        transforms = {}

        # define poses for tags 1-6
        tag_cube_poses = {
            1: ([ 0, 0, -L], tfs.quaternion_from_euler(0, np.pi, 0)),          # front (blue)
            2: ([ 0, 0, -L], tfs.quaternion_from_euler(0, -np.pi/2, 0)),       # left (yellow)
            3: ([ 0, 0, -L], tfs.quaternion_from_euler(-np.pi/2, 0, np.pi/2)), # bottom (red)
            4: ([ 0, 0, -L], tfs.quaternion_from_euler(np.pi/2, 0, -np.pi/2)), # top (white)
            5: ([ 0, 0, -L], tfs.quaternion_from_euler(0, np.pi/2, 0)),        # right (green)
            6: ([ 0, 0, -L], tfs.quaternion_from_euler(0, 0, 0)),              # back (cyan)
        }

        for tag_id, (p_tag_cube, q_tag_cube) in tag_cube_poses.items():
            T_tag_cube = tfs.quaternion_matrix(q_tag_cube)
            T_tag_cube[:3, 3] = p_tag_cube
            transforms[tag_id] = T_tag_cube

        return transforms

    # convert from euler to quaternion: tfs.quaternion_from_euler(roll, pitch yaw, axes='sxyz')
    # convert from quaternion to euler: tfs.euler_from_quaternion(quat, axes='sxyz')
    # convert from quaternion to a rotation matrix: tfs.quaternion_matrix(quat)
    # convert from rotation matrix to quaternion: tfs.quaternion_from_matrix(matrix)
    # convert from translation matrix to translation vector: tfs.translation_from_matrix(matrix)

    # combine two rotations: tfs.quaternion_multiply(q1, q2)
    # combine two transformation matrices: np.dot(T1, T2)

    # inverse of a rotation: tfs.quaternion_inverse(q)
    # inverse of a transformation matrix: tfs.inverse_matrix(T)

    def detections_callback(self, msg):
        if not msg.detections:
            return
        
        if self.camera_frame is None:
            self.camera_frame = msg.detections[0].pose.header.frame_id
            rospy.loginfo("locked camera frame to: %s", self.camera_frame)

        poses = []

        for detection in msg.detections:
            tag_id = detection.id[0]

            if tag_id not in self.tag_cube_transforms:
                rospy.logwarn("Unknown tag ID: %d", tag_id)
                continue

            # calculate tensor of the camera to tag transform
            p = detection.pose.pose.pose.position
            q = detection.pose.pose.pose.orientation
            T_camera_tag = tfs.quaternion_matrix([q.x, q.y, q.z, q.w])
            T_camera_tag[:3, 3] = [p.x, p.y, p.z]

            trans = tfs.translation_from_matrix(T_camera_tag)
            quat = tfs.quaternion_from_matrix(T_camera_tag)
            self.publish_tf(trans, quat, self.camera_frame, "t_{}".format(tag_id))

            # calculate tensor of the camera to cube transform
            T_tag_cube = self.tag_cube_transforms[tag_id]
            T_camera_cube = np.dot(T_camera_tag, T_tag_cube)

            poses.append(T_camera_cube) # append to list for averaging
            trans = tfs.translation_from_matrix(T_camera_cube)
            quat = tfs.quaternion_from_matrix(T_camera_cube)
            self.publish_tf(trans, quat, self.camera_frame, "c_{}".format(tag_id))
        
        # calculate average position for the cube
        T_camera_cube_avg = self.average_pose(poses)
        if T_camera_cube_avg is not None:
            trans = tfs.translation_from_matrix(T_camera_cube_avg)
            quat = tfs.quaternion_from_matrix(T_camera_cube_avg)
            self.publish_tf(trans, quat, self.camera_frame, "avg")
    
    def average_pose(self, poses):
        if not poses:
            return None
        if len(poses) == 1:
            return poses[0]
        
        # average the translations
        translations = [tfs.translation_from_matrix(p) for p in poses]
        avg_translation = np.mean(translations, axis=0)

        # average the quaternions
        quaternions = [tfs.quaternion_from_matrix(p) for p in poses]
        avg_quaternion = np.mean(quaternions, axis=0)
        avg_quaternion /= np.linalg.norm(avg_quaternion)  # normalize quaternion

        avg_matrix = tfs.quaternion_matrix(avg_quaternion)
        avg_matrix[:3, 3] = avg_translation

        return avg_matrix

    def publish_tf(self, trans, quat, parent_frame, child_frame):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = trans[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(t)
            
    def publish_pose(self, trans, quat, frame_id):
        p = PoseStamped()
        p.header.stamp = rospy.Time.now()
        p.header.frame_id = frame_id
        p.pose.position.x = trans[0]
        p.pose.position.y = trans[1]
        p.pose.position.z = trans[2]
        p.pose.orientation.x = quat[0]
        p.pose.orientation.y = quat[1]
        p.pose.orientation.z = quat[2]
        p.pose.orientation.w = quat[3]
        self.pose_publisher.publish(p)

if __name__ == "__main__":
    try:
        CubePoseEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass