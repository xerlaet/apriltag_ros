#!/usr/bin/env python

import rospy
import numpy as np
import tf.transformations as tfs
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from apriltag_ros.msg import AprilTagDetectionArray

class CubePublisher():
    def __init__(self):
        self.tag_cube_transforms = self.initialize_transforms()
        self.camera_frame = None
        
        self.pub = rospy.Publisher('/obj_odometry', Odometry, queue_size=1)
        self.sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.detections_callback, queue_size=1)

        # state variables for cube velocity calculation
        self.last_time = None
        self.last_pose_matrix = None
        self.current_time = None
        self.current_pose_matrix = None

        # logs
        rospy.loginfo("cube pose estimator node initialized")
        rospy.loginfo("waiting for tag detections on /tag_detections...")

    def initialize_transforms(self):
        # define cube size
        self.cube_side_length = 0.06
        L = self.cube_side_length / 2.0

        # define poses for tags 1-6
        tag_cube_poses = {
            1: ([ 0, 0, -L], tfs.quaternion_from_euler(0, np.pi, 0)),          # front (blue)
            2: ([ 0, 0, -L], tfs.quaternion_from_euler(0, -np.pi/2, 0)),       # left (yellow)
            3: ([ 0, 0, -L], tfs.quaternion_from_euler(-np.pi/2, 0, np.pi/2)), # bottom (red)
            4: ([ 0, 0, -L], tfs.quaternion_from_euler(np.pi/2, 0, -np.pi/2)), # top (white)
            5: ([ 0, 0, -L], tfs.quaternion_from_euler(0, np.pi/2, 0)),        # right (green)
            6: ([ 0, 0, -L], tfs.quaternion_from_euler(0, 0, 0)),              # back (cyan)
        }

        # list transforms
        transforms = {}
        for tag_id, (p_tag_cube, q_tag_cube) in tag_cube_poses.items():
            T_tag_cube = tfs.quaternion_matrix(q_tag_cube)
            T_tag_cube[:3, 3] = p_tag_cube
            transforms[tag_id] = T_tag_cube
        return transforms

    def detections_callback(self, msg):
        if not msg.detections:
            return
        if self.camera_frame is None:
            self.camera_frame = msg.detections[0].pose.header.frame_id
            rospy.loginfo("locked camera frame to: %s", self.camera_frame)

        # process detections
        poses = []
        for detection in msg.detections:
            tag_id = detection.id[0]

            # only recognize tags that are on the cube
            if tag_id in self.tag_cube_transforms:
                # calculate camera to tag transform
                p = detection.pose.pose.pose.position
                q = detection.pose.pose.pose.orientation
                T_camera_tag = tfs.quaternion_matrix([q.x, q.y, q.z, q.w])
                T_camera_tag[:3, 3] = [p.x, p.y, p.z]

                # calculate camera to cube transform and append to list
                T_tag_cube = self.tag_cube_transforms[tag_id]
                T_camera_cube = np.dot(T_camera_tag, T_tag_cube)
                poses.append(T_camera_cube)

        # calculate the current average pose and time
        self.current_time = rospy.Time.now()
        self.current_pose_matrix = self.calculate_average_pose(poses)

        # publish odometry if everything is valid
        if self.current_pose_matrix is not None:
            if self.last_pose_matrix is not None:
                # calculate values
                dt = (self.current_time - self.last_time).to_sec()
                linear_vel, angular_vel = self.calculate_velocities(dt)
                trans = tfs.translation_from_matrix(self.current_pose_matrix)
                quat = tfs.quaternion_from_matrix(self.current_pose_matrix)

                # publish odometry message
                self.publish_odom(trans, quat, linear_vel, angular_vel)

            # keep track of previous pose and time
            self.last_time = self.current_time
            self.last_pose_matrix = self.current_pose_matrix.copy()
    
    def publish_odom(self, trans, quat, linear_vel, angular_vel):
        msg = Odometry()
        msg.header.stamp = self.current_time
        msg.header.frame_id = self.camera_frame
        msg.child_frame_id = self.camera_frame

        msg.pose.pose.position.x = trans[0]
        msg.pose.pose.position.y = trans[1]
        msg.pose.pose.position.z = trans[2]
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]

        msg.twist.twist.linear = linear_vel
        msg.twist.twist.angular = angular_vel

        self.pub.publish(msg)
            
    def calculate_average_pose(self, poses):
        if not poses:
            return None
        if len(poses) == 1:
            return poses[0]
        
        # average the translations
        translations = [tfs.translation_from_matrix(p) for p in poses]
        avg_translation = np.mean(translations, axis=0)

        # average the quaternions
        quaternions = [tfs.quaternion_from_matrix(p) for p in poses]
        for i in range(1, len(quaternions)): # ensure all are in the same hemisphere
            if np.dot(quaternions[0], quaternions[i]) < 0:
                quaternions[i] *= -1
        avg_quaternion = np.mean(quaternions, axis=0)
        avg_quaternion /= np.linalg.norm(avg_quaternion) # normalize

        # construct average transformation matrix
        avg_matrix = tfs.quaternion_matrix(avg_quaternion)
        avg_matrix[:3, 3] = avg_translation
        return avg_matrix

    def calculate_velocities(self, dt):
        linear_vel = Vector3()
        angular_vel = Vector3()

        if dt > 0:
            # calculate linear velocity
            trans = tfs.translation_from_matrix(self.current_pose_matrix)
            last_trans = tfs.translation_from_matrix(self.last_pose_matrix)
            linear_vel.x = (trans[0] - last_trans[0]) / dt
            linear_vel.y = (trans[1] - last_trans[1]) / dt
            linear_vel.z = (trans[2] - last_trans[2]) / dt

            # calculate angular velocity
            R1 = self.last_pose_matrix[:3,:3]
            R2 = self.current_pose_matrix[:3,:3]
            angular_vel_camera = self.calculate_angular_velocity(R1, R2, dt)
            angular_vel.x = angular_vel_camera[0]
            angular_vel.y = angular_vel_camera[1]
            angular_vel.z = angular_vel_camera[2]
        
        return linear_vel, angular_vel

    def calculate_angular_velocity(self, R1, R2, dt):
        # compute relative rotation matrix
        R_rel = R2 @ R1.T

        # compute rotation angle
        trace = np.trace(R_rel)
        cos_theta = (trace - 1) / 2
        cos_theta = np.clip(cos_theta, -1, 1) # handle numerical errors
        theta = np.arccos(cos_theta)

        # compute rotation axis
        if np.abs(np.sin(theta)) > 1e-6: # avoid division by zero
            n = (1 / (2 * np.sin(theta))) * np.array([
                R_rel[2, 1] - R_rel[1, 2],
                R_rel[0, 2] - R_rel[2, 0],
                R_rel[1, 0] - R_rel[0, 1]
            ])
        else:
            n = np.zeros(3) # no rotation

        # compute angular velocity
        omega = (theta / dt) * n
        return omega
    
if __name__ == "__main__":
    try:
        rospy.init_node('cube_publisher', anonymous=True)
        node = CubePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
