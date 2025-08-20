#!/usr/bin/env python

import rospy
import numpy as np
import tf.transformations as tfs
import tf2_ros
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose, Twist, Vector3
from nav_msgs.msg import Odometry

class CubePoseEstimator:
    def __init__(self):
        rospy.init_node('cube_pose_estimator', anonymous=True)
        self.tag_cube_transforms = self.initialize_transforms()
        self.camera_frame = None

        # tf broadcaster and pose publisher
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.pose_publisher = rospy.Publisher('/cube_pose', PoseStamped, queue_size=10)
        self.odom_pub = rospy.Publisher('/obj_odometry', Odometry, queue_size=10)

        # state variables for velocity/acceleration calculation
        self.last_time = None
        self.last_pose_matrix = None
        self.odom_parent_frame = "camera_color_optical_frame"
        self.odom_child_frame = "cube"

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

    def detections_callback(self, msg):
        if not msg.detections:
            return
        
        if self.camera_frame is None:
            self.camera_frame = msg.detections[0].pose.header.frame_id
            self.odom_parent_frame = self.camera_frame
            rospy.loginfo("locked camera frame to: %s", self.camera_frame)

        poses = []
        for detection in msg.detections:
            tag_id = detection.id[0]

            if tag_id not in self.tag_cube_transforms:
                #rospy.logwarn("Unknown tag ID: %d", tag_id)
                continue

            # calculate tensor of the camera to tag transform
            p = detection.pose.pose.pose.position
            q = detection.pose.pose.pose.orientation
            T_camera_tag = tfs.quaternion_matrix([q.x, q.y, q.z, q.w])
            T_camera_tag[:3, 3] = [p.x, p.y, p.z]

            # trans = tfs.translation_from_matrix(T_camera_tag)
            # quat = tfs.quaternion_from_matrix(T_camera_tag)
            # self.publish_tf(trans, quat, self.camera_frame, "t_{}".format(tag_id))

            # calculate tensor of the camera to cube transform
            T_tag_cube = self.tag_cube_transforms[tag_id]
            T_camera_cube = np.dot(T_camera_tag, T_tag_cube)

            poses.append(T_camera_cube) # append to list for averaging
            # trans = tfs.translation_from_matrix(T_camera_cube)
            # quat = tfs.quaternion_from_matrix(T_camera_cube)
            # self.publish_tf(trans, quat, self.camera_frame, "c_{}".format(tag_id))
        
        # calculate average position for the cube from all visible tags
        T_camera_cube_avg = self.average_pose(poses)
        if T_camera_cube_avg is not None:
            current_time = rospy.Time.now()

            # extract pose from the final matrix
            trans = tfs.translation_from_matrix(T_camera_cube_avg)
            quat = tfs.quaternion_from_matrix(T_camera_cube_avg)

            # publish the pose
            self.publish_tf(trans, quat, self.camera_frame, "cube_pos")
    
            # calculate velocities
            linear_vel = Vector3()
            angular_vel = Vector3()
            if self.last_time is not None and self.last_pose_matrix is not None:
                dt = (current_time - self.last_time).to_sec()
                if dt > 0:
                    # calculate linear velocity
                    last_trans = tfs.translation_from_matrix(self.last_pose_matrix)
                    linear_vel.x = (trans[0] - last_trans[0]) / dt
                    linear_vel.y = (trans[1] - last_trans[1]) / dt
                    linear_vel.z = (trans[2] - last_trans[2]) / dt

                    # calculate angular velocity
                    T_last_inv = tfs.inverse_matrix(self.last_pose_matrix)
                    T_rot_diff = np.dot(T_last_inv, T_camera_cube_avg)
                    angle, axis, _ = tfs.rotation_from_matrix(T_rot_diff)

                    angular_vel.x = axis[0] * angle / dt
                    angular_vel.y = axis[1] * angle / dt
                    angular_vel.z = axis[2] * angle / dt
            
            # assemble and publish the odometry message
            odom_msg = Odometry()
            odom_msg.header.stamp = current_time
            odom_msg.header.frame_id = self.odom_parent_frame
            odom_msg.child_frame_id = self.odom_child_frame

            odom_msg.pose.pose.position.x = trans[0]
            odom_msg.pose.pose.position.y = trans[1]
            odom_msg.pose.pose.position.z = trans[2]
            odom_msg.pose.pose.orientation.x = quat[0]
            odom_msg.pose.pose.orientation.y = quat[1]
            odom_msg.pose.pose.orientation.z = quat[2]
            odom_msg.pose.pose.orientation.w = quat[3]

            odom_msg.twist.twist.linear = linear_vel
            odom_msg.twist.twist.angular = angular_vel

            self.odom_pub.publish(odom_msg)

            # update state
            self.last_time = current_time
            self.last_pose_matrix = T_camera_cube_avg

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

        # ensure all quaternions are in the same hemisphere for stable averaging
        for i in range(1, len(quaternions)):
            if np.dot(quaternions[0], quaternions[i]) < 0:
                quaternions[i] *= -1

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