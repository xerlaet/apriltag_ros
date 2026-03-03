#!/usr/bin/env python

import rospy
import numpy as np
import tf.transformations as tfs
from geometry_msgs.msg import Vector3, Point, Quaternion
from nav_msgs.msg import Odometry
from apriltag_ros.msg import AprilTagDetectionArray
import tf2_ros

class CubeBundlePublisher():
    def __init__(self):
        # Parameters
        self.camera_frame = rospy.get_param('~camera_frame', 'rs_camera_infra1_optical_frame')
        self.bundle_frame = rospy.get_param('~bundle_frame', 'cube')
        self.odom_topic = rospy.get_param('~odom_topic', '/obj_odometry')
        
        self.pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=1)
        
        # TF listener to get the cube's pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # State variables for velocity calculation
        self.last_time = None
        self.last_pose_matrix = None

        rospy.loginfo("Cube Bundle Odometry Publisher initialized.")
        rospy.loginfo("Syncing with detections on /tag_detections...")

        # We subscribe to detections to trigger the TF lookup at the exact moment of detection
        self.sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.detections_callback, queue_size=1)

    def detections_callback(self, msg):
        try:
            # Lookup the transform for the bundle 'cube' at the detection timestamp
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame, 
                self.bundle_frame, 
                msg.header.stamp, 
                rospy.Duration(0.05) 
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return

        current_time = transform.header.stamp
        
        # Avoid duplicate processing
        if self.last_time is not None and current_time <= self.last_time:
            return

        # Convert to matrix
        p = transform.transform.translation
        q = transform.transform.rotation
        current_pose_matrix = tfs.quaternion_matrix([q.x, q.y, q.z, q.w])
        current_pose_matrix[:3, 3] = [p.x, p.y, p.z]
        
        # Initialize odom message
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = self.camera_frame
        odom.child_frame_id = self.bundle_frame 
        
        odom.pose.pose.position = Point(p.x, p.y, p.z)
        odom.pose.pose.orientation = q
        
        # Calculate velocities if we have a previous pose
        if self.last_pose_matrix is not None:
            dt = (current_time - self.last_time).to_sec()
            if dt > 0:
                # Linear velocity in camera frame
                v_cam = (current_pose_matrix[:3, 3] - self.last_pose_matrix[:3, 3]) / dt
                
                # Angular velocity in camera frame
                R1 = self.last_pose_matrix[:3, :3]
                R2 = current_pose_matrix[:3, :3]
                R_rel = np.dot(R2, R1.T)
                
                # Prepare a 4x4 matrix for tf.transformations.rotation_from_matrix
                R_rel_44 = np.eye(4)
                R_rel_44[:3, :3] = R_rel
                
                try:
                    angle, axis, _ = tfs.rotation_from_matrix(R_rel_44)
                    omega_cam = (angle / dt) * axis
                except ValueError:
                    # Handle cases where rotation is zero or invalid
                    omega_cam = np.array([0.0, 0.0, 0.0])
                
                # Transform twist into child frame (bundle frame)
                # Twist in Odometry messages is conventionally in the child frame
                R_cam_bundle = R2.T
                v_bundle = np.dot(R_cam_bundle, v_cam)
                omega_bundle = np.dot(R_cam_bundle, omega_cam)
                
                odom.twist.twist.linear = Vector3(*v_bundle)
                odom.twist.twist.angular = Vector3(*omega_bundle)
                
                self.pub.publish(odom)

        self.last_time = current_time
        self.last_pose_matrix = current_pose_matrix

if __name__ == "__main__":
    try:
        rospy.init_node('cube_bundle_publisher', anonymous=True)
        node = CubeBundlePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
