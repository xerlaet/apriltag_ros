#!/usr/bin/env python
"""
Cube Bundle Odometry Publisher (Individual Tags + Robust Outlier Rejection)

This node subscribes to /tag_detections and uses the known physical layout
of all 24 AprilTags on the cube to compute a robust cube pose in the camera frame.

Features:
- Per-tag cube pose estimation from individual detections
- Robust outlier rejection (translation + rotation consensus)
- Minimum number of valid tags required before publishing
- Velocity computation in camera frame (no Kalman filter)
"""

import rospy
import numpy as np
import tf.transformations as tfs
from geometry_msgs.msg import Vector3, Point, Quaternion
from nav_msgs.msg import Odometry
from apriltag_ros.msg import AprilTagDetectionArray


# =============================================================================
# CONFIGURATION FLAGS
# =============================================================================
ENABLE_OUTLIER_REJECTION = True          # Master switch for outlier rejection


class CubeBundlePublisher():
    def __init__(self):
        # === ROS Parameters ===
        self.camera_frame = rospy.get_param('~camera_frame', None)
        self.bundle_frame = rospy.get_param('~bundle_frame', 'cube')
        self.odom_topic = rospy.get_param('~odom_topic', '/obj_odometry')
        self.detections_topic = rospy.get_param('~detections_topic', '/tag_detections')

        self.pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=1)

        # === Cube physical layout (from tags.yaml bundle) ===
        # Each entry defines the pose of the tag in the cube frame (T_cube_to_tag)
        self.cube_tag_layout = {
            3:  {'x': 0.016577, 'y': 0.016577, 'z': -0.035000, 'qw': 0.0,     'qx': 0.0,     'qy': 1.0,     'qz': 0.0},
            4:  {'x': -0.016577, 'y': 0.016577, 'z': -0.035000, 'qw': 0.0,    'qx': 0.0,     'qy': 1.0,     'qz': 0.0},
            1:  {'x': 0.016577, 'y': -0.016577, 'z': -0.035000, 'qw': 0.0,    'qx': 0.0,     'qy': 1.0,     'qz': 0.0},
            2:  {'x': -0.016577, 'y': -0.016577, 'z': -0.035000, 'qw': 0.0,   'qx': 0.0,     'qy': 1.0,     'qz': 0.0},
            11: {'x': 0.035000, 'y': 0.016577, 'z': 0.016577, 'qw': 0.707107, 'qx': 0.0,     'qy': 0.707107, 'qz': 0.0},
            12: {'x': 0.035000, 'y': 0.016577, 'z': -0.016577, 'qw': 0.707107, 'qx': 0.0,    'qy': 0.707107, 'qz': 0.0},
            9:  {'x': 0.035000, 'y': -0.016577, 'z': 0.016577, 'qw': 0.707107, 'qx': 0.0,    'qy': 0.707107, 'qz': 0.0},
            10: {'x': 0.035000, 'y': -0.016577, 'z': -0.016577, 'qw': 0.707107, 'qx': 0.0,   'qy': 0.707107, 'qz': 0.0},
            19: {'x': 0.016577, 'y': -0.035000, 'z': 0.016577, 'qw': 0.5,     'qx': 0.5,     'qy': 0.5,     'qz': -0.5},
            20: {'x': 0.016577, 'y': -0.035000, 'z': -0.016577, 'qw': 0.5,    'qx': 0.5,     'qy': 0.5,     'qz': -0.5},
            17: {'x': -0.016577, 'y': -0.035000, 'z': 0.016577, 'qw': 0.5,    'qx': 0.5,     'qy': 0.5,     'qz': -0.5},
            18: {'x': -0.016577, 'y': -0.035000, 'z': -0.016577, 'qw': 0.5,   'qx': 0.5,     'qy': 0.5,     'qz': -0.5},
            23: {'x': -0.016577, 'y': 0.035000, 'z': 0.016577, 'qw': 0.5,     'qx': -0.5,    'qy': 0.5,     'qz': 0.5},
            24: {'x': -0.016577, 'y': 0.035000, 'z': -0.016577, 'qw': 0.5,    'qx': -0.5,    'qy': 0.5,     'qz': 0.5},
            21: {'x': 0.016577, 'y': 0.035000, 'z': 0.016577, 'qw': 0.5,      'qx': -0.5,    'qy': 0.5,     'qz': 0.5},
            22: {'x': 0.016577, 'y': 0.035000, 'z': -0.016577, 'qw': 0.5,     'qx': -0.5,    'qy': 0.5,     'qz': 0.5},
            15: {'x': -0.035000, 'y': 0.016577, 'z': -0.016577, 'qw': 0.707107, 'qx': 0.0,   'qy': -0.707107, 'qz': 0.0},
            16: {'x': -0.035000, 'y': 0.016577, 'z': 0.016577, 'qw': 0.707107, 'qx': 0.0,    'qy': -0.707107, 'qz': 0.0},
            13: {'x': -0.035000, 'y': -0.016577, 'z': -0.016577, 'qw': 0.707107, 'qx': 0.0,  'qy': -0.707107, 'qz': 0.0},
            14: {'x': -0.035000, 'y': -0.016577, 'z': 0.016577, 'qw': 0.707107, 'qx': 0.0,   'qy': -0.707107, 'qz': 0.0},
            7:  {'x': -0.016577, 'y': 0.016577, 'z': 0.035000, 'qw': 1.0,     'qx': 0.0,     'qy': 0.0,     'qz': 0.0},
            8:  {'x': 0.016577, 'y': 0.016577, 'z': 0.035000, 'qw': 1.0,      'qx': 0.0,     'qy': 0.0,     'qz': 0.0},
            5:  {'x': -0.016577, 'y': -0.016577, 'z': 0.035000, 'qw': 1.0,    'qx': 0.0,     'qy': 0.0,     'qz': 0.0},
            6:  {'x': 0.016577, 'y': -0.016577, 'z': 0.035000, 'qw': 1.0,     'qx': 0.0,     'qy': 0.0,     'qz': 0.0},
        }

        # Pre-compute 4x4 transforms T_cube_to_tag
        self.T_cube_to_tag = {}
        for tag_id, layout in self.cube_tag_layout.items():
            quat = [layout['qx'], layout['qy'], layout['qz'], layout['qw']]
            T = tfs.quaternion_matrix(quat)
            T[:3, 3] = [layout['x'], layout['y'], layout['z']]
            self.T_cube_to_tag[tag_id] = T

        # === Outlier Rejection Parameters ===
        self.outlier_trans_threshold = rospy.get_param('~outlier_trans_threshold_m', 0.01)
        self.outlier_rot_threshold_rad = rospy.get_param('~outlier_rot_threshold_rad', 0.075)
        self.min_tags_for_outlier_rejection = rospy.get_param('~min_tags_for_outlier_rejection', 3)

        # === Minimum tags required to publish odometry ===
        self.min_tags_to_publish = rospy.get_param('~min_tags_to_publish', 3)

        # State for velocity calculation
        self.last_time = None
        self.last_pose_matrix = None

        rospy.loginfo("CubeBundlePublisher initialized.")
        rospy.loginfo("  - Detections topic: %s", self.detections_topic)
        rospy.loginfo("  - Outlier rejection enabled: %s", ENABLE_OUTLIER_REJECTION)
        rospy.loginfo("  - Min tags to publish: %d", self.min_tags_to_publish)
        rospy.loginfo("  - Outlier thresholds: trans=%.3fm, rot=%.2frad",
                      self.outlier_trans_threshold, self.outlier_rot_threshold_rad)

        self.sub = rospy.Subscriber(self.detections_topic, AprilTagDetectionArray,
                                    self.detections_callback, queue_size=1)

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================

    def _normalize_quaternion(self, q):
        """Normalize a quaternion to unit length."""
        q = np.array(q, dtype=float)
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return q / norm

    def _angular_distance(self, q1, q2):
        """
        Compute the angular distance (in radians) between two quaternions.
        Result is always between 0 and pi.
        """
        q1 = self._normalize_quaternion(q1)
        q2 = self._normalize_quaternion(q2)
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)
        return 2.0 * np.arccos(dot)

    def _choose_best_rotation_reference(self, quats):
        """
        Choose the quaternion that has the smallest total angular distance
        to all other quaternions in the set.
        This is a robust, order-independent way to pick a reference orientation.
        """
        if len(quats) == 1:
            return quats[0]

        best_quat = quats[0]
        best_score = float('inf')

        for q_i in quats:
            total_dist = 0.0
            for q_j in quats:
                total_dist += self._angular_distance(q_i, q_j)
            if total_dist < best_score:
                best_score = total_dist
                best_quat = q_i

        return best_quat

    def _reject_outlier_estimates(self, estimates):
        """
        Reject tag-derived cube poses that deviate too much in translation
        OR rotation from the group consensus.

        A pose is kept only if BOTH conditions are satisfied:
            - Translation distance to median <= outlier_trans_threshold
            - Angular distance to best consensus quaternion <= outlier_rot_threshold_rad
        """
        if len(estimates) < self.min_tags_for_outlier_rejection:
            return estimates

        # === Translation: use median (robust to outliers) ===
        positions = np.array([T[:3, 3] for T in estimates])
        median_pos = np.median(positions, axis=0)

        # === Rotation: use best consensus reference ===
        quats = [tfs.quaternion_from_matrix(T) for T in estimates]
        ref_quat = self._choose_best_rotation_reference(quats)

        inliers = []
        for T in estimates:
            trans_dist = np.linalg.norm(T[:3, 3] - median_pos)
            q = tfs.quaternion_from_matrix(T)
            ang_dist = self._angular_distance(q, ref_quat)

            if (trans_dist <= self.outlier_trans_threshold and
                ang_dist <= self.outlier_rot_threshold_rad):
                inliers.append(T)

        rejected_count = len(estimates) - len(inliers)
        if rejected_count > 0:
            rospy.loginfo_throttle(
                1.5,
                f"Rejected {rejected_count} outlier tag(s) | "
                f"trans_thresh={self.outlier_trans_threshold:.3f}m, "
                f"rot_thresh={self.outlier_rot_threshold_rad:.2f}rad"
            )

        if len(inliers) == 0:
            rospy.logwarn_throttle(1.0, "All tags rejected as outliers — using full set")
            return estimates

        return inliers

    def _fuse_estimates(self, estimates):
        """
        Fuse multiple tag-derived cube poses into one.
        Uses mean position + sign-aligned quaternion average.
        """
        if len(estimates) == 1:
            return estimates[0].copy()

        positions = np.array([T[:3, 3] for T in estimates])
        avg_pos = np.mean(positions, axis=0)

        quats = []
        ref_quat = tfs.quaternion_from_matrix(estimates[0])
        for T in estimates:
            q = tfs.quaternion_from_matrix(T)
            if np.dot(q, ref_quat) < 0.0:
                q = -q
            quats.append(q)

        avg_quat = np.mean(quats, axis=0)
        avg_quat = self._normalize_quaternion(avg_quat)

        T_avg = tfs.quaternion_matrix(avg_quat)
        T_avg[:3, 3] = avg_pos
        return T_avg

    # =========================================================================
    # MAIN CALLBACK
    # =========================================================================

    def detections_callback(self, msg):
        if not msg.detections:
            return

        current_time = msg.header.stamp

        # Auto-detect camera frame from first message
        if self.camera_frame is None and msg.detections:
            self.camera_frame = msg.detections[0].pose.header.frame_id
            rospy.loginfo("Locked camera_frame to: %s", self.camera_frame)

        # Avoid duplicate processing
        if self.last_time is not None and current_time <= self.last_time:
            return

        # === Step 1: Collect cube pose estimates from every visible tag ===
        estimates = []
        for detection in msg.detections:
            tag_id = detection.id[0] if detection.id else None
            if tag_id not in self.T_cube_to_tag:
                continue

            pose = detection.pose.pose.pose
            p = [pose.position.x, pose.position.y, pose.position.z]
            q = [pose.orientation.x, pose.orientation.y,
                 pose.orientation.z, pose.orientation.w]

            T_cam_tag = tfs.quaternion_matrix(q)
            T_cam_tag[:3, 3] = p

            T_cube_tag = self.T_cube_to_tag[tag_id]
            T_cam_cube = T_cam_tag @ np.linalg.inv(T_cube_tag)
            estimates.append(T_cam_cube)

        if not estimates:
            return

        # === Step 2: Outlier rejection (if enabled) ===
        if ENABLE_OUTLIER_REJECTION:
            estimates = self._reject_outlier_estimates(estimates)

        # === Step 3: Minimum tags gate ===
        if len(estimates) < self.min_tags_to_publish:
            rospy.loginfo_throttle(
                2.0,
                f"Only {len(estimates)} valid cube tag(s) visible — "
                f"need at least {self.min_tags_to_publish} to publish odometry"
            )
            return

        # === Step 4: Fuse remaining estimates ===
        current_pose_matrix = self._fuse_estimates(estimates)

        # === Step 5: Build and publish Odometry ===
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = self.camera_frame if self.camera_frame else "camera"
        odom.child_frame_id = self.bundle_frame

        p = current_pose_matrix[:3, 3]
        q = tfs.quaternion_from_matrix(current_pose_matrix)

        odom.pose.pose.position = Point(*p)
        odom.pose.pose.orientation = Quaternion(*q)

        # === Velocity calculation (camera frame) ===
        if self.last_pose_matrix is not None:
            dt = (current_time - self.last_time).to_sec()
            if dt > 0:
                v_cam = (p - self.last_pose_matrix[:3, 3]) / dt

                R1 = self.last_pose_matrix[:3, :3]
                R2 = current_pose_matrix[:3, :3]
                R_rel = R2 @ R1.T

                R_rel_44 = np.eye(4)
                R_rel_44[:3, :3] = R_rel

                try:
                    angle, axis, _ = tfs.rotation_from_matrix(R_rel_44)
                    omega_cam = (angle / dt) * np.array(axis)
                except ValueError:
                    omega_cam = np.zeros(3)

                odom.twist.twist.linear = Vector3(*v_cam)
                odom.twist.twist.angular = Vector3(*omega_cam)

                self.pub.publish(odom)

        # Update state for next iteration
        self.last_time = current_time
        self.last_pose_matrix = current_pose_matrix


if __name__ == "__main__":
    try:
        rospy.init_node('cube_bundle_publisher', anonymous=True)
        node = CubeBundlePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass