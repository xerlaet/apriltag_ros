#!/usr/bin/env python

import rospy
import numpy as np
import tf.transformations as tfs
from geometry_msgs.msg import Vector3, Point, Quaternion
from nav_msgs.msg import Odometry
from apriltag_ros.msg import AprilTagDetectionArray
import tf2_ros

ENABLE_KALMAN_FILTER = False

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

        # Camera-frame Kalman filter for translational cube motion.
        self.filter_initialized = False
        self.filter_state = np.zeros(9)  # [x, y, z, vx, vy, vz, ax, ay, az]
        self.filter_covariance = np.eye(9)
        self.filter_last_time = None
        self.filtered_acceleration = np.zeros(3)
        self.last_motion_gate_distance = 0.0
        self.last_position_jump = 0.0
        self.last_motion_measurement_accepted = True
        self.rejected_motion_updates = 0

        # AprilTag tracking defaults: trust position most, velocity less.
        # Higher values let the filter follow faster acceleration changes; lower values smooth motion more.
        self.process_jerk_std = rospy.get_param('~process_jerk_std', 0.5)
        # Expected AprilTag position noise in meters; higher values trust measured positions less.
        self.position_measurement_std = rospy.get_param('~position_measurement_std', 0.01)
        # Expected finite-difference velocity noise in m/s; higher values smooth velocity more.
        self.velocity_measurement_std = rospy.get_param('~velocity_measurement_std', 0.5)
        # Initial uncertainty for cube position in meters when the filter first starts.
        self.initial_position_std = rospy.get_param('~initial_position_std', 0.01)
        # Initial uncertainty for cube velocity in m/s when the filter first starts.
        self.initial_velocity_std = rospy.get_param('~initial_velocity_std', 0.10)
        # Initial uncertainty for inferred cube acceleration in m/s^2.
        self.initial_acceleration_std = rospy.get_param('~initial_acceleration_std', 1.0)
        # Maximum allowed Mahalanobis distance before a measurement is treated as an outlier.
        self.mahalanobis_gate_threshold = rospy.get_param('~mahalanobis_gate_threshold', 16.0)
        # Maximum allowed measured position jump in meters before rejecting the update.
        self.position_jump_gate_m = rospy.get_param('~position_jump_gate_m', 0.02)
        # Reinitialize to a new stable pose after this many rejected motion updates.
        self.max_rejected_motion_updates = rospy.get_param('~max_rejected_motion_updates', 5)
        # Velocity multiplier applied when rejecting motion updates to prevent coasting forever.
        self.rejected_velocity_decay = rospy.get_param('~rejected_velocity_decay', 0.2)
        # Fallback timestep in seconds when timestamps are invalid or unavailable.
        self.default_dt = rospy.get_param('~default_dt', 1.0 / 30.0)
        # Largest timestep used by the filter after dropped detections to avoid huge predictions.
        self.max_filter_dt = rospy.get_param('~max_filter_dt', 0.2)
        self.measurement_matrix = np.zeros((6, 9))
        self.measurement_matrix[:3, :3] = np.eye(3)
        self.measurement_matrix[3:, 3:6] = np.eye(3)
        self.measurement_covariance = np.diag(
            [self.position_measurement_std ** 2] * 3 +
            [self.velocity_measurement_std ** 2] * 3
        )

        # Quaternion-aware Kalman filter for cube orientation and angular velocity.
        self.orientation_filter_initialized = False
        self.orientation_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.orientation_angular_velocity = np.zeros(3)
        self.orientation_covariance = np.eye(6)
        self.orientation_last_time = None
        self.last_orientation_jump = 0.0
        self.last_orientation_measurement_accepted = True
        self.rejected_orientation_updates = 0
        # Expected AprilTag orientation noise in radians; higher values trust measured orientation less.
        self.orientation_measurement_std_rad = rospy.get_param('~orientation_measurement_std_rad', 0.10)
        # Expected finite-difference angular velocity noise in rad/s; higher values smooth angular velocity more.
        self.angular_velocity_measurement_std = rospy.get_param('~angular_velocity_measurement_std', 1.0)
        # Orientation process noise in radians; higher values let filtered orientation drift/change faster.
        self.orientation_process_std_rad = rospy.get_param('~orientation_process_std_rad', 0.03)
        # Angular velocity process noise in rad/s; higher values allow faster angular velocity changes.
        self.angular_velocity_process_std = rospy.get_param('~angular_velocity_process_std', 1.5)
        # Initial uncertainty for cube orientation in radians when the orientation filter starts.
        self.initial_orientation_std_rad = rospy.get_param('~initial_orientation_std_rad', 0.05)
        # Initial uncertainty for cube angular velocity in rad/s when the orientation filter starts.
        self.initial_angular_velocity_std = rospy.get_param('~initial_angular_velocity_std', 0.5)
        # Maximum allowed measured orientation jump in radians before rejecting the update.
        self.orientation_jump_gate_rad = rospy.get_param('~orientation_jump_gate_rad', 0.35)
        # Reinitialize to a new stable orientation after this many rejected orientation updates.
        self.max_rejected_orientation_updates = rospy.get_param('~max_rejected_orientation_updates', 5)
        # Angular velocity multiplier applied on rejected orientation updates to stop drift.
        self.rejected_angular_velocity_decay = rospy.get_param('~rejected_angular_velocity_decay', 0.2)
        self.orientation_measurement_covariance = np.diag(
            [self.orientation_measurement_std_rad ** 2] * 3 +
            [self.angular_velocity_measurement_std ** 2] * 3
        )

        rospy.loginfo("Cube Bundle Odometry Publisher initialized.")
        rospy.loginfo("Syncing with detections on /tag_detections...")

        # We subscribe to detections to trigger the TF lookup at the exact moment of detection
        self.sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.detections_callback, queue_size=1)

    def _sanitize_dt(self, dt):
        if dt <= 0.0 or not np.isfinite(dt):
            return self.default_dt
        return min(dt, self.max_filter_dt)

    def _build_motion_transition(self, dt):
        F = np.eye(9)
        F[:3, 3:6] = dt * np.eye(3)
        F[:3, 6:9] = 0.5 * dt * dt * np.eye(3)
        F[3:6, 6:9] = dt * np.eye(3)
        return F

    def _compute_process_noise(self, dt):
        # White jerk noise integrated through a constant-acceleration model.
        G_1d = np.array([dt ** 3 / 6.0, dt ** 2 / 2.0, dt])
        Q_1d = (self.process_jerk_std ** 2) * np.outer(G_1d, G_1d)

        Q = np.zeros((9, 9))
        for axis in range(3):
            idx = [axis, axis + 3, axis + 6]
            Q[np.ix_(idx, idx)] = Q_1d
        return Q

    def _initialize_motion_filter(self, position, velocity, stamp):
        self.filter_state[:] = 0.0
        self.filter_state[:3] = position
        self.filter_state[3:6] = velocity

        self.filter_covariance = np.diag(
            [self.initial_position_std ** 2] * 3 +
            [self.initial_velocity_std ** 2] * 3 +
            [self.initial_acceleration_std ** 2] * 3
        )
        self.filter_last_time = stamp
        self.filtered_acceleration = self.filter_state[6:9].copy()
        self.filter_initialized = True
        self.last_motion_measurement_accepted = True
        self.last_motion_gate_distance = 0.0
        self.rejected_motion_updates = 0

    def _passes_motion_gate(self, measurement, predicted_measurement, innovation_covariance):
        innovation = measurement - predicted_measurement
        try:
            normalized_innovation = np.linalg.solve(innovation_covariance, innovation)
        except np.linalg.LinAlgError:
            return True

        gate_distance = float(np.dot(innovation, normalized_innovation))
        self.last_motion_gate_distance = gate_distance
        return gate_distance <= self.mahalanobis_gate_threshold

    def _passes_position_jump_gate(self, position):
        self.last_position_jump = float(np.linalg.norm(position - self.filter_state[:3]))
        return self.last_position_jump <= self.position_jump_gate_m

    def _hold_motion_filter(self, stamp, covariance):
        self.filter_state[3:6] *= self.rejected_velocity_decay
        self.filter_state[6:9] = 0.0
        self.filter_covariance = covariance
        self.filter_last_time = stamp
        self.filtered_acceleration = self.filter_state[6:9].copy()
        self.last_motion_measurement_accepted = False
        return self.filter_state[:3].copy(), self.filter_state[3:6].copy()

    def _filter_motion(self, position, velocity, stamp):
        if self.filter_last_time is None:
            dt = self.default_dt
        else:
            dt = self._sanitize_dt((stamp - self.filter_last_time).to_sec())

        F = self._build_motion_transition(dt)
        Q = self._compute_process_noise(dt)
        H = self.measurement_matrix
        R = self.measurement_covariance

        predicted_state = F.dot(self.filter_state)
        predicted_covariance = F.dot(self.filter_covariance).dot(F.T) + Q

        if not self._passes_position_jump_gate(position):
            self.rejected_motion_updates += 1
            if self.rejected_motion_updates >= self.max_rejected_motion_updates:
                self._initialize_motion_filter(position, np.zeros(3), stamp)
                return self.filter_state[:3].copy(), self.filter_state[3:6].copy()

            rospy.logwarn_throttle(
                1.0,
                "Rejected cube position jump: %.4f m > %.4f m; holding pose",
                self.last_position_jump,
                self.position_jump_gate_m
            )
            return self._hold_motion_filter(stamp, predicted_covariance)

        measurement = np.concatenate((position, velocity))
        predicted_measurement = H.dot(predicted_state)
        innovation = measurement - predicted_measurement
        innovation_covariance = H.dot(predicted_covariance).dot(H.T) + R

        if self._passes_motion_gate(measurement, predicted_measurement, innovation_covariance):
            try:
                kalman_gain = np.linalg.solve(
                    innovation_covariance,
                    H.dot(predicted_covariance)
                ).T
            except np.linalg.LinAlgError:
                kalman_gain = predicted_covariance.dot(H.T).dot(np.linalg.pinv(innovation_covariance))

            self.filter_state = predicted_state + kalman_gain.dot(innovation)
            I = np.eye(9)
            covariance_update = I - kalman_gain.dot(H)
            self.filter_covariance = (
                covariance_update.dot(predicted_covariance).dot(covariance_update.T) +
                kalman_gain.dot(R).dot(kalman_gain.T)
            )
            self.last_motion_measurement_accepted = True
            self.rejected_motion_updates = 0
        else:
            self.rejected_motion_updates += 1
            if self.rejected_motion_updates >= self.max_rejected_motion_updates:
                self._initialize_motion_filter(position, np.zeros(3), stamp)
                return self.filter_state[:3].copy(), self.filter_state[3:6].copy()

            rospy.logwarn_throttle(
                1.0,
                "Rejected cube motion outlier, Mahalanobis distance: %.2f; holding pose",
                self.last_motion_gate_distance
            )
            return self._hold_motion_filter(stamp, predicted_covariance)

        self.filter_last_time = stamp
        self.filtered_acceleration = self.filter_state[6:9].copy()
        return self.filter_state[:3].copy(), self.filter_state[3:6].copy()

    def _normalize_quaternion(self, quat):
        quat = np.array(quat, dtype=float)
        norm = np.linalg.norm(quat)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return quat / norm

    def _align_quaternion_to_reference(self, quat, reference):
        quat = self._normalize_quaternion(quat)
        if np.dot(quat, reference) < 0.0:
            quat = -quat
        return quat

    def _quaternion_inverse(self, quat):
        return self._normalize_quaternion(tfs.quaternion_inverse(self._normalize_quaternion(quat)))

    def _rotation_vector_to_quaternion(self, rotation_vector):
        angle = np.linalg.norm(rotation_vector)
        if angle < 1e-12:
            quat = np.array([
                0.5 * rotation_vector[0],
                0.5 * rotation_vector[1],
                0.5 * rotation_vector[2],
                1.0
            ])
            return self._normalize_quaternion(quat)

        axis = rotation_vector / angle
        half_angle = 0.5 * angle
        quat = np.array([
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle),
            np.cos(half_angle)
        ])
        return self._normalize_quaternion(quat)

    def _quaternion_to_rotation_vector(self, quat):
        quat = self._normalize_quaternion(quat)
        if quat[3] < 0.0:
            quat = -quat

        vector_norm = np.linalg.norm(quat[:3])
        if vector_norm < 1e-12:
            return 2.0 * quat[:3]

        angle = 2.0 * np.arctan2(vector_norm, quat[3])
        return (quat[:3] / vector_norm) * angle

    def _integrate_orientation(self, quat, angular_velocity, dt):
        delta_quat = self._rotation_vector_to_quaternion(angular_velocity * dt)
        return self._normalize_quaternion(tfs.quaternion_multiply(delta_quat, quat))

    def _initialize_orientation_filter(self, measured_quat, measured_angular_velocity, stamp):
        self.orientation_quat = self._normalize_quaternion(measured_quat)
        self.orientation_angular_velocity = np.array(measured_angular_velocity, dtype=float)
        self.orientation_covariance = np.diag(
            [self.initial_orientation_std_rad ** 2] * 3 +
            [self.initial_angular_velocity_std ** 2] * 3
        )
        self.orientation_last_time = stamp
        self.orientation_filter_initialized = True
        self.last_orientation_measurement_accepted = True
        self.last_orientation_jump = 0.0
        self.rejected_orientation_updates = 0

    def _compute_orientation_process_noise(self, dt):
        return np.diag(
            [self.orientation_process_std_rad ** 2 * dt] * 3 +
            [self.angular_velocity_process_std ** 2 * dt] * 3
        )

    def _hold_orientation_filter(self, stamp, covariance):
        self.orientation_angular_velocity *= self.rejected_angular_velocity_decay
        self.orientation_covariance = covariance
        self.orientation_last_time = stamp
        self.last_orientation_measurement_accepted = False
        return self.orientation_quat.copy(), self.orientation_angular_velocity.copy()

    def _filter_orientation(self, measured_quat, measured_angular_velocity, stamp):
        measured_quat = self._align_quaternion_to_reference(measured_quat, self.orientation_quat)
        measured_angular_velocity = np.array(measured_angular_velocity, dtype=float)

        if self.orientation_last_time is None:
            dt = self.default_dt
        else:
            dt = self._sanitize_dt((stamp - self.orientation_last_time).to_sec())

        F = np.eye(6)
        F[:3, 3:6] = dt * np.eye(3)
        Q = self._compute_orientation_process_noise(dt)
        R = self.orientation_measurement_covariance

        predicted_quat = self._integrate_orientation(
            self.orientation_quat,
            self.orientation_angular_velocity,
            dt
        )
        predicted_angular_velocity = self.orientation_angular_velocity.copy()
        predicted_covariance = F.dot(self.orientation_covariance).dot(F.T) + Q

        measured_quat = self._align_quaternion_to_reference(measured_quat, predicted_quat)
        quat_error = tfs.quaternion_multiply(measured_quat, self._quaternion_inverse(predicted_quat))
        orientation_error = self._quaternion_to_rotation_vector(quat_error)
        self.last_orientation_jump = float(np.linalg.norm(orientation_error))

        if self.last_orientation_jump > self.orientation_jump_gate_rad:
            self.rejected_orientation_updates += 1
            if self.rejected_orientation_updates >= self.max_rejected_orientation_updates:
                self._initialize_orientation_filter(measured_quat, np.zeros(3), stamp)
                return self.orientation_quat.copy(), self.orientation_angular_velocity.copy()

            rospy.logwarn_throttle(
                1.0,
                "Rejected cube orientation jump: %.4f rad > %.4f rad; holding orientation",
                self.last_orientation_jump,
                self.orientation_jump_gate_rad
            )
            return self._hold_orientation_filter(stamp, predicted_covariance)

        innovation = np.concatenate((
            orientation_error,
            measured_angular_velocity - predicted_angular_velocity
        ))
        innovation_covariance = predicted_covariance + R

        try:
            kalman_gain = np.linalg.solve(innovation_covariance, predicted_covariance).T
        except np.linalg.LinAlgError:
            kalman_gain = predicted_covariance.dot(np.linalg.pinv(innovation_covariance))

        correction = kalman_gain.dot(innovation)
        correction_quat = self._rotation_vector_to_quaternion(correction[:3])
        self.orientation_quat = self._normalize_quaternion(
            tfs.quaternion_multiply(correction_quat, predicted_quat)
        )
        self.orientation_angular_velocity = predicted_angular_velocity + correction[3:6]

        I = np.eye(6)
        covariance_update = I - kalman_gain
        self.orientation_covariance = (
            covariance_update.dot(predicted_covariance).dot(covariance_update.T) +
            kalman_gain.dot(R).dot(kalman_gain.T)
        )
        self.orientation_last_time = stamp
        self.last_orientation_measurement_accepted = True
        self.rejected_orientation_updates = 0
        return self.orientation_quat.copy(), self.orientation_angular_velocity.copy()

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
        measured_quat = np.array([q.x, q.y, q.z, q.w])
        
        # Calculate velocities if we have a previous pose
        if self.last_pose_matrix is not None:
            dt = (current_time - self.last_time).to_sec()
            if dt > 0:
                position_cam = current_pose_matrix[:3, 3]

                # Linear velocity in camera frame
                v_cam = (position_cam - self.last_pose_matrix[:3, 3]) / dt
                
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
                
                if not ENABLE_KALMAN_FILTER:
                    filtered_position = position_cam
                    filtered_velocity = v_cam
                    filtered_quat = self._normalize_quaternion(measured_quat)
                    filtered_omega = omega_cam
                else:
                    if not self.filter_initialized:
                        self._initialize_motion_filter(position_cam, v_cam, current_time)
                        filtered_position = position_cam
                        filtered_velocity = v_cam
                    else:
                        filtered_position, filtered_velocity = self._filter_motion(position_cam, v_cam, current_time)

                    if not self.orientation_filter_initialized:
                        self._initialize_orientation_filter(measured_quat, omega_cam, current_time)
                        filtered_quat = self.orientation_quat.copy()
                        filtered_omega = omega_cam
                    else:
                        filtered_quat, filtered_omega = self._filter_orientation(
                            measured_quat,
                            omega_cam,
                            current_time
                        )
                
                odom.pose.pose.position = Point(*filtered_position)
                odom.pose.pose.orientation = Quaternion(*filtered_quat)
                odom.twist.twist.linear = Vector3(*filtered_velocity)
                odom.twist.twist.angular = Vector3(*filtered_omega)
                
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
