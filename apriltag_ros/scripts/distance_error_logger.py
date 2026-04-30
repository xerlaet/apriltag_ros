#!/usr/bin/env python

import csv
import math
import os
from datetime import datetime

import rospy
import tf2_ros
from visualization_msgs.msg import Marker


class DistanceErrorLogger(object):
    def __init__(self):
        self.camera_frame = rospy.get_param("~camera_frame", "rs_camera_infra1_optical_frame")
        self.cube_frame = rospy.get_param("~cube_frame", "cube")
        self.tag_frame = rospy.get_param("~tag_frame", "tag_frame")
        self.sample_rate_hz = rospy.get_param("~sample_rate_hz", 15.0)
        self.lookup_timeout_s = rospy.get_param("~lookup_timeout_s", 0.05)
        self.stats_print_every_n = rospy.get_param("~stats_print_every_n", 30)
        self.placement_print_every_n = rospy.get_param("~placement_print_every_n", 10)
        self.distance_tolerance_m = rospy.get_param("~distance_tolerance_m", 0.02)
        self.orientation_tolerance_deg = rospy.get_param("~orientation_tolerance_deg", 10.0)
        self.status_marker_topic = rospy.get_param("~status_marker_topic", "/distance_error_status")
        self.max_transform_age_s = rospy.get_param("~max_transform_age_s", 0.2)
        self.tag_min_delta_m = rospy.get_param("~tag_min_delta_m", 0.0005)
        self.max_tag_static_samples = rospy.get_param("~max_tag_static_samples", 5)
        self.log_only_when_ready = rospy.get_param("~log_only_when_ready", False)

        # Physical setup defaults from your described paper experiment.
        self.tag_width_in = rospy.get_param("~tag_width_in", 2.08)
        self.edge_gap_in = rospy.get_param("~edge_gap_in", 4.0)
        self.cube_size_cm = rospy.get_param("~cube_size_cm", 7.0)
        self.cube_center_height_cm = rospy.get_param("~cube_center_height_cm", 3.5)

        self.known_distance_m = self._compute_known_distance_m()

        csv_path_param = rospy.get_param("~csv_path", "")
        self.csv_path = self._resolve_csv_path(csv_path_param)
        self._ensure_parent_dir(self.csv_path)
        self.csv_file = open(self.csv_path, "w")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "stamp",
            "cube_x",
            "cube_y",
            "cube_z",
            "tag_x",
            "tag_y",
            "tag_z",
            "measured_dist_m",
            "known_dist_m",
            "signed_error_m",
            "abs_error_m",
            "sq_error_m",
        ])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.status_pub = rospy.Publisher(self.status_marker_topic, Marker, queue_size=1)

        self.samples = 0
        self.skipped_samples = 0
        self.tf_failures = 0
        self.sum_signed_error = 0.0
        self.sum_abs_error = 0.0
        self.sum_sq_error = 0.0
        self.last_tag_xyz = None
        self.tag_static_count = 0

        rospy.on_shutdown(self._shutdown)

        rospy.loginfo("distance_error_logger initialized")
        rospy.loginfo("camera_frame=%s cube_frame=%s tag_frame=%s",
                      self.camera_frame, self.cube_frame, self.tag_frame)
        rospy.loginfo("known distance (m): %.6f", self.known_distance_m)
        rospy.loginfo("logging csv: %s", self.csv_path)
        rospy.loginfo("placement tolerances: distance<=%.3f m orientation<=%.1f deg",
                      self.distance_tolerance_m, self.orientation_tolerance_deg)
        rospy.loginfo("validity gates: max_age<=%.3fs tag_min_delta>=%.6fm static_limit=%d ready_only=%s",
                      self.max_transform_age_s, self.tag_min_delta_m, self.max_tag_static_samples, str(self.log_only_when_ready))

    def _compute_known_distance_m(self):
        in_to_m = 0.0254
        cm_to_m = 0.01
        tag_width_m = self.tag_width_in * in_to_m
        edge_gap_m = self.edge_gap_in * in_to_m
        cube_size_m = self.cube_size_cm * cm_to_m
        cube_center_height_m = self.cube_center_height_cm * cm_to_m

        horizontal_center_to_center_m = edge_gap_m + 0.5 * tag_width_m + 0.5 * cube_size_m
        return math.sqrt(horizontal_center_to_center_m ** 2 + cube_center_height_m ** 2)

    def _resolve_csv_path(self, csv_path_param):
        if csv_path_param:
            return os.path.expanduser(csv_path_param)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return "/tmp/apriltag_distance_error_%s.csv" % stamp

    def _ensure_parent_dir(self, path):
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)

    def _lookup_transform(self, source_frame, target_frame):
        transform = self.tf_buffer.lookup_transform(
            source_frame,
            target_frame,
            rospy.Time(0),
            rospy.Duration(self.lookup_timeout_s),
        )
        return transform

    def _quat_angle_deg(self, qa, qb):
        dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]
        dot = max(min(abs(dot), 1.0), 0.0)
        return math.degrees(2.0 * math.acos(dot))

    def _norm3(self, a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _publish_status_marker(self, stamp, cube_xyz, ready, distance_error_m, orient_error_deg):
        marker = Marker()
        marker.header.frame_id = self.camera_frame
        marker.header.stamp = stamp
        marker.ns = "distance_error_logger"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = cube_xyz[0]
        marker.pose.position.y = cube_xyz[1]
        marker.pose.position.z = cube_xyz[2] + 0.08
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.05

        if ready:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.text = "READY d=%.3fm o=%.1fdeg" % (distance_error_m, orient_error_deg)
        else:
            marker.color.r = 1.0
            marker.color.g = 0.3
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.text = "ADJUST d=%.3fm o=%.1fdeg" % (distance_error_m, orient_error_deg)

        self.status_pub.publish(marker)

    def run(self):
        rate = rospy.Rate(self.sample_rate_hz)
        while not rospy.is_shutdown():
            try:
                cube_tf = self._lookup_transform(self.camera_frame, self.cube_frame)
                tag_tf = self._lookup_transform(self.camera_frame, self.tag_frame)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.tf_failures += 1
                rate.sleep()
                continue

            cube_t = cube_tf.transform.translation
            tag_t = tag_tf.transform.translation
            cube_xyz = (cube_t.x, cube_t.y, cube_t.z)
            tag_xyz = (tag_t.x, tag_t.y, tag_t.z)

            dx = cube_xyz[0] - tag_xyz[0]
            dy = cube_xyz[1] - tag_xyz[1]
            dz = cube_xyz[2] - tag_xyz[2]
            measured_dist_m = math.sqrt(dx * dx + dy * dy + dz * dz)
            signed_error_m = measured_dist_m - self.known_distance_m
            abs_error_m = abs(signed_error_m)
            sq_error_m = signed_error_m * signed_error_m

            cube_q = cube_tf.transform.rotation
            tag_q = tag_tf.transform.rotation
            orient_error_deg = self._quat_angle_deg(
                (cube_q.x, cube_q.y, cube_q.z, cube_q.w),
                (tag_q.x, tag_q.y, tag_q.z, tag_q.w),
            )
            distance_error_m = abs_error_m
            ready = (distance_error_m <= self.distance_tolerance_m and
                     orient_error_deg <= self.orientation_tolerance_deg)

            stamp_cube = cube_tf.header.stamp
            stamp_tag = tag_tf.header.stamp
            stamp = stamp_cube if stamp_cube >= stamp_tag else stamp_tag
            stamp_s = stamp.to_sec()
            now_s = rospy.Time.now().to_sec()

            cube_age_s = now_s - stamp_cube.to_sec()
            tag_age_s = now_s - stamp_tag.to_sec()
            if cube_age_s > self.max_transform_age_s or tag_age_s > self.max_transform_age_s:
                self.skipped_samples += 1
                rate.sleep()
                continue

            if self.last_tag_xyz is not None:
                tag_delta_m = self._norm3(tag_xyz, self.last_tag_xyz)
                if tag_delta_m < self.tag_min_delta_m:
                    self.tag_static_count += 1
                else:
                    self.tag_static_count = 0
            self.last_tag_xyz = tag_xyz

            if self.tag_static_count > self.max_tag_static_samples:
                self.skipped_samples += 1
                rate.sleep()
                continue

            if self.log_only_when_ready and not ready:
                self.skipped_samples += 1
                rate.sleep()
                continue

            self.csv_writer.writerow([
                "%.9f" % stamp_s,
                "%.9f" % cube_xyz[0], "%.9f" % cube_xyz[1], "%.9f" % cube_xyz[2],
                "%.9f" % tag_xyz[0], "%.9f" % tag_xyz[1], "%.9f" % tag_xyz[2],
                "%.9f" % measured_dist_m,
                "%.9f" % self.known_distance_m,
                "%.9f" % signed_error_m,
                "%.9f" % abs_error_m,
                "%.9f" % sq_error_m,
            ])
            self.csv_file.flush()

            self.samples += 1
            self.sum_signed_error += signed_error_m
            self.sum_abs_error += abs_error_m
            self.sum_sq_error += sq_error_m

            if self.samples % self.stats_print_every_n == 0:
                mean_signed_error = self.sum_signed_error / float(self.samples)
                mae = self.sum_abs_error / float(self.samples)
                rmse = math.sqrt(self.sum_sq_error / float(self.samples))
                rospy.loginfo(
                    "samples=%d skipped=%d tf_failures=%d mean_signed=%.6f m mae=%.6f m rmse=%.6f m",
                    self.samples, self.skipped_samples, self.tf_failures, mean_signed_error, mae, rmse
                )

            if self.samples % self.placement_print_every_n == 0:
                rospy.loginfo(
                    "placement %s dist_meas=%.4f dist_target=%.4f dist_abs_err=%.4f orient_err=%.2f deg",
                    "READY" if ready else "ADJUST",
                    measured_dist_m,
                    self.known_distance_m,
                    distance_error_m,
                    orient_error_deg,
                )

            self._publish_status_marker(stamp, cube_xyz, ready, distance_error_m, orient_error_deg)

            rate.sleep()

    def _shutdown(self):
        if hasattr(self, "csv_file") and self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
            rospy.loginfo("distance_error_logger closed csv after %d samples (%d skipped, %d tf failures)",
                          self.samples, self.skipped_samples, self.tf_failures)


if __name__ == "__main__":
    try:
        rospy.init_node("distance_error_logger", anonymous=False)
        node = DistanceErrorLogger()
        node.run()
    except rospy.ROSInterruptException:
        pass
