#!/usr/bin/env python
"""Log cube pose from /obj_odometry to CSV for offline smoothness analysis."""

import csv
import math
import os
from datetime import datetime

import rospy
import tf.transformations as tfs
from nav_msgs.msg import Odometry


class CubePoseLogger(object):
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/obj_odometry")
        self.sample_print_every_n = rospy.get_param("~sample_print_every_n", 30)

        csv_path_param = rospy.get_param("~csv_path", "")
        self.csv_path = self._resolve_csv_path(csv_path_param)
        self._ensure_parent_dir(self.csv_path)

        self.csv_file = open(self.csv_path, "w")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "stamp",
            "x", "y", "z",
            "qx", "qy", "qz", "qw",
            "roll_deg", "pitch_deg", "yaw_deg",
            "vx", "vy", "vz",
            "wx", "wy", "wz",
        ])

        self.samples = 0
        self.last_stamp = None
        rospy.on_shutdown(self._shutdown)

        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=10)

        rospy.loginfo("cube_pose_logger initialized")
        rospy.loginfo("subscribing to: %s", self.odom_topic)
        rospy.loginfo("logging csv: %s", self.csv_path)

    def _resolve_csv_path(self, csv_path_param):
        if csv_path_param:
            return os.path.expanduser(csv_path_param)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return "/tmp/cube_pose_%s.csv" % stamp

    def _ensure_parent_dir(self, path):
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)

    def odom_callback(self, msg):
        stamp = msg.header.stamp
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return
        self.last_stamp = stamp

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        roll, pitch, yaw = tfs.euler_from_quaternion(quat)

        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular

        self.csv_writer.writerow([
            "%.9f" % stamp.to_sec(),
            "%.9f" % p.x, "%.9f" % p.y, "%.9f" % p.z,
            "%.9f" % q.x, "%.9f" % q.y, "%.9f" % q.z, "%.9f" % q.w,
            "%.6f" % math.degrees(roll),
            "%.6f" % math.degrees(pitch),
            "%.6f" % math.degrees(yaw),
            "%.9f" % lin.x, "%.9f" % lin.y, "%.9f" % lin.z,
            "%.9f" % ang.x, "%.9f" % ang.y, "%.9f" % ang.z,
        ])
        self.csv_file.flush()

        self.samples += 1
        if self.samples % self.sample_print_every_n == 0:
            rospy.loginfo(
                "samples=%d pos=(%.4f, %.4f, %.4f) rpy_deg=(%.2f, %.2f, %.2f)",
                self.samples, p.x, p.y, p.z,
                math.degrees(roll), math.degrees(pitch), math.degrees(yaw),
            )

    def _shutdown(self):
        if hasattr(self, "csv_file") and self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
            rospy.loginfo("cube_pose_logger closed csv after %d samples -> %s",
                          self.samples, self.csv_path)


if __name__ == "__main__":
    try:
        rospy.init_node("cube_pose_logger", anonymous=False)
        CubePoseLogger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
