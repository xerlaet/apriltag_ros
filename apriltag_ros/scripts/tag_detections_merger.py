#!/usr/bin/env python
"""
Merge AprilTag detections from two cameras into the primary (D435) optical frame.

Cube tags (IDs 1-24) from the secondary camera are transformed using a startup-locked
T_cam1_cam2 computed from tag 0 visible in both cameras. Duplicate tag IDs keep the
detection with the higher decision_margin.
"""

import copy

import numpy as np
import rospy
import tf.transformations as tfs
from apriltag_ros.msg import AprilTagDetectionArray

BASE_TAG_ID = 0
CUBE_TAG_IDS = set(range(1, 25))


class TagDetectionsMerger:
    def __init__(self):
        self.cam1_topic = rospy.get_param("~cam1_topic", "/tag_detections")
        self.cam2_topic = rospy.get_param("~cam2_topic", "/tag_detections_cam2")
        self.output_topic = rospy.get_param("~output_topic", "/tag_detections_merged")
        self.output_frame = rospy.get_param(
            "~output_frame", "rs_camera_infra1_optical_frame"
        )
        self.cam2_max_age_sec = rospy.get_param("~cam2_max_age_sec", 0.05)
        self.min_decision_margin = rospy.get_param("~min_decision_margin", 50.0)

        self.T_cam1_cam2 = None
        self.cam2_msg = None
        self.cam2_stamp = None
        self.extrinsic_locked = False

        self.pub = rospy.Publisher(self.output_topic, AprilTagDetectionArray, queue_size=1)
        rospy.Subscriber(self.cam2_topic, AprilTagDetectionArray, self.cam2_callback, queue_size=1)
        rospy.Subscriber(self.cam1_topic, AprilTagDetectionArray, self.cam1_callback, queue_size=1)

        rospy.loginfo("TagDetectionsMerger initialized.")
        rospy.loginfo("  cam1=%s cam2=%s output=%s", self.cam1_topic, self.cam2_topic, self.output_topic)
        rospy.loginfo("  output_frame=%s min_decision_margin=%.1f", self.output_frame, self.min_decision_margin)

    @staticmethod
    def _pose_to_matrix(pose):
        q = pose.orientation
        T = tfs.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return T

    @staticmethod
    def _matrix_to_pose(T, pose):
        pose.position.x = T[0, 3]
        pose.position.y = T[1, 3]
        pose.position.z = T[2, 3]
        q = tfs.quaternion_from_matrix(T)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

    @staticmethod
    def _get_tag_detection(msg, tag_id):
        for det in msg.detections:
            if det.id and det.id[0] == tag_id:
                return det
        return None

    def _try_lock_extrinsic(self, cam1_msg, cam2_msg):
        if self.extrinsic_locked:
            return

        det1 = self._get_tag_detection(cam1_msg, BASE_TAG_ID)
        det2 = self._get_tag_detection(cam2_msg, BASE_TAG_ID)
        if det1 is None or det2 is None:
            return

        T_cam1_tag0 = self._pose_to_matrix(det1.pose.pose.pose)
        T_cam2_tag0 = self._pose_to_matrix(det2.pose.pose.pose)
        self.T_cam1_cam2 = T_cam1_tag0 @ np.linalg.inv(T_cam2_tag0)
        self.extrinsic_locked = True
        rospy.loginfo("Locked T_cam1_cam2 from tag 0 detections in both cameras.")

    @staticmethod
    def _is_cube_tag(det):
        return det.id and det.id[0] in CUBE_TAG_IDS

    def _passes_margin(self, det):
        return det.decision_margin >= self.min_decision_margin

    def _transform_detection_to_cam1(self, det):
        out = copy.deepcopy(det)
        T_cam2_tag = self._pose_to_matrix(det.pose.pose.pose)
        T_cam1_tag = self.T_cam1_cam2 @ T_cam2_tag
        self._matrix_to_pose(T_cam1_tag, out.pose.pose.pose)
        out.pose.header.frame_id = self.output_frame
        return out

    def _merge_cube_tags(self, cam1_msg, cam2_detections):
        merged = {}

        for det in cam1_msg.detections:
            if not self._is_cube_tag(det) or not self._passes_margin(det):
                continue
            tag_id = det.id[0]
            merged[tag_id] = copy.deepcopy(det)

        if self.T_cam1_cam2 is not None:
            for det in cam2_detections:
                if not self._is_cube_tag(det) or not self._passes_margin(det):
                    continue
                tag_id = det.id[0]
                transformed = self._transform_detection_to_cam1(det)
                if (
                    tag_id not in merged
                    or transformed.decision_margin > merged[tag_id].decision_margin
                ):
                    merged[tag_id] = transformed

        return list(merged.values())

    def _cam2_is_fresh(self, cam1_stamp):
        if self.cam2_msg is None or self.cam2_stamp is None:
            return False
        age = abs((cam1_stamp - self.cam2_stamp).to_sec())
        return age <= self.cam2_max_age_sec

    def cam2_callback(self, msg):
        self.cam2_msg = msg
        if msg.header.stamp.to_sec() > 0:
            self.cam2_stamp = msg.header.stamp
        else:
            self.cam2_stamp = rospy.Time.now()

    def cam1_callback(self, msg):
        if not msg.detections:
            return

        cam1_stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0 else rospy.Time.now()
        cam2_fresh = self.cam2_msg if self._cam2_is_fresh(cam1_stamp) else None

        if cam2_fresh is not None:
            self._try_lock_extrinsic(msg, cam2_fresh)

        if not self.extrinsic_locked:
            rospy.logwarn_throttle(
                5.0,
                "Waiting for tag 0 in both cameras to lock extrinsics; publishing cam1 cube tags only.",
            )

        cube_tags = self._merge_cube_tags(
            msg, cam2_fresh.detections if cam2_fresh is not None else []
        )
        if not cube_tags:
            return

        out = AprilTagDetectionArray()
        out.header = msg.header
        out.header.frame_id = self.output_frame
        out.detections = cube_tags
        self.pub.publish(out)


if __name__ == "__main__":
    try:
        rospy.init_node("tag_detections_merger", anonymous=False)
        TagDetectionsMerger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
