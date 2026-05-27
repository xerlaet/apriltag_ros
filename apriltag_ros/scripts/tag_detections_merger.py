#!/usr/bin/env python
"""
Merge AprilTag detections from two cameras into the primary (D435) optical frame.

Cube tags (IDs 1-24) from the secondary camera are transformed using T_cam1_cam2.
Extrinsics are estimated from co-visible tag 0 and/or cube tags, then refined in
continuous mode with EMA smoothing. Large inter-camera shifts snap immediately.

Extrinsic update modes (param ~extrinsic_update_mode):
  - continuous: refine T_cam1_cam2 whenever co-visible reference tags exist
  - once: lock on first successful estimate (legacy behavior)
"""

import copy

import numpy as np
import rospy
import tf.transformations as tfs
from apriltag_ros.msg import AprilTagDetectionArray
from std_srvs.srv import Trigger, TriggerResponse

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
        self.cam2_max_age_sec = rospy.get_param("~cam2_max_age_sec", 0.1)
        self.min_decision_margin = rospy.get_param("~min_decision_margin", 50.0)
        self.extrinsic_update_mode = rospy.get_param("~extrinsic_update_mode", "continuous")
        self.extrinsic_update_alpha = rospy.get_param("~extrinsic_update_alpha", 0.5)
        self.tag0_min_decision_margin = rospy.get_param("~tag0_min_decision_margin", 50.0)
        self.extrinsic_update_min_interval_sec = rospy.get_param(
            "~extrinsic_update_min_interval_sec", 0.05
        )
        self.extrinsic_use_cube_tags = rospy.get_param("~extrinsic_use_cube_tags", True)
        self.extrinsic_snap_trans_m = rospy.get_param("~extrinsic_snap_trans_m", 0.002)
        self.extrinsic_snap_rot_rad = rospy.get_param("~extrinsic_snap_rot_rad", 0.05)

        self.T_cam1_cam2 = None
        self.cam1_msg = None
        self.cam1_stamp = None
        self.cam2_msg = None
        self.cam2_stamp = None
        self.extrinsic_locked = False
        self.last_extrinsic_update_time = None
        self._empty_cam1_msg = AprilTagDetectionArray()
        self._empty_cam1_msg.detections = []

        self.pub = rospy.Publisher(self.output_topic, AprilTagDetectionArray, queue_size=1)
        rospy.Subscriber(self.cam2_topic, AprilTagDetectionArray, self.cam2_callback, queue_size=1)
        rospy.Subscriber(self.cam1_topic, AprilTagDetectionArray, self.cam1_callback, queue_size=1)
        rospy.Service("~relock_extrinsics", Trigger, self._relock_extrinsics_callback)

        rospy.loginfo("TagDetectionsMerger initialized.")
        rospy.loginfo("  cam1=%s cam2=%s output=%s", self.cam1_topic, self.cam2_topic, self.output_topic)
        rospy.loginfo("  output_frame=%s min_decision_margin=%.1f", self.output_frame, self.min_decision_margin)
        rospy.loginfo(
            "  extrinsic_update_mode=%s alpha=%.2f tag0_margin=%.1f interval=%.2fs "
            "use_cube_tags=%s snap_trans=%.3fm snap_rot=%.2frad cam2_age=%.2fs",
            self.extrinsic_update_mode,
            self.extrinsic_update_alpha,
            self.tag0_min_decision_margin,
            self.extrinsic_update_min_interval_sec,
            self.extrinsic_use_cube_tags,
            self.extrinsic_snap_trans_m,
            self.extrinsic_snap_rot_rad,
            self.cam2_max_age_sec,
        )

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

    @staticmethod
    def _detection_index(msg):
        indexed = {}
        for det in msg.detections:
            if not det.id:
                continue
            tag_id = det.id[0]
            if tag_id in indexed:
                if det.decision_margin > indexed[tag_id].decision_margin:
                    indexed[tag_id] = det
            else:
                indexed[tag_id] = det
        return indexed

    @staticmethod
    def _quat_slerp(q0, q1, t):
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)
        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return s0 * q0 + s1 * q1

    @classmethod
    def _blend_transforms(cls, T_old, T_new, alpha):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        T_out = np.eye(4)
        T_out[:3, 3] = (1.0 - alpha) * T_old[:3, 3] + alpha * T_new[:3, 3]
        q_old = tfs.quaternion_from_matrix(T_old)
        q_new = tfs.quaternion_from_matrix(T_new)
        q_blend = cls._quat_slerp(q_old, q_new, alpha)
        T_out[:3, :3] = tfs.quaternion_matrix(q_blend)[:3, :3]
        return T_out

    @staticmethod
    def _rotation_delta_rad(T_a, T_b):
        R_delta = T_a[:3, :3].T @ T_b[:3, :3]
        trace = np.trace(R_delta)
        cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        return float(np.arccos(cos_angle))

    @classmethod
    def _adaptive_alpha(cls, T_old, T_new, base_alpha, snap_trans_m, snap_rot_rad):
        trans_delta = float(np.linalg.norm(T_old[:3, 3] - T_new[:3, 3]))
        rot_delta = cls._rotation_delta_rad(T_old, T_new)
        if trans_delta >= snap_trans_m or rot_delta >= snap_rot_rad:
            return 1.0
        return base_alpha

    @staticmethod
    def _extrinsic_from_pair(det1, det2):
        T_cam1_tag = TagDetectionsMerger._pose_to_matrix(det1.pose.pose.pose)
        T_cam2_tag = TagDetectionsMerger._pose_to_matrix(det2.pose.pose.pose)
        return T_cam1_tag @ np.linalg.inv(T_cam2_tag)

    @classmethod
    def _fuse_extrinsic_estimates(cls, estimates):
        if len(estimates) == 1:
            return estimates[0].copy()

        positions = np.array([T[:3, 3] for T in estimates])
        avg_pos = np.median(positions, axis=0)

        quats = []
        ref_quat = tfs.quaternion_from_matrix(estimates[0])
        for T in estimates:
            q = tfs.quaternion_from_matrix(T)
            if np.dot(q, ref_quat) < 0.0:
                q = -q
            quats.append(q)

        avg_quat = np.mean(quats, axis=0)
        avg_quat = avg_quat / np.linalg.norm(avg_quat)

        T_out = tfs.quaternion_matrix(avg_quat)
        T_out[:3, 3] = avg_pos
        return T_out

    def _collect_extrinsic_estimates(self, cam1_msg, cam2_msg):
        estimates = []
        used_tag0 = False
        used_cube = False
        cam1_tags = self._detection_index(cam1_msg)
        cam2_tags = self._detection_index(cam2_msg)

        det1 = cam1_tags.get(BASE_TAG_ID)
        det2 = cam2_tags.get(BASE_TAG_ID)
        if (
            det1 is not None
            and det2 is not None
            and det1.decision_margin >= self.tag0_min_decision_margin
            and det2.decision_margin >= self.tag0_min_decision_margin
        ):
            estimates.append(self._extrinsic_from_pair(det1, det2))
            used_tag0 = True

        if self.extrinsic_use_cube_tags:
            for tag_id in sorted(set(cam1_tags.keys()) & set(cam2_tags.keys()) & CUBE_TAG_IDS):
                det_a = cam1_tags[tag_id]
                det_b = cam2_tags[tag_id]
                if (
                    det_a.decision_margin < self.min_decision_margin
                    or det_b.decision_margin < self.min_decision_margin
                ):
                    continue
                estimates.append(self._extrinsic_from_pair(det_a, det_b))
                used_cube = True

        if used_tag0 and used_cube:
            source = "tag0+cube"
        elif used_tag0:
            source = "tag0"
        elif used_cube:
            source = "cube"
        else:
            source = "none"

        return estimates, source

    def _can_update_extrinsic_now(self, force=False):
        if force or self.last_extrinsic_update_time is None:
            return True
        elapsed = (rospy.Time.now() - self.last_extrinsic_update_time).to_sec()
        return elapsed >= self.extrinsic_update_min_interval_sec

    def _update_extrinsic(self, cam1_msg, cam2_msg):
        if self.extrinsic_update_mode == "once" and self.extrinsic_locked:
            return

        estimates, source = self._collect_extrinsic_estimates(cam1_msg, cam2_msg)
        if not estimates:
            return

        T_new = self._fuse_extrinsic_estimates(estimates)
        first_lock = self.T_cam1_cam2 is None

        if first_lock:
            alpha = 1.0
        elif self.extrinsic_update_mode == "continuous":
            alpha = self._adaptive_alpha(
                self.T_cam1_cam2,
                T_new,
                self.extrinsic_update_alpha,
                self.extrinsic_snap_trans_m,
                self.extrinsic_snap_rot_rad,
            )
        else:
            return

        if not self._can_update_extrinsic_now(force=(alpha >= 1.0)):
            return

        if first_lock:
            self.T_cam1_cam2 = T_new
            rospy.loginfo(
                "Locked T_cam1_cam2 from %d co-visible reference tag(s) (%s).",
                len(estimates),
                source,
            )
        else:
            self.T_cam1_cam2 = self._blend_transforms(self.T_cam1_cam2, T_new, alpha)
            rospy.loginfo_throttle(
                1.0,
                "Updated T_cam1_cam2 from %d tag(s) (%s, alpha=%.2f).",
                len(estimates),
                source,
                alpha,
            )

        self.extrinsic_locked = True
        self.last_extrinsic_update_time = rospy.Time.now()

    def _relock_extrinsics_callback(self, _req):
        self.T_cam1_cam2 = None
        self.extrinsic_locked = False
        self.last_extrinsic_update_time = None
        rospy.loginfo("Extrinsic relock requested; waiting for co-visible reference tags.")
        return TriggerResponse(
            success=True,
            message="Extrinsics cleared; will relock on next co-visible tag 0/cube tag(s).",
        )

    @staticmethod
    def _msg_stamp(msg):
        if msg is None:
            return None
        if msg.header.stamp.to_sec() > 0:
            return msg.header.stamp
        return rospy.Time.now()

    @staticmethod
    def _is_base_tag(det):
        return det.id and det.id[0] == BASE_TAG_ID

    @staticmethod
    def _is_cube_tag(det):
        return det.id and det.id[0] in CUBE_TAG_IDS

    def _passes_tag0_margin(self, det):
        return det.decision_margin >= self.tag0_min_decision_margin

    def _merge_base_tag(self, cam1_msg, cam2_detections):
        merged = None

        det1 = self._get_tag_detection(cam1_msg, BASE_TAG_ID)
        if det1 is not None and self._passes_tag0_margin(det1):
            merged = copy.deepcopy(det1)
            merged.pose.header.frame_id = self.output_frame

        if self.T_cam1_cam2 is not None:
            for det in cam2_detections:
                if not self._is_base_tag(det) or not self._passes_tag0_margin(det):
                    continue
                transformed = self._transform_detection_to_cam1(det)
                if merged is None or transformed.decision_margin > merged.decision_margin:
                    merged = transformed

        return merged

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

    def _passes_margin(self, det):
        return det.decision_margin >= self.min_decision_margin

    def _cam2_is_fresh(self, ref_stamp):
        if self.cam2_msg is None or self.cam2_stamp is None or ref_stamp is None:
            return False
        age = abs((ref_stamp - self.cam2_stamp).to_sec())
        return age <= self.cam2_max_age_sec

    def _cam1_is_fresh(self, ref_stamp):
        if self.cam1_msg is None or self.cam1_stamp is None or ref_stamp is None:
            return False
        age = abs((ref_stamp - self.cam1_stamp).to_sec())
        return age <= self.cam2_max_age_sec

    def _select_paired_messages(self):
        cam1 = self.cam1_msg
        cam2 = self.cam2_msg
        if cam1 is None and cam2 is None:
            return None, None, None

        cam1_stamp = self._msg_stamp(cam1)
        cam2_stamp = self._msg_stamp(cam2)

        if cam1 is not None and cam2 is not None:
            if self._cam2_is_fresh(cam1_stamp):
                return cam1, cam2, cam1_stamp
            if self._cam1_is_fresh(cam2_stamp):
                return cam1, cam2, cam2_stamp

        if cam2 is not None and (cam1 is None or not cam1.detections):
            return self._empty_cam1_msg, cam2, cam2_stamp

        if cam1 is not None:
            return cam1, None, cam1_stamp

        return None, None, None

    def _try_publish(self):
        cam1_paired, cam2_paired, out_stamp = self._select_paired_messages()
        if cam1_paired is None and cam2_paired is None:
            return

        if cam1_paired is not None and cam2_paired is not None:
            self._update_extrinsic(cam1_paired, cam2_paired)

        if not self.extrinsic_locked:
            rospy.logwarn_throttle(
                5.0,
                "Waiting for co-visible tag 0 or cube tags to lock extrinsics; publishing cam1 cube tags only.",
            )

        cam2_detections = cam2_paired.detections if cam2_paired is not None else []
        cube_tags = self._merge_cube_tags(cam1_paired, cam2_detections)
        base_tag = self._merge_base_tag(cam1_paired, cam2_detections)

        detections = list(cube_tags)
        if base_tag is not None:
            detections.append(base_tag)

        if not detections:
            return

        out = AprilTagDetectionArray()
        out.header.stamp = out_stamp if out_stamp is not None else rospy.Time.now()
        out.header.frame_id = self.output_frame
        out.detections = detections
        self.pub.publish(out)

    def cam2_callback(self, msg):
        self.cam2_msg = msg
        self.cam2_stamp = self._msg_stamp(msg)
        cam1_has_tags = self.cam1_msg is not None and self.cam1_msg.detections
        if not cam1_has_tags:
            self._try_publish()

    def cam1_callback(self, msg):
        self.cam1_msg = msg
        self.cam1_stamp = self._msg_stamp(msg)
        if msg.detections:
            self._try_publish()


if __name__ == "__main__":
    try:
        rospy.init_node("tag_detections_merger", anonymous=False)
        TagDetectionsMerger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
