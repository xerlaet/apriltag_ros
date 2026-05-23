#!/usr/bin/env python
"""
Determine AprilTag bundle relative poses with respect to a master tag.

Port of scripts/calibrate_bundle.m (no MATLAB required).

Data collection (run once with cube + camera still, master tag visible):
  roslaunch apriltag_ros camera_all.launch enable_second_camera:=false
  mkdir -p $(rospack find apriltag_ros)/scripts/data
  rosbag record -O $(rospack find apriltag_ros)/scripts/data/calibration /tag_detections

Requires tags_single.yaml (standalone tags only, no tag_bundles) during recording.
"""

from __future__ import print_function

import argparse
import os
import sys
import warnings

import numpy as np
import rosbag
import tf.transformations as tfs
from apriltag_ros.msg import AprilTagDetectionArray


def pose_to_matrix(position, orientation):
    """Build 4x4 transform (tag frame -> camera frame) from geometry_msgs pose."""
    quat = [
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    ]
    T = tfs.quaternion_matrix(quat)
    T[:3, 3] = [position.x, position.y, position.z]
    return T


def invert_transform(T):
    R = T[:3, :3]
    p = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ p
    return Tinv


def matrix_to_quat_wxyz(T):
    """Return quaternion as [w, x, y, z] (MATLAB / tags.yaml convention)."""
    q_xyzw = tfs.quaternion_from_matrix(T)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def geometric_median(samples, tol=1e-6, maxiter=1000):
    """Robust median of 3D points via Weiszfeld algorithm."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 3:
        raise ValueError("samples must be Nx3")
    if samples.shape[0] == 1:
        return samples[0].copy()

    y = samples.mean(axis=0)
    for _ in range(maxiter):
        dists = np.linalg.norm(samples - y, axis=1)
        mask = dists > tol
        if not np.any(mask):
            break
        weights = 1.0 / dists[mask]
        y_new = np.sum(samples[mask] * weights[:, None], axis=0) / np.sum(weights)
        if np.linalg.norm(y_new - y) < tol:
            y = y_new
            break
        y = y_new
    return y


def quat_multiply(q1, q2):
    """Hamilton product for quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_inverse(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def average_quaternion(quats):
    """Landis et al. eigenvector average; quats are [w, x, y, z]."""
    Q = np.asarray(quats, dtype=float).T  # 4 x N
    if Q.shape[1] == 1:
        q = Q[:, 0]
        return q / np.linalg.norm(q)

    for j in range(Q.shape[1]):
        for k in range(j + 1, Q.shape[1]):
            q_err = quat_multiply(quat_inverse(Q[:, j]), Q[:, k])
            q_err_w = np.clip(q_err[0], -1.0, 1.0)
            angle = 2.0 * np.arccos(q_err_w)
            if angle >= np.pi / 2.0:
                warnings.warn(
                    "Quaternion pair {} and {} are {:.1f} deg apart".format(
                        j, k, np.degrees(angle)
                    )
                )

    _, eigvecs = np.linalg.eigh(Q @ Q.T)
    q_mean = eigvecs[:, -1]
    if q_mean[0] < 0:
        q_mean = -q_mean
    return q_mean / np.linalg.norm(q_mean)


def load_tag_detections(bag_path, topic="/tag_detections"):
    """Return list of dicts: {stamp, tags: {id: (T_cam_tag, size)}}."""
    frames = []
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            tags = {}
            for det in msg.detections:
                if len(det.id) != 1:
                    warnings.warn(
                        "Skipping bundle detection with IDs {}".format(list(det.id))
                    )
                    continue
                tag_id = int(det.id[0])
                pose = det.pose.pose.pose
                T = pose_to_matrix(pose.position, pose.orientation)
                size = float(det.size[0]) if det.size else float("nan")
                tags[tag_id] = (T, size)
            if tags:
                frames.append({"stamp": t.to_sec(), "tags": tags})
    return frames


def calibrate(frames, master_id):
    rel_p = {}
    rel_q = {}
    master_size = None
    used_frames = 0

    for frame in frames:
        tags = frame["tags"]
        if master_id not in tags:
            continue
        used_frames += 1
        T_cm, master_size_frame = tags[master_id]
        if master_size is None:
            master_size = master_size_frame

        T_mc = invert_transform(T_cm)
        for tag_id, (T_cj, _size) in tags.items():
            if tag_id == master_id:
                continue
            T_mj = T_mc @ T_cj
            if tag_id not in rel_p:
                rel_p[tag_id] = []
                rel_q[tag_id] = []
            rel_p[tag_id].append(T_mj[:3, 3])
            rel_q[tag_id].append(matrix_to_quat_wxyz(T_mj))

    rel_sizes = {}
    for frame in frames:
        if master_id not in frame["tags"]:
            continue
        for tag_id, (_, size) in frame["tags"].items():
            if tag_id != master_id:
                rel_sizes.setdefault(tag_id, size)

    if master_size is None:
        raise RuntimeError(
            "Master tag with ID {} not found in detections".format(master_id)
        )

    other_ids = sorted(rel_p.keys())
    rel_p_median = {}
    rel_q_mean = {}
    sample_counts = {}
    for tag_id in other_ids:
        samples = np.array(rel_p[tag_id]).T  # 3 x N
        rel_p_median[tag_id] = geometric_median(samples.T)
        rel_q_mean[tag_id] = average_quaternion(rel_q[tag_id])
        sample_counts[tag_id] = len(rel_p[tag_id])

    return {
        "master_id": master_id,
        "master_size": master_size,
        "other_ids": other_ids,
        "rel_p_median": rel_p_median,
        "rel_q_mean": rel_q_mean,
        "rel_sizes": rel_sizes,
        "sample_counts": sample_counts,
        "used_frames": used_frames,
        "total_frames": len(frames),
    }


def format_bundle_yaml(result, bundle_name):
    lines = [
        "tag_bundles:",
        "  [",
        "    {",
        "      name: '{}',".format(bundle_name),
        "      layout:",
        "        [",
    ]

    master_id = result["master_id"]
    lines.append(
        "          {{id: {}, size: {:.5f}, x: {:.4f}, y: {:.4f}, z: {:.4f}, "
        "qw: {:.4f}, qx: {:.4f}, qy: {:.4f}, qz: {:.4f}}},".format(
            master_id, result["master_size"], 0, 0, 0, 1, 0, 0, 0
        )
    )

    for i, tag_id in enumerate(result["other_ids"]):
        p = result["rel_p_median"][tag_id]
        q = result["rel_q_mean"][tag_id]
        size = result["rel_sizes"].get(tag_id, result["master_size"])
        comma = "," if i < len(result["other_ids"]) - 1 else ""
        lines.append(
            "          {{id: {}, size: {:.5f}, x: {:.4f}, y: {:.4f}, z: {:.4f}, "
            "qw: {:.4f}, qx: {:.4f}, qy: {:.4f}, qz: {:.4f}}}{}".format(
                tag_id, size, p[0], p[1], p[2], q[0], q[1], q[2], q[3], comma
            )
        )

    lines.extend([
        "        ]",
        "    }",
        "  ]",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate AprilTag bundle relative poses from a rosbag.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data collection:
  roslaunch apriltag_ros camera_all.launch enable_second_camera:=false
  rosbag record -O $(rospack find apriltag_ros)/scripts/data/calibration /tag_detections

Use tags_single.yaml during recording (no tag_bundles section).
        """.strip(),
    )
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to calibration rosbag",
    )
    parser.add_argument(
        "--topic",
        default="/tag_detections",
        help="Topic name (default: /tag_detections)",
    )
    parser.add_argument(
        "--master-id",
        type=int,
        default=7,
        help="Master tag ID defining bundle origin (default: 7)",
    )
    parser.add_argument(
        "--name",
        default="cube_measured",
        help="Bundle name in output YAML",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Write YAML to this file (default: scripts/data/calibrated_bundle.yaml)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.bag):
        print("Bag file not found: {}".format(args.bag), file=sys.stderr)
        return 1

    print("Loading {} ...".format(args.bag))
    frames = load_tag_detections(args.bag, args.topic)
    print("Loaded {} messages with detections".format(len(frames)))

    result = calibrate(frames, args.master_id)
    print(
        "Used {}/{} frames containing master tag {}".format(
            result["used_frames"], result["total_frames"], args.master_id
        )
    )
    for tag_id in result["other_ids"]:
        print("  tag {:2d}: {} samples".format(tag_id, result["sample_counts"][tag_id]))

    yaml_text = format_bundle_yaml(result, args.name)
    print("\n" + yaml_text)

    output_path = args.output
    if not output_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "data", "calibrated_bundle.yaml")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(yaml_text)
    print("Wrote {}".format(output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
