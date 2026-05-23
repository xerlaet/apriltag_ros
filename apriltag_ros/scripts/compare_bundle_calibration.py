#!/usr/bin/env python
"""
Compare measured bundle calibration (master-tag frame) against tags.yaml (cube-center frame).

Reference poses in tags.yaml are transformed into the master-tag frame:
T_master_tag = inv(T_cube_master) @ T_cube_tag
"""

from __future__ import print_function

import argparse
import os
import re
import sys

import numpy as np
import tf.transformations as tfs


def parse_layout_entries(yaml_text):
    """Extract {id, size, x, y, z, qw, qx, qy, qz} dicts from tag_bundles layout."""
    entries = []
    pattern = re.compile(
        r"\{id:\s*(\d+),\s*size:\s*([\d.eE+-]+),\s*"
        r"x:\s*([\d.eE+-]+),\s*y:\s*([\d.eE+-]+),\s*z:\s*([\d.eE+-]+),\s*"
        r"qw:\s*([\d.eE+-]+),\s*qx:\s*([\d.eE+-]+),\s*qy:\s*([\d.eE+-]+),\s*qz:\s*([\d.eE+-]+)\}"
    )
    for match in pattern.finditer(yaml_text):
        entries.append({
            "id": int(match.group(1)),
            "size": float(match.group(2)),
            "x": float(match.group(3)),
            "y": float(match.group(4)),
            "z": float(match.group(5)),
            "qw": float(match.group(6)),
            "qx": float(match.group(7)),
            "qy": float(match.group(8)),
            "qz": float(match.group(9)),
        })
    return entries


def load_bundle_layout(path, bundle_name=None):
    with open(path) as f:
        text = f.read()

    if bundle_name:
        # Restrict to named bundle block if multiple exist.
        name_pattern = re.compile(
            r"name:\s*['\"]{}['\"]".format(re.escape(bundle_name))
        )
        if not name_pattern.search(text):
            raise ValueError(
                "Bundle '{}' not found in {}".format(bundle_name, path)
            )

    entries = parse_layout_entries(text)
    if not entries:
        raise ValueError("No layout entries found in {}".format(path))
    return {e["id"]: e for e in entries}


def layout_to_matrix(entry):
    quat = [entry["qx"], entry["qy"], entry["qz"], entry["qw"]]
    T = tfs.quaternion_matrix(quat)
    T[:3, 3] = [entry["x"], entry["y"], entry["z"]]
    return T


def invert_transform(T):
    R = T[:3, :3]
    p = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ p
    return Tinv


def quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def quat_angle_deg(q1_wxyz, q2_wxyz):
    """Angle between two orientations in degrees."""
    q1 = quat_wxyz_to_xyzw(q1_wxyz)
    q2 = quat_wxyz_to_xyzw(q2_wxyz)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = abs(np.dot(q1, q2))
    dot = min(1.0, dot)
    return np.degrees(2.0 * np.arccos(dot))


def reference_in_master_frame(reference_layout, master_id):
    if master_id not in reference_layout:
        raise ValueError("Master tag {} not in reference layout".format(master_id))

    T_cube_master = layout_to_matrix(reference_layout[master_id])
    T_master_cube = invert_transform(T_cube_master)

    result = {}
    for tag_id, entry in reference_layout.items():
        T_cube_tag = layout_to_matrix(entry)
        T_master_tag = T_master_cube @ T_cube_tag
        q_xyzw = tfs.quaternion_from_matrix(T_master_tag)
        result[tag_id] = {
            "size": entry["size"],
            "x": T_master_tag[0, 3],
            "y": T_master_tag[1, 3],
            "z": T_master_tag[2, 3],
            "qw": q_xyzw[3],
            "qx": q_xyzw[0],
            "qy": q_xyzw[1],
            "qz": q_xyzw[2],
        }
    return result


def compare(reference_master, measured_layout, master_id):
    rows = []
    ref_ids = set(reference_master.keys())
    meas_ids = set(measured_layout.keys())
    common = sorted(ref_ids & meas_ids)
    missing_meas = sorted(ref_ids - meas_ids)
    extra_meas = sorted(meas_ids - ref_ids - {master_id})

    for tag_id in common:
        ref = reference_master[tag_id]
        meas = measured_layout[tag_id]
        ref_p = np.array([ref["x"], ref["y"], ref["z"]])
        meas_p = np.array([meas["x"], meas["y"], meas["z"]])
        trans_err_m = np.linalg.norm(meas_p - ref_p)
        ref_q = [ref["qw"], ref["qx"], ref["qy"], ref["qz"]]
        meas_q = [meas["qw"], meas["qx"], meas["qy"], meas["qz"]]
        rot_err_deg = quat_angle_deg(meas_q, ref_q)
        rows.append({
            "id": tag_id,
            "trans_mm": trans_err_m * 1000.0,
            "rot_deg": rot_err_deg,
            "ref_x": ref["x"],
            "ref_y": ref["y"],
            "ref_z": ref["z"],
            "meas_x": meas["x"],
            "meas_y": meas["y"],
            "meas_z": meas["z"],
        })

    rows.sort(key=lambda r: r["trans_mm"], reverse=True)
    return rows, missing_meas, extra_meas


def print_table(rows, master_id):
    print("")
    print("Comparison in master-tag frame (master id = {})".format(master_id))
    print("")
    header = (
        "{:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}"
    ).format(
        "id", "trans_mm", "rot_deg",
        "ref_x", "ref_y", "ref_z",
        "meas_x", "meas_y", "meas_z",
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            "{:4d} {:10.3f} {:10.3f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}".format(
                r["id"], r["trans_mm"], r["rot_deg"],
                r["ref_x"], r["ref_y"], r["ref_z"],
                r["meas_x"], r["meas_y"], r["meas_z"],
            )
        )

    if rows:
        trans = [r["trans_mm"] for r in rows if r["id"] != master_id]
        rot = [r["rot_deg"] for r in rows if r["id"] != master_id]
        if trans:
            print("")
            print("Summary (excluding master):")
            print("  tags compared: {}".format(len(trans)))
            print("  translation RMSE: {:.3f} mm".format(
                np.sqrt(np.mean(np.square(trans)))
            ))
            print("  translation max:  {:.3f} mm".format(max(trans)))
            print("  rotation max:     {:.3f} deg".format(max(rot)))


def main():
    parser = argparse.ArgumentParser(
        description="Compare measured bundle calibration to tags.yaml reference.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference tags.yaml (cube-center frame)",
    )
    parser.add_argument(
        "--measured",
        required=True,
        help="Measured bundle YAML from calibrate_bundle.py",
    )
    parser.add_argument(
        "--master-id",
        type=int,
        default=7,
        help="Master tag ID used during calibration (default: 7)",
    )
    parser.add_argument(
        "--bundle-name",
        default="cube",
        help="Bundle name in reference file (default: cube)",
    )
    args = parser.parse_args()

    for path in (args.reference, args.measured):
        if not os.path.isfile(path):
            print("File not found: {}".format(path), file=sys.stderr)
            return 1

    ref_layout = load_bundle_layout(args.reference, args.bundle_name)
    meas_layout = load_bundle_layout(args.measured)

    ref_master = reference_in_master_frame(ref_layout, args.master_id)
    rows, missing_meas, extra_meas = compare(
        ref_master, meas_layout, args.master_id
    )

    print_table(rows, args.master_id)

    if missing_meas:
        print("")
        print("Not measured (never co-visible with master in bag):")
        print("  {}".format(", ".join(str(i) for i in missing_meas)))

    if extra_meas:
        print("")
        print("Measured but not in reference:")
        print("  {}".format(", ".join(str(i) for i in extra_meas)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
