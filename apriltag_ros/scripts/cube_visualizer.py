#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped


class CubeVisualizer:
    def __init__(self):
        rospy.init_node("cube_visualizer")

        # Parameters
        self.marker_topic = rospy.get_param("~marker_topic", "/cube_marker")
        self.cube_size = rospy.get_param("~cube_size", 0.07)  # 7cm actual cube
        self.mesh_resource = rospy.get_param(
            "~mesh_resource", "package://apriltag_ros/meshes/model.obj"
        )

        self.goal_frame = rospy.get_param("~goal_frame", "goal_rot")
        self.reference_frame = rospy.get_param("~reference_frame", "palm_lower")

        # Publisher
        self.marker_pub = rospy.Publisher(self.marker_topic, Marker, queue_size=10)

        # TF Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # State
        self.current_odom = None

        # Subscriber for actual cube
        rospy.Subscriber("/obj_odometry", Odometry, self.odom_callback)

        # Timer for markers (20Hz)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.timer_callback)

        # Clear any legacy SUCCESS text marker
        self._delete_success_text_marker()

        rospy.loginfo(
            "Cube Visualizer Layered Initialized (Size: %.2fm)", self.cube_size
        )

    def odom_callback(self, msg):
        self.current_odom = msg

    def _delete_success_text_marker(self):
        m_text = Marker()
        m_text.header.frame_id = self.reference_frame
        m_text.header.stamp = rospy.Time.now()
        m_text.ns = "success_text"
        m_text.id = 4
        m_text.action = Marker.DELETE
        self.marker_pub.publish(m_text)

    def create_marker(self, ns, id, m_type, pose_p, pose_q, scale, color, mesh=""):
        m = Marker()
        m.header.frame_id = self.reference_frame
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = id
        m.type = m_type
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = pose_p
        (
            m.pose.orientation.x,
            m.pose.orientation.y,
            m.pose.orientation.z,
            m.pose.orientation.w,
        ) = pose_q

        if m_type == Marker.MESH_RESOURCE:
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.mesh_resource = mesh
            m.mesh_use_embedded_materials = True
        else:
            m.scale.x = m.scale.y = m.scale.z = scale

        m.color.r, m.color.g, m.color.b, m.color.a = color
        return m

    def timer_callback(self, event):
        goal_q = None
        goal_p = None
        actual_q = None
        actual_p = None

        # 1. Get Goal Pose (with -0.1 offset from old script)
        try:
            goal_tf = self.tf_buffer.lookup_transform(
                self.reference_frame, self.goal_frame, rospy.Time(0)
            )
            goal_q = [
                goal_tf.transform.rotation.x,
                goal_tf.transform.rotation.y,
                goal_tf.transform.rotation.z,
                goal_tf.transform.rotation.w,
            ]
            goal_p = [
                goal_tf.transform.translation.x - 0.1,
                goal_tf.transform.translation.y - 0.1,
                goal_tf.transform.translation.z - 0.1,
            ]
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            pass

        # 2. Get Actual Pose
        if self.current_odom is not None:
            try:
                actual_pose_stamped = PoseStamped()
                actual_pose_stamped.header = self.current_odom.header
                actual_pose_stamped.pose = self.current_odom.pose.pose
                actual_tf = self.tf_buffer.transform(
                    actual_pose_stamped, self.reference_frame, rospy.Duration(0.05)
                )
                actual_q = [
                    actual_tf.pose.orientation.x,
                    actual_tf.pose.orientation.y,
                    actual_tf.pose.orientation.z,
                    actual_tf.pose.orientation.w,
                ]
                actual_p = [
                    actual_tf.pose.position.x,
                    actual_tf.pose.position.y,
                    actual_tf.pose.position.z,
                ]
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                pass

        # 3. Publish Actual Cube
        if actual_p is not None:
            m_mesh = self.create_marker(
                "actual_mesh",
                0,
                Marker.MESH_RESOURCE,
                actual_p,
                actual_q,
                self.cube_size,
                (1, 1, 1, 1.0),
                self.mesh_resource,
            )
            self.marker_pub.publish(m_mesh)

        # 4. Publish Goal Cube
        if goal_p is not None:
            m_gmesh = self.create_marker(
                "goal_mesh",
                2,
                Marker.MESH_RESOURCE,
                goal_p,
                goal_q,
                self.cube_size,
                (1, 1, 1, 0.5),
                self.mesh_resource,
            )
            self.marker_pub.publish(m_gmesh)


if __name__ == "__main__":
    try:
        CubeVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
