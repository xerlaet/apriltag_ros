#!/usr/bin/env python

import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose

class CubeMarkerPublisher:
    def __init__(self):
        rospy.init_node('cube_marker_publisher')
        
        # Parameters
        self.marker_topic = rospy.get_param('~marker_topic', '/cube_marker')
        self.mesh_resource = rospy.get_param('~mesh_resource', 'package://apriltag_ros/meshes/model.obj')
        
        # Publisher
        self.marker_pub = rospy.Publisher(self.marker_topic, Marker, queue_size=10)
        
        # TF Buffer to track goal_rot
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # State
        self.last_goal_pose = None
        self.current_pose = None
        self.current_header = None
        
        # Subscriber for current pose
        rospy.Subscriber('/obj_odometry', Odometry, self.odometry_callback)
        
        # Timer for markers (10Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        
        rospy.loginfo("Cube Marker Publisher (Offset Goal) Initialized")

    def odometry_callback(self, msg):
        """Saves current odometry data."""
        self.current_pose = msg.pose.pose
        self.current_header = msg.header

    def timer_callback(self, event):
        """Publishes both the current and goal markers."""
        # 1. Publish Current Cube
        if self.current_pose:
            m = Marker()
            m.header = self.current_header
            m.ns = "dex_cube"
            m.id = 0
            m.type = Marker.MESH_RESOURCE
            m.action = Marker.ADD
            m.pose = self.current_pose
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color.r = m.color.g = m.color.b = m.color.a = 1.0
            m.mesh_resource = self.mesh_resource
            m.mesh_use_embedded_materials = True
            self.marker_pub.publish(m)

        # 2. Try to update Goal Pose from TF relative to palm_lower
        try:
            # Look up the transform from palm_lower to goal_rot
            t = self.tf_buffer.lookup_transform("palm_lower", "goal_rot", rospy.Time(0))
            
            # Basic validation: ensure no NaNs in translation
            if not any(map(lambda v: (v != v), [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])):
                self.last_goal_pose = Pose()
                
                # VISUALIZATION OFFSET: Move the goal cube 0.15m away from the hand
                # Adjust which axis (x, y, or z) as needed based on your view
                self.last_goal_pose.position.x = t.transform.translation.x + 0.15
                self.last_goal_pose.position.y = t.transform.translation.y - 0.15
                self.last_goal_pose.position.z = t.transform.translation.z
                self.last_goal_pose.orientation = t.transform.rotation
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

        # 3. Publish Goal Cube (Textured and Semi-Transparent)
        if self.last_goal_pose:
            m = Marker()
            m.header.frame_id = "palm_lower"
            m.header.stamp = rospy.Time.now()
            m.ns = "dex_cube_goal"
            m.id = 1
            m.type = Marker.MESH_RESOURCE
            m.action = Marker.ADD
            m.pose = self.last_goal_pose
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color.r = 1.0; m.color.g = 1.0; m.color.b = 1.0; m.color.a = 1.0
            m.mesh_resource = self.mesh_resource
            m.mesh_use_embedded_materials = True
            self.marker_pub.publish(m)

if __name__ == '__main__':
    try:
        node = CubeMarkerPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
