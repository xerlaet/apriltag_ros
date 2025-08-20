#!/usr/bin/env python

import rospy
import tf2_ros
import geometry_msgs.msg
from apriltag_ros.msg import AprilTagDetectionArray

class StaticTagBroadcaster(object):
    def __init__(self):
        self.pose_published = False

        rospy.init_node('static_qr_broadcaster', anonymous=True)
        
        # Create a StaticTransformBroadcaster
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()

        detection_topic = "/tag_detections"
        self.sub = rospy.Subscriber(detection_topic, AprilTagDetectionArray, self.detection_callback)

        rospy.loginfo("Waiting to broadcast static transform for AprilTag 0...")
        rospy.spin()

    def detection_callback(self, msg):
        if self.pose_published:
            return

        for detection in msg.detections:
            if detection.id[0] == 0:
                rospy.loginfo("Found tag 0! Publishing static transform.")

                # Create a TransformStamped message
                t = geometry_msgs.msg.TransformStamped()

                # Populate the message headers
                t.header.stamp = rospy.Time.now()
                # The parent frame is the frame the tag was detected in (e.g., 'camera_link')
                t.header.frame_id = detection.pose.header.frame_id 
                # The child frame is the new frame we are creating for the QR tag
                t.child_frame_id = "qr_code_frame" 
                
                # Copy the pose data into the transform message
                pose = detection.pose.pose.pose
                t.transform.translation.x = pose.position.x
                t.transform.translation.y = pose.position.y
                t.transform.translation.z = pose.position.z
                t.transform.rotation.x = pose.orientation.x
                t.transform.rotation.y = pose.orientation.y
                t.transform.rotation.z = pose.orientation.z
                t.transform.rotation.w = pose.orientation.w

                # Send the transform to tf2. It will be latched automatically.
                self.static_broadcaster.sendTransform(t)

                self.pose_published = True
                self.sub.unregister()
                rospy.loginfo("Transform for 'qr_code_frame' published. Node is now idle.")
                break

if __name__ == '__main__':
    try:
        StaticTagBroadcaster()
    except rospy.ROSInterruptException:
        pass