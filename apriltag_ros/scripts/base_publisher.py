#!/usr/bin/env python

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from apriltag_ros.msg import AprilTagDetectionArray

class BasePublisher():
    def __init__(self):
        self.camera_frame = None

        self.pub = tf2_ros.TransformBroadcaster()
        self.sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.detection_callback)

        # logs
        rospy.loginfo("base pose estimator node initialized")
        rospy.loginfo("waiting for tag detections on /tag_detections...")

    def detection_callback(self, msg):
        if not msg.detections:
            return
        if self.camera_frame is None:
            self.camera_frame = msg.detections[0].pose.header.frame_id
            rospy.loginfo("locked camera frame to: %s", self.camera_frame)

        for detection in msg.detections:
            if detection.id[0] == 0:

                # create the transform message
                msg = TransformStamped()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = detection.pose.header.frame_id 
                msg.child_frame_id = 'tag_frame'
                
                pose = detection.pose.pose.pose
                msg.transform.translation.x = pose.position.x
                msg.transform.translation.y = pose.position.y
                msg.transform.translation.z = pose.position.z
                msg.transform.rotation.x = pose.orientation.x
                msg.transform.rotation.y = pose.orientation.y
                msg.transform.rotation.z = pose.orientation.z
                msg.transform.rotation.w = pose.orientation.w

                # publish the transform message
                self.pub.sendTransform(msg)
                break

if __name__ == '__main__':
    try:
        rospy.init_node('base_publisher', anonymous=True)
        node = BasePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
