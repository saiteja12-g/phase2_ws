#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64

rear_Lwheel = rospy.Publisher('/carRobot2/driveL_controller/command', Float64, queue_size=10)
rear_Rwheel = rospy.Publisher('/carRobot2/driveR_controller/command', Float64, queue_size=10)


def straight_push(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", - data.data)
    rear_Lwheel.publish(data.data)
    rear_Rwheel.publish(data.data)

def listener():
    rospy.init_node('controller', anonymous=True)
    rospy.Subscriber("straight", Float64, straight_push)
    rospy.loginfo("Data is received")
    rospy.spin()

if __name__ == '__main__':
    listener()
