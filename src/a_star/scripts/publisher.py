#! /usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

def drive():

    rospy.init_node('command')
    move_straight = rospy.Publisher('straight', Float64, queue_size=10)
    rate = rospy.Rate(5)
    
    while not rospy.is_shutdown():
        t0 = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - t0 < 15:
            message = Float64()
            message =-10
            move_straight.publish(message)
            rospy.loginfo("Data is being sent")
            rate.sleep()
        

        message = 0
        move_straight.publish(message)
        break
    


if __name__ == '__main__':
    try:
        drive()
    except rospy.ROSInterruptException: 
        pass
