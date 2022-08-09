#!/usr/bin/env python3
#coding=utf-8

"""
Created on Thu Jun  15 15:53:09 2021



@author: Pablo
"""

import rospy
from geometry_msgs.msg import Twist
PI = 3.1415926535897

def rotate(speed, angle, clockwise):
    #Starts a new node
    rospy.init_node('routine', anonymous=True)
    velocity_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    vel_msg = Twist()

    #Converting from angles to radians
    angular_speed = speed*2*PI/360
    relative_angle = angle*2*PI/360

    #We wont use linear components
    vel_msg.linear.x=0
    vel_msg.linear.y=0
    vel_msg.linear.z=0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0

    # Checking if our movement is CW or CCW
    if clockwise:
        vel_msg.angular.z = -abs(angular_speed)
    else:
        vel_msg.angular.z = abs(angular_speed)
    # Setting the current time for distance calculus
    t0 = rospy.Time.now().to_sec()
    current_angle = 0

    while(current_angle < relative_angle):
        velocity_publisher.publish(vel_msg)
        t1 = rospy.Time.now().to_sec()
        current_angle = angular_speed*(t1-t0)


    #Forcing our robot to stop
    vel_msg.angular.z = 0
    velocity_publisher.publish(vel_msg)
    


def linear(speed, moving_time):
    #Starts a new node
    rospy.init_node('routine', anonymous=True)
    velocity_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    vel_msg = Twist()



    #We wont use linear components
    vel_msg.linear.x = speed
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    
    t0 = rospy.Time.now().to_sec()
    while t0 < 5:
        velocity_publisher.publish(vel_msg)
        t0 = rospy.Time.now().to_sec()

    

    #Forcing our robot to stop
    #vel_msg.linear.x = 0
    #velocity_publisher.publish(vel_msg)

if __name__ == '__main__':
    try:
        # Testing our function
        rotate(15, 90, True)
        rotate(15, 50, False)
        #linear(0.2, 3)
    except rospy.ROSInterruptException:
        pass

