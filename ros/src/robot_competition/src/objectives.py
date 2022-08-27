#!/usr/bin/env python3
#coding=utf-8

import rospy
import numpy as np
#import RPi.GPIO as GPIO
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance
from geometry_msgs.msg import Pose


# Pin Definitions
#output_pin = 18  # BCM pin 18, BOARD pin 12
# Pin Setup:
#GPIO.setmode(GPIO.BCM)                              # BCM pin-numbering scheme from Raspberry Pi
#GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH) # Set pin as an output pin with optional initial state of HIGH --- High is OFF

def laserData(msg):
    obsDetec(140, 220, 0.2, msg)


def odomData(msg):
    robot_position = msg.pose.pose.position
    robot_orientation = msg.pose.pose.orientation

    
def obsDetec(initialAngle, finalAngle, obsDistance, msg):
    lidar_min_angle_rad   = msg.angle_min
    lidar_max_angle_rad   = msg.angle_max
    lidar_angle_incre_rad = msg.angle_increment
    distances = msg.ranges

    iniAngle = np.deg2rad(initialAngle)
    finAngle = np.deg2rad(finalAngle)
    ini_index = int((iniAngle - lidar_min_angle_rad) / lidar_angle_incre_rad)
    fin_index = int((finAngle - lidar_min_angle_rad) / lidar_angle_incre_rad)

    for i in range(ini_index,fin_index):
        if distances[i] <= obsDistance:
            print("LED ON")
            #GPIO.output(output_pin, GPIO.LOW)
        else:
            print("LED OFF")
            #GPIO.output(output_pin, GPIO.HIGH)



# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':
    rospy.init_node('objectives')
    rospy.Subscriber("/scan", LaserScan, laserData, queue_size=1)
    rospy.Subscriber("/odom", Odometry, odomData, queue_size=1)
    rospy.spin()
