#!/usr/bin/env python3
#coding=utf-8

import rospy
import numpy as np
import math
import RPi.GPIO as GPIO
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance
from geometry_msgs.msg import Pose


# Pin Definitions
"""
PWM pins must be configured using:
    sudo /opt/nvidia/jetson-io/jetson-io.py

If the screen just flash replace the lines in:
    sudo /opt/nvidia/jetson-io/jetson-io.py

     def __init__(self):
        self.appdir = None
        self.bootdir = '/boot'
        self.extlinux = '/boot/extlinux/extlinux.conf'
        dtbdir = os.path.join(self.bootdir, 'dtb')          <----------------------
        fio.is_rw(self.bootdir)

by 

    def __init__(self):
        self.appdir = None
        self.bootdir = '/boot'
        self.extlinux = '/boot/extlinux/extlinux.conf'
        dtbdir = os.path.join(self.bootdir, '')
        fio.is_rw(self.bootdir)

 https://forums.developer.nvidia.com/t/jetpack-4-3-l4t-r32-3-1-released/109271/13

"""

# PWM pin
output_pins = {
    'JETSON_XAVIER': 18,
    'JETSON_NANO': 33,
    'JETSON_NX': 33,
    'CLARA_AGX_XAVIER': 18,
    'JETSON_TX2_NX': 32,
    'JETSON_ORIN': 18,
}

output_pwm = output_pins.get(GPIO.model, None)
if output_pwm is None:
    raise Exception('PWM not supported on this board')

# Board pin-numbering scheme
GPIO.setmode(GPIO.BOARD)

# set pin as an output pin with optional initial state of LOW
GPIO.setup(output_pwm, GPIO.OUT, initial=GPIO.LOW)
p = GPIO.PWM(output_pwm, 50) # frecuency 50Hz
p.start(7.75) # is middle position of the servo
"""
7.75:  Middle position L ----- R
3  :  Left camera on the front and Right camera on the back
12.5: Right camera on the front and Left camera on the back
"""

# LED pin
output_pin = 12  # BCM pin 18, BOARD pin 12
GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH) # Set pin as an output pin with optional initial state of HIGH --- High is OFF


def laserData(msg):
    obsDetec(140, 220, 0.2, msg)


def odomData(msg):
    robot_orientation = msg.pose.pose.orientation
    servo(robot_orientation)

    
def obsDetec(initialAngle, finalAngle, obsDistance, msg):
    lidar_min_angle_rad   = msg.angle_min
    lidar_max_angle_rad   = msg.angle_max
    lidar_angle_incre_rad = msg.angle_increment
    distances = msg.ranges

    iniAngle = np.deg2rad(initialAngle)
    finAngle = np.deg2rad(finalAngle)
    ini_index = int((iniAngle - lidar_min_angle_rad) / lidar_angle_incre_rad)
    fin_index = int((finAngle - lidar_min_angle_rad) / lidar_angle_incre_rad)

    count = 0
    for i in range(ini_index,fin_index):
        if distances[i] <= obsDistance:
            count = count + 1

    if count >= (fin_index-ini_index)/2:
        #print("LED ON")
        GPIO.output(output_pin, GPIO.LOW)

    else:
        #print("LED OFF")
        GPIO.output(output_pin, GPIO.HIGH)


def servo(orientation):
    roll, pitch, yaw = euler_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w)
    percent = angle2percent(yaw)
    print(percent)
    p.ChangeDutyCycle(percent)



def angle2percent(angle):
    """
    Odom values are in rad
    If robot is pointing to the front = 0.0 rad
    If robot is pointing to the left  = 1.5 rad
    If robot is pointing to the right = -1.5 rad
    If robot points backwards         = -3.0 rad
    """
    if angle > 1.5 or angle < -1.5: # robot going from bottom to top
        percent = -3.166*angle + 7.75
    else: # robot going from top to bottom

    return percent


def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':
    rospy.init_node('objectives')
    rospy.Subscriber("/scan", LaserScan, laserData, queue_size=1)
    rospy.Subscriber("/odom", Odometry, odomData, queue_size=1)
    rospy.spin()
