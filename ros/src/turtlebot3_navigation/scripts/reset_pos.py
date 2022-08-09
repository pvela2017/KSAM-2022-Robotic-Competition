#!/usr/bin/env python3
#coding=utf-8

"""
Created on Thu Jun  15 15:53:09 2021



@author: Pablo
"""


import rospy
from std_srvs.srv import Empty

rospy.wait_for_service('/gazebo/reset_simulation')
reset_world = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
reset_world()



