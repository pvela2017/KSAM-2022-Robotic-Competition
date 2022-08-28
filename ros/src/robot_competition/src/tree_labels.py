#!/usr/bin/env python3
#coding=utf-8

import rospy
import message_filters
import numpy as np

import socket

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance
from geometry_msgs.msg import Pose

x_pos = [0.55, 0.85, 1.15, 1.45]
y_pos = [0.45, 1.0, 1.55]


def callback(laser_sub, odom_sub):
    # laser_sub msgs from /scan topic
    # odom_sub  msgs from /odom topic
    # plus minus 2 centimeters from the tree
    epsilon = 0.02
    # range for getting the column in which the robot is
    column_range = 0.273 # got this value from map calculation
    robot_position = odom_sub.pose.pose.position

    for j in y_pos: # Check in which column the robot is j = 0: first column, j = 1: Middle column, j = 2: Third column
        if abs(robot_position.y - j) <= column_range:
            for i in x_pos: # Check near which tree line the robot is
                if abs(robot_position.x - i) <= epsilon:
                    left_tree = treeDetec() # Left tree detection
                    right_tree = treeDetec() # Right tree detection
                    if left_tree:
                        tree = tree_labelling(j, i, False) # Get tree label
                        message(tree, "l") # Send tree number and side
                    if right_tree:
                        tree = tree_labelling(j, i, True) # Get tree label
                        message(tree, "r") # Send tree number and side

    
def treeDetec(initialAngle, finalAngle, treeDistance, msg):
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
        if distances[i] <= treeDistance:
            count = count + 1

    if count >= (fin_index-ini_index)/2:
        return True # Tree detected

    else:
        return False # No tree detected


def tree_labelling(column, tree_line, left_right):
    # if left_right = True ----> right tree
    # if left_right = False ----> left tree
    # Right & Left trees organize on lists
    row0 = [(1, 5), (2, 6), (3, 7), (4, 8)]
    row1 = [(12, 8), (11, 7), (10, 6), (9, 5)]
    row2 = [(9, 13), (10, 14), (11, 15), (12, 16)]

    if column == 0 and tree_line < 4:
        # Right tree
        if left_right:
            return row0[tree_line][1]
        # Left tree
        else:
            return row0[tree_line][0]

    elif column == 1 and tree_line < 4:
            # Right tree
        if left_right:
            return row1[tree_line][1]
        # Left tree
        else:
            return row1[tree_line][0]
            
    elif column == 2 and tree_line < 4:
            # Right tree
        if left_right:
            return row2[tree_line][1]
        # Left tree
        else:
            return row2[tree_line][0]
    # Error        
    else:
        return -1

def message(tree, left_right):
    # Setup server connection
    server_address = ('127.0.0.1', 20001)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(server_address)

    # History of messages sent
    history = []

    # Create message
    message = str(tree) + "-" + left_right

    # Check if the msg has already been sent
    if not (message in history):
        history.append(message)
        # Send message
        sock.sendall(message.encode()) # Send data
    

# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':
    rospy.init_node('tree_labelling')
    laser_sub = message_filters.Subscriber('/scan', LaserScan)
    odom_sub = message_filters.Subscriber('/odom', Odometry)

    ts = message_filters.ApproximateTimeSynchronizer([laser_sub, odom_sub],  queue_size=5, slop=0.1, allow_headerless=True)
    ts.registerCallback(callback)

    rospy.spin()


"""
SERVER SIDE CODE

import socket

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 20001  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    while True:
        conn, addr = s.accept()
        data = conn.recv(3)
        print(data.decode())
        DO SOMETHING BLABLABLA
"""