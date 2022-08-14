#!/usr/bin/env python3

# https://answers.ros.org/question/277830/applying-restrictions-on-360-degree-lidar-for-slam-and-autonomous-navigation/

import rospy
import math
from sensor_msgs.msg import LaserScan



#copy the range and intensities of "/scan" topic to "ranges_filter" and "intensities_filter" 
#you need to convert them to "list" as "data.ranges" and "data.intensities" are "tuple"
def callback_scan(data):

    angle_min = 0
    angle_max = 60
    
    angle_min2 = 280
    angle_max2 = 360


    # Convert data to list
    ranges_filter = list(data.ranges)
    intensities_filter = list(data.intensities)

    increment = 360 / len(ranges_filter) 
    
     # Array index
    lower_index = math.floor(angle_min / increment)
    upper_index = math.floor(angle_max / increment)
    
    lower_index2 = math.floor(angle_min2 / increment)
    upper_index2 = math.floor(angle_max2 / increment)
    
    #print(upper_index2)

    # Remove values
    for x in range(lower_index, upper_index):
        ranges_filter[x] = 0
        intensities_filter[x] = 0
    
    for y in range(lower_index2, len(ranges_filter)):
        ranges_filter[y] = 0
        intensities_filter[y] = 0



    # Create new message
    current_time = rospy.Time.now()
    filterScan = LaserScan()
    filterScan.header.stamp = current_time
    filterScan.header.frame_id = 'base_scan'
    filterScan.angle_min = 0    # start angle of the scan [rad]
    filterScan.angle_max = 2 * math.pi   # end angle of the scan [rad]
    filterScan.angle_increment = data.angle_increment    # angular distance between measurements [rad]
    filterScan.time_increment = data.time_increment      # time between measurements [seconds]
    filterScan.range_min = 0.0                           # minimum range value [m]
    filterScan.range_max = 3.5                           # maximum range value [m]

    filterScan.ranges = ranges_filter
    filterScan.intensities = intensities_filter
        

    scan_pub.publish(filterScan)




# Define a new topic called "filterScan" to store all laser scanner data
rospy.init_node('laser_scan_filter')
rospy.Subscriber("/pre_scan", LaserScan, callback_scan) 
scan_pub = rospy.Publisher('/scan', LaserScan, queue_size=1)
rospy.spin()
