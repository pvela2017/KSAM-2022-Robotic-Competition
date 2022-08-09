#!/usr/bin/env python3
#coding=utf-8

import rospy
# Brings in the SimpleActionClient
import actionlib
# Brings in the .action file and messages used by the move base action
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import transforms3d

def movebase_client(current_goal):
   # Create an action client called "move_base" with action definition file "MoveBaseAction"
    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
 
   # Waits until the action server has started up and started listening for goals.
    client.wait_for_server()

   # Creates a new goal with the MoveBaseGoal constructor
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
   # Move 0.5 meters forward along the x axis of the "map" coordinate frame 
    goal.target_pose.pose.position.x = current_goal[0]
    goal.target_pose.pose.position.y = current_goal[1]
   # No rotation of the mobile base frame w.r.t. map frame
    q = transforms3d.quaternions.axangle2quat([0, 0, 0] ,current_goal[2], True)
    # print(q)
    # TODO fix the quaternion!
    goal.target_pose.pose.orientation.x = 0#q[0]
    goal.target_pose.pose.orientation.y = 0#q[1]
    goal.target_pose.pose.orientation.z = 0#q[2]
    goal.target_pose.pose.orientation.w = 1#q[3]
    #print(goal)

   # Sends the goal to the action server.
    client.send_goal(goal)
   # Waits for the server to finish performing the action.
    wait = client.wait_for_result()
   # If the result doesn't arrive, assume the Server is not available
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
    # Result of executing the action
        return client.get_result()   

# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':
    goals = [[0.426538, 0.222959, -1.5436], [0.4265, 1.6298, -1.5436], [0.9812, 1.6298, 1.5527], [0.9812, 0.4523, 1.5527], [1.5574, 0.4523, -1.5436], [1.6832, 1.9112, -1.5436]]
    path_completed = False
    try:
       # Initializes a rospy node to let the SimpleActionClient publish and subscribe
        rospy.init_node('goals')
        while not path_completed:
            result = movebase_client(goals.pop(0))
            if result:
                rospy.loginfo("Goal execution done!")
                if not goals: 
                    rospy.loginfo("Path completed")
                    path_completed = True

    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")
