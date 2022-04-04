#!/usr/bin/env python

import numpy as np
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from std_msgs.msg import Float64


line = "----------------------------------------------------------"

# This node subscribes to the ROS Topic created by `square_size_generator
# and waits for messages (which will include the desired length of the side of the square)
class MoveGroupPythonInterfaceTutorial(object):
  def __init__(self):

    super(MoveGroupPythonInterfaceTutorial, self).__init__()
    print(line + "\nMove Panda - Initializing\n" + line)

    # Initialize subscription to the topic containing the length of square side
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_panda_square')

    # Store information such as the kinematic model and the joint states
    robot = moveit_commander.RobotCommander()

    # Robot's interface with the external environment:
    scene = moveit_commander.PlanningSceneInterface()

    # Plan and execute the movements:
    group_name = "panda_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    # Display the trajectories
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)

    self.robot = robot
    self.scene = scene
    self.move_group = move_group
    self.display_trajectory_publisher = display_trajectory_publisher
    self.flagA = 0

  def callback(self, data ):
      side_length = 0
      side_length = data.data
      self.flagA = self.flagA + 1
        

  def go_to_joint_state(self):
    # Starting configuration
    joint_goal = self.move_group.get_current_joint_values()
    start_conf = [0, -pi/4, 0, -pi/2, 0, -pi/3, 0]
    for i in range(len(start_conf)):
        joint_goal[i] = start_conf[i]

    # Move the robot to the starting configuration
    self.move_group.go(joint_goal, wait=True)
    self.move_group.stop()

  # Realize the desired motion of the robot end-effector
  def plan_cartesian_path(self, scale=1):

    side_length = 0
   w_points = []

    # Current pose
    w_pose = self.move_group.get_current_pose().pose

    # X-Y directional movement
    w_pose.position.y += scale * side_length
    w_points.append(copy.deepcopy(w_pose))

    w_pose.position.x += scale * side_length
    w_points.append(copy.deepcopy(w_pose))

    w_pose.position.y -= scale * side_length
    w_points.append(copy.deepcopy(w_pose))

    w_pose.position.x -= scale * side_length
    w_points.append(copy.deepcopy(w_pose))

    (plan, fraction) = self.move_group.compute_cartesian_path(
                                       w_points,
                                       0.01,
                                       0.0)

    return plan, fraction

  # Displays the trajectories
  def display_trajectory(self, plan):
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = self.robot.get_current_state()
    display_trajectory.trajectory.append(plan)
    self.display_trajectory_publisher.publish(display_trajectory)

  def execute_plan(self, plan):
    self.move_group.execute(plan, wait=True)

  def main(self):
    side_length = 0
    A = 1
    rospy.Subscriber('square_size', Float64, self.callback)
    print(line + "\nMove Panda - Waiting for desired size of square trajectory\n" + line)

    while not rospy.is_shutdown():
       # Conditional check to ensure execution after the data is received from 'square_size'
       if self.flagA == A:
           print(line + "\nMove Panda - Received desired size, s= {}\n".format(side_length) + line )
           print(line + "\nMove Panda - Going to start configuration\n" + line )
           self.go_to_joint_state()

	   print(line + "\nMove Panda - Planning motion trajectory\n" + line)
           cartesian_plan, fraction = self.plan_cartesian_path()
           rospy.sleep(32 * (side_length + 0.4))

           print(line + "\nMove Panda - Showing planned trajectory\n" + line)
           self.display_trajectory(cartesian_plan)
           rospy.sleep(32 * (side_length + 0.4))

           print(line + "\nMove Panda - Executing planned trajectory\n" + line)
           self.execute_plan(cartesian_plan)

           print(line + "\nMove Panda - Waiting for desired size of square trajectory\n" + line)
           A = A + 1

# Instantiate class
if __name__ == '__main__':
   my_node = MoveGroupPythonInterfaceTutorial().main()
   
