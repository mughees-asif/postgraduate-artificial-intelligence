#!/usr/bin/env python

import rospy
from ar_week5_test.msg import *
from ar_week5_test.srv import *
from std_msgs.msg import Float64
import numpy as np


# Subscribes to the ROS Topic created by Node 2
# Reads the a0,a1,a2,a3 coefficients and t0,tf time parameters published every 20 seconds,
# Publishes three separate ROS topics: position trajectory, velocity trajectory and acceleration trajectory
# Visualized with the rqt_plot GUI, on the same plot, with different colors
class ReadCoeffs():

    def __init__(self):
        self.param = None
        rospy.Subscriber("final_data", cubic_traj_coeffs, self.callback)

        self.pub_position = rospy.Publisher("position_trajectory", Float64, queue_size=10)
        self.pub_velocity = rospy.Publisher("velocity_trajectory", Float64, queue_size=10)
        self.pub_acceleration = rospy.Publisher("acceleration_trajectory", Float64, queue_size=10)

        print
        "Reading Coefficients: receiving messages on topic: 'final_data', publishing messages on topics \n 'position_trajectory', \n 'velocity_trajectory', \n 'acceleration_trajectory' "

    def callback(self, data):
        self.param = data

        rospy.loginfo(rospy.get_caller_id() + "\n Trajectory coefficients \n %s", data)

        nint = int(round(data.tf) * 10)
        t = np.linspace(data.t0, data.tf, num=nint)

        self.pos = np.zeros(nint)
        self.vel = np.zeros(nint)
        self.acc = np.zeros(nint)

        for i in range(nint):
            self.pos[i] = (data.a0) + (data.a1) * t[i] + (data.a2) * (t[i]) ** 2 + (data.a3) * (t[i]) ** 3
            self.vel[i] = (data.a1) + 2 * (data.a2) * (t[i]) + 3 * (data.a3) * (t[i]) ** 2
            self.acc[i] = 2 * (data.a2) + 6 * (data.a3) * (t[i])

            self.rate = rospy.Rate(int(round(nint / self.param.tf)))

            self.pub_position.publish(self.pos[i])
            self.pub_velocity.publish(self.vel[i])
            self.pub_acceleration.publish(self.acc[i])

            self.rate.sleep()


# Initializer
if __name__ == '__main__':
    rospy.init_node('plot_cubic_traj')
    mynode = ReadCoeffs()
    rospy.spin()
