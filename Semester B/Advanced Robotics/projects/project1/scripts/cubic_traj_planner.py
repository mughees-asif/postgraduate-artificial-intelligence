#!/usr/bin/env python

from rospy import init_node, spin, Subscriber, Publisher, \
    loginfo, wait_for_service, \
    get_caller_id, ServiceProxy, ServiceException
from ar_week5_test.msg import *
from ar_week5_test.srv import *


# Subscribes to the ROS Topic created by Node 1
# Reads the desired p0, pf, v0, vf, t0, tf published every 20 seconds
# Computes the a0,a1,a2,a3 coefficients of the cubic polynomial trajectory.
class ReadTrajectory():

    def __init__(self):
        self.param = None
        Subscriber("initial_data", cubic_traj_params, self.callback)
        self.pub = Publisher("final_data", cubic_traj_coeffs, queue_size=10)
        print
        "Reading Trajectory: receiving messages on topic 'initial_data', publishing messages on topic 'final_data'"

    def callback(self, data):
        self.param = data
        loginfo(get_caller_id() + "\n Received parameters \n %s", data)
        wait_for_service('compute_coeff')

        try:
            f_coeff = ServiceProxy('compute_coeff', compute_cubic_traj)
            resp1 = f_coeff(self.param.p0, self.param.pf, self.param.v0, self.param.vf, self.param.t0, self.param.tf)
        except ServiceException, e:
            print
            "Service call failed: %s" % e

        self.pub.publish(resp1.a0, resp1.a1, resp1.a2, resp1.a3, self.param.t0, self.param.tf)


# Initializer
if __name__ == '__main__':
    init_node('cubic_traj_planner')
    mynode = ReadTrajectory()
    spin()
