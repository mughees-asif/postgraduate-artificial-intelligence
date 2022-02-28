#!/usr/bin/env python

from rospy import init_mode, Service, spin
from numpy import array, linalg
from ar_week5_test.srv import *


# Returns the coefficients of the trajectory by solving a system characterised as: Ax=b
def get_coeffs(req):
    A = array([
        [1, req.t0, req.t0 ** 2, req.t0 ** 3],
        [0, 1, 2 * req.t0, 3 * req.t0 ** 2],
        [1, req.tf, req.tf ** 2, req.tf ** 3],
        [0, 1, 2 * req.tf, 3 * req.tf ** 2]
    ])
    b = array([req.p0, req.v0, req.pf, req.vf])
    x = linalg.solve(A, b)

    return compute_cubic_trajResponse(x[0], x[1], x[2], x[3])


# Initializing the ROS Node and defining the required service
def main():
    print("get_coeffs: Providing service -> compute_coeffs")
    init_node('compute_cubic_coeffs')
    Service('compute_coeffs', compute_cubic_traj, get_coeffs)
    spin()


# Initializer
if __name__ == "__main__":
    main()
