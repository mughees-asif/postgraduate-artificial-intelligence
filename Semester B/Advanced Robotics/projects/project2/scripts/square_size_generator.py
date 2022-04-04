#!/usr/bin/env python

import rospy
import random
from std_msgs.msg import Float64

# Generate a random value for the size of the square (i.e. the length of the side of the square)
# Every 20 seconds#
# Publish them on a ROS Topic using an appropriate message.
def main():
    pub = rospy.Publisher('square_size', Float64, queue_size=10)
    rospy.init_node('square_size_generator')
    rate = rospy.Rate(0.05)

    print("square_size_generator: publishing messages on topic 'square_size'")

    while not rospy.is_shutdown():
        side_length = round(random.uniform(0.05, 0.20), 6)
        rospy.loginfo( "Length of square side: %s", side_length)
        pub.publish(side_length)
        rate.sleep()


# Invoke the `main` function
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

