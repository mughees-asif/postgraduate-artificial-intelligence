cmake_minimum_required(VERSION 3.0.2)
project(ar_week10_test)

find_package(catkin REQUIRED COMPONENTS
  moveit_commander
  moveit_msgs
  rospy
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ar_week10_test
#  CATKIN_DEPENDS moveit_commander moveit_msgs rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS scripts/move_panda_square.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
catkin_install_python(PROGRAMS scripts/square_size_generator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
