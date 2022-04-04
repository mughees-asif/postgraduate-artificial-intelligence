# `ROS` package to generate Cartesian space movements of the end-effector of the Panda robot manipulator

<p align="center">
    <img height=500 src="https://github.com/mughees-asif/postgraduate-artificial-intelligence/blob/master/Semester%20B/Advanced%20Robotics/projects/project2/assets/demo.gif">
</p>

## Name: Mughees Asif
## Student ID: 180288337


### Setup instructions

- Recommended with `Python version 2.7`.

- Navigate to the the `src` sub-folder within the `catkin_ws` package and clone the dependencies:
	`$ cd ~/catkin_ws/src/`
	`$ git clone -b melodic-devel https://github.com/ros-planning/panda_moveit_config.git`
	`$ rosdep update`
	`$ rosdep install --from-paths . --ignore-src -r -y`

- Change the directory to the root folder to build the package:
	`$ cd ~/catkin_ws/`
	`$ catkin_make`

- Execute the setup file:
	`$ . ~/catkin_ws/devel/setup.bash`

- Open four different terminals and execute the following commands sequentially:
	`$ roslaunch panda_moveit_config demo.launch`
	`$ rosrun AR_week10_test square_size_generator.py`
	`$ rosrun AR_week10_test move_panda_square.py`
	`$ rosrun rqt_plot rqt_plot`

- If the plotter does not show any output, you can manually add the required trajectories. Using the `Topic` textfield in the upper left-hand corner, add the following sequentially:
	- /joint_states/position[0]
	- /joint_states/position[1]
	- /joint_states/position[2]
	- /joint_states/position[3]
	- /joint_states/position[4]
	- /joint_states/position[5]
	- /joint_states/position[6]

#### Note

- If you get an error that suggests that the `.py` file extensions are not being recognised during the launch of the Panda GUI, you will have to [change the permission](https://askubuntu.com/questions/443789/what-does-chmod-x-filename-do-and-how-do-i-use-it) for each file in the `~/catkin_ws/src/ar_week5_test/scripts/` folder.
	- Navigate to the scripts folder using the Terminal:
		`cd ~/catkin_ws/src/ar_week10_test/scripts/`
	- Execute:
		`chmod -R 777 <PYTHON_FILE_NAME>`
