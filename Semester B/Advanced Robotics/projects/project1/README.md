# Automatically generating point-to-point cubic trajectories connecting pairs of randomly-generated points using `ROS`

<p align="center">
    <img height=500 src="https://github.com/mughees-asif/postgraduate-artificial-intelligence/blob/master/Semester%20B/Advanced%20Robotics/projects/project1/assets/demo.gif">
</p>

### Name: Mughees Asif

### Student ID: 180288337

### Setup instructions

- Python version 2.7.

- Unzip the folder in the `catkin_ws` directory in the `src` sub-folder:
	- `~/catkin_ws/src/ar_week5_test`

- Launch terminal, change directory to `~/catkin_ws/`, and build the package:
	- `$ cd ~/catkin_ws/`
	- `$ catkin_make`

- Run the setup file:
	- `$ . ~/catkin_ws/devel/setup.bash`

- Change directory to the launch folder using `ROS`:
	- `$ roscd ar_week5_test/launch`

- Launch GUI:
	- `$roslaunch cubic_traj_gen.launch$`

#### Note

- If you get an error that suggests that the `nodes` are not being recognised during the launch of the GUI, you will have to [change the permission](https://askubuntu.com/questions/443789/what-does-chmod-x-filename-do-and-how-do-i-use-it) for each `.py` file in the `~/catkin_ws/src/ar_week5_test/scripts/` folder.
