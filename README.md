# KSAM-2022-Robotic-Competition

## Introduction
Write something

### Team Digi Ag
Write something

### Gazebo simulator
Write something

![This is an image](images/gazebo1.jpg)

## Instalation & run
To install this repository on your home folder:
```
cd ~
git clone https://github.com/mefisto2017/KSAM-2022-Robotic-Competition
cd KSAM-2022-Robotic-Competition/ros
catkin_make
```
Before running the repository the models path needs to be setup:
```
echo 'export GAZEBO_MODEL_PATH=~/KSAM-2022-Robotic-Competition/ros/src/robot_gazebo/models:${GAZEBO_MODEL_PATH}' >> ~/.bashrc
source ~/.bashrc
```
Finally to run the repository:
```
source ./devel/setup.bash
roslaunch robot_gazebo scenario_1_world.launch
```
In another terminal:
```
cd ~/KSAM-2022-Robotic-Competition/ros
source ./devel/setup.bash
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

