# OpenPTrack Calibration and Startup procedures

---
#### REBUILD OPEN_PTRACK
set this to run make serially for debugging: `export ROS_PARALLEL_JOBS="-j1 -l1" `
```
. /opt/ros/melodic/setup.bash
rosdep install -y -r --from-paths .
catkin_make
```
---
#### AFTER LOGIN
```
cd /opt/catkin_ws/
export DISPLAY=:0
. /opt/catkin_ws/devel/setup.bash 
```

#### SET RESOLUTION IN VNC SESSION
```
sudo xrandr --fb 1920x1080
```

#### CHANGE REALSENSE CAMERA RESOLUTION:
```
vim $(find /opt/catkin_ws -name rs_rgbd.launch)
```

### OPT LINKS
https://github.com/KayneWest/open_ptrack_lite/tree/matt-branch
https://github.com/OpenPTrack/open_ptrack_v2/wiki/Camera-Network-Calibration
https://github.com/OpenPTrack/open_ptrack_v2/wiki/Calibration-in-Practice
https://github.com/OpenPTrack/open_ptrack_v2/wiki/Calibration-Refinement-(Person-Based)
https://github.com/OpenPTrack/open_ptrack_v2/wiki/Multi-Imager-Calibration-Refinement-(Manual)
https://github.com/OpenPTrack/open_ptrack_v2/wiki/Manual-Ground-Plane
https://github.com/OpenPTrack/open_ptrack_v2/wiki/Supported-Hardware
https://github.com/OpenPTrack/open_ptrack_v2/wiki/Tested-Hardware
https://github.com/OpenPTrack/open_ptrack_v2/wiki/Deployment-Guide
https://github.com/OpenPTrack/open_ptrack_v2/wiki/Imager-Settings

## CALIBRATION
* MASTER: `roscore`
* MASTER: `roslaunch opt_calibration calibration_initializer.launch`
* SENSOR: `roslaunch opt_calibration listener.launch`

* MASTER: stop calibration_initializer.launch
* SENSOR: stop listener.launch

* MASTER: `roslaunch opt_calibration opt_calibration_master.launch`
* SENSOR: `roslaunch opt_calibration $(find src/open_ptrack/opt_calibration -type f -name sensor_realsense\*.launch -exec basename '{}' \;)`

##### show checkerboard to each sensor until all are added to the tree, then place checkerboard flat and run:
MASTER: `rostopic pub /opt_calibration/action std_msgs/String "save" -1`

##### CREATE ADDITIONAL FILES for multi sensor people tracking
* MASTER: stop opt_calibration_master.launch
* SENSOR: stop listener.launch

* MASTER: `roslaunch opt_calibration detection_initializer.launch`
* SENSOR: `roslaunch opt_calibration listener.launch`

### CALIBRATION REFINEMENT
* MASTER: `roslaunch opt_calibration opt_calibration_refinement.launch`
* SENSOR: `roslaunch detection $(find src/open_ptrack -type f -name detection_node_realsense\*.launch -exec basename '{}' \;)`

###### walk entire room cover all space more than once then run:
* MASTER: `rostopic pub /opt_calibration/action std_msgs/String "save" -1`

## START TRACKING
* MASTER: `roscore`
* MASTER: `roslaunch tracking tracking_node.launch `
* SENSOR: `roslaunch detection $(find src/open_ptrack -type f -name detection_node_realsense\*.launch -exec basename '{}' \;)`



