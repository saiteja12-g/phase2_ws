# To Execute

catkin_make

source devel/setup.bash

roslaunch plan.launch

python3 planner.py -s 1 1 60 -g 10 10 -r 100 200 -c 0.1
