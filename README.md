# AA275-Final-Project

## Setup

1. Setup a ROS2 workspace with the ASL Turtlebot 3 utility packages by following the instructions [here](https://github.com/StanfordASL/asl-tb3-utils).

2. Create a new ROS2 workspace:

   ```bash
   # Create the directories
   mkdir -p ~/ros2_ws/src
   ```

3. Clone this repository:

    ```bash
    # Navigate to the src directory
    cd ~/ros2_ws/src

    # Clone the repo
    git clone https://github.com/abhyudit309/AA275-Final-Project
    ```

4. Build the code:

    ```bash
    cd ~/ros2_ws
    colcon build --symlink-install
    ```

5. Source the workspace:

    ```bash
    source ~/ros2_ws/install/setup.bash
    ```

## Run

1. Start ROS and Gazebo with a simulated TurtleBot in an arena world:

   ```bash
   ros2 launch asl_tb3_sim arena.launch.py
   ```

2. From another terminal, launch the localization node:

   ```bash
   source ~/ros2_ws/install/setup.bash
   ros2 launch motion localization_arena.launch.py
   ```
3. From another terminal, run the teleoperation node:

   ```bash
   source ~/ros2_ws/install/setup.bash
   ros2 run motion keyboard_teleop.py
   ```
