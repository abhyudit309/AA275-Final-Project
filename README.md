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
    cd ~/tb_ws
    colcon build --symlink-install
    ```

5. Source the workspace:

    ```bash
    source ~/ros2_ws/install/setup.bash
    ```
