# WarpSAC Sim-to-Sim and Sim-to-Real Deployment for Unitree G1

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)


## Overview

This project provides a set of reinforcement learning environments for Unitree robots, built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab) and [Unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab).

> [!WARNING]
> Always complete the Sim-to-Sim test before proceeding to Sim-to-Real. During
> the initial Sim-to-Real test, secure the Unitree G1 with its safety harness
> and have an operator support the robot by hand. Do not release the robot until
> you have confirmed that the system is operating correctly and safely.

Currently supports Unitree  **G1-29dof** robots.

<div align="center">

| <div align="center"> Sim-to-Sim </div> | <div align="center">  |  <div align="center"> Sim-to-Real </div> |
|--- | --- | --- |
| <video src="https://github.com/user-attachments/assets/553dccdc-860e-4828-a325-ec76d8e5b7db" controls muted loop playsinline width="320"></video><br>[Open Sim-to-Sim video](https://github.com/user-attachments/assets/553dccdc-860e-4828-a325-ec76d8e5b7db) | | <video src="https://github.com/user-attachments/assets/0dccd420-ed34-48cd-95bf-ebf9b65c6350" controls muted loop playsinline width="320"></video><br>[Open Sim-to-Real video](https://github.com/user-attachments/assets/0dccd420-ed34-48cd-95bf-ebf9b65c6350) |

</div>


## 1. Sim-to-Sim

### 1.1 Deploy the Controller

WarpSAC uses a C++ controller for Sim-to-Sim deployment. The controller can be
deployed directly on the server used to control the robot.

```bash
# Install dependencies.
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

# Install unitree_sdk2.
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF  # Install to /usr/local.
sudo make install

# Build the robot controller.
cd /home/uas/unitree_rl_lab/deploy/robots/g1_29dof  # Adjust for other robots.
mkdir build && cd build
cmake .. && make
```

### 1.2 Set Up the Simulator

First, install `unitree_sdk2_python`:

```bash
cd ~
sudo apt install python3-pip
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```

Then install MuJoCo and Pygame:

```bash
pip3 install mujoco
pip3 install pygame
```

After installation, update the relevant settings in
`unitree_mujoco/simulate_python/config.py`:

```python
ROBOT = "g1"                 # Robot model.
DOMAIN_ID = 0                # Must match the controller's DDS domain ID.
INTERFACE = "lo"             # Adjust this for your network interface.
USE_JOYSTICK = 1
ENABLE_ELASTIC_BAND = True
```

Open two terminals and run the following commands separately.

In the first terminal, start the Python simulator:

```bash
cd /home/uas/unitree_rl_lab/unitree_mujoco/simulate_python
/home/uas/anaconda3/envs/unitree_isaac45/bin/python unitree_mujoco.py
```

In the second terminal, start the controller:

```bash
cd /home/uas/unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl -n lo
```

The original Sim-to-Sim setup uses an Xbox controller. This version uses
keyboard input instead. Once the keyboard controls have been configured, begin
the test as described below.

### 1.3 Run the Sim-to-Sim Test

1. Click the MuJoCo window to give it keyboard focus.
2. Press `8` repeatedly. Each press increases the harness length by `0.1 m`,
   gradually lowering the robot. Press `7` repeatedly to shorten the harness
   and raise the robot.
3. Click the `g1_ctrl` terminal and press `1` to enter `FixStand` mode. The robot
   may shake at this stage; this is expected.
4. In the `g1_ctrl` terminal, press `2` to start the policy. The robot should
   stop shaking.
5. Return to the MuJoCo window and press `9` to disable the harness completely.
6. Return to the `g1_ctrl` terminal. Use `W`, `A`, `S`, and `D` to move the
   robot, `X` to stop it, and `0` to exit the controller.

## 2. Sim-to-Real

Connect to the G1 remotely and use `ping` to verify network connectivity. Then
open a terminal on the control server and start the controller:

```bash
# Replace <network-interface> with the appropriate interface, such as eth0.
./g1_ctrl --network <network-interface>
```

1. In the `g1_ctrl` terminal, press `1` to enter `FixStand` mode.
2. Press `2` to start the policy. The robot should stop shaking.
3. Use `W`, `A`, `S`, and `D` to move the robot, `X` to stop it, and `0` to exit
   the controller.

## Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.
- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [robot_lab](https://github.com/fan-ziqi/robot_lab): Referenced for project structure and parts of the implementation.
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): Versatile humanoid control framework for motion tracking.
