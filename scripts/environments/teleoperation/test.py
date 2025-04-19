import argparse
import os
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
import csv
import cv2
import numpy as np

import omni.log

from isaaclab.devices import Se3Gamepad, Se3HandTracking, Se3Keyboard, Se3SpaceMouse
from isaaclab.envs import ViewerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # 4) Define + create the camera config
    camera_cfg = CameraCfg(
        prim_path="/World/Origin_00/camera",
        update_period=0, 
        height=480, 
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=25.0, focus_distance=400.0, horizontal_aperture=20.955,# clipping_range=(0.05, 2.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.84, 0.994, 1.29855), rot=(0.41329, 0.29477, -0.4949, -0.70525), convention="opengl"),
    )
    camera = Camera(cfg=camera_cfg)

    env.reset()

    breakpoint()
    # 6) Now that replicator is active, reset the camera
    camera.reset()

    # 7) Reset your environment
    obs = env.reset()

    # 8) Simulation loop
    while sim.is_playing():
        # Step simulation (this runs physics and rendering).
        sim.step()
        # Ask camera to refresh its buffers
        camera.update(dt=sim.get_physics_dt())
        # Step your environment’s logic
        action = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)
        obs, rew, done, info = env.step(action)

        # Retrieve the latest image
        if "rgb" in camera.data.output:
            frame = camera.data.output["rgb"]
            # Do something with 'frame'...
        else:
            print("Camera data not ready yet.")

if __name__ == "__main__":
    main()
