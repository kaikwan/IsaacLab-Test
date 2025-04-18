# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def object_ee_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return object_ee_distance


def object_idx(
    env: ManagerBasedRLEnv,
    object1_cfgs: SceneEntityCfg = SceneEntityCfg("object"),
    threshold: float = 0.075,
) -> torch.Tensor:
    """The index of the object that has distance within the threshold radius."""
    res = torch.tensor([[0]] * env.scene.num_envs).to(env.device)
    object_cfgs = [object1_cfgs]
    for object_cfg in object_cfgs:
        idx = 1
        ee_distance = object_ee_distance(env, object_cfg=object_cfg)
        close_mask = torch.where(ee_distance < threshold, 1, 0).bool().unsqueeze(1)
        res = torch.where((close_mask & (res == 0)), idx, res)

    # if torch.any(res != 0.0):
    #     print(f"unique object idx: {torch.unique(res)}")
    return res


# def object_idx(
#     env: ManagerBasedRLEnv,
#     object1_cfgs: SceneEntityCfg = SceneEntityCfg("object1"),
#     object2_cfgs: SceneEntityCfg = SceneEntityCfg("object2"),
#     threshold: float = 0.1,
# ) -> torch.Tensor:
#     """The index of the object that has distance within the threshold radius."""
#     res = torch.tensor([[0]] * env.scene.num_envs).to(env.device)
#     object_cfgs = [object1_cfgs, object2_cfgs]
#     for object_cfg in object_cfgs:
#         idx = int(object_cfg.name[-1])
#         ee_distance = object_ee_distance(env, object_cfg=object_cfg)
#         close_mask = torch.where(ee_distance < threshold, 1, 0).bool().unsqueeze(1)
#         res = torch.where((close_mask & (res == 0)), idx, res)

#     if torch.any(res != 0.0):
#         print(f"unique object idx: {torch.unique(res)}")
#     return res
