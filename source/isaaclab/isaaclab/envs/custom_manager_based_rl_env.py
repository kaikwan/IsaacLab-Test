# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar


from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager

from .manager_based_rl_env import ManagerBasedRLEnv
from .custom_manager_based_rl_env_cfg import CustomManagerBasedRLEnvCfg


class CustomManagerBasedRLEnv(ManagerBasedRLEnv):
    """The superclass for the manager-based workflow reinforcement learning-based environments."""


    def __init__(self, cfg: CustomManagerBasedRLEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.num_clutter_objects = cfg.num_clutter_objects
        self.position_dim = 3
        self.adversary_action = torch.zeros((self.num_envs, self.num_clutter_objects * self.position_dim)).to(self.device)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        self.adversarial_reset(env_ids)
        super()._reset_idx(env_ids)

    def adversarial_reset(
        self, reset_env_ids: Sequence[int]
    ) -> tuple[VecEnvObs, dict]:
        """Reset the environment.

        Returns:
            np.ndarray: The initial observation.
        """
        adversary_pos = self.adversary_action[reset_env_ids]
    
        for asset_name, rigid_object in self.scene._rigid_objects.items():
            if "clutter_object" not in asset_name:
                continue
            clutter_idx = int(asset_name.split("clutter_object")[-1]) - 1
            clutter_obj_state = rigid_object.data.default_root_state[reset_env_ids].clone() # get states of only envs we want to reset
            clutter_obj_state[:, 0:3] += self.scene.env_origins[reset_env_ids]
            root_pose = clutter_obj_state[:, :7]
            root_velocity = clutter_obj_state[:, 7:] * 0.0 # zero out the velocity
            root_pose[:, :3] += torch.stack([adversary_pos[:, clutter_idx*3] * 0.05,
                                             adversary_pos[:, clutter_idx*3+1] * 0.05,
                                             adversary_pos[:, clutter_idx*3+2] * 0.025 + 0.2]
                                             , dim=-1).to(root_pose.device)
            rigid_object.write_root_link_pose_to_sim(root_pose, env_ids=reset_env_ids)
            rigid_object.write_root_com_velocity_to_sim(root_velocity, env_ids=reset_env_ids)
        self.scene.write_data_to_sim()