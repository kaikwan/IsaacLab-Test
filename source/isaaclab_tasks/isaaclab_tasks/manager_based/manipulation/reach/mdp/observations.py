from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def get_camera_data(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    type: str = "rgb"
) -> torch.Tensor:
  
    camera = env.scene[camera_cfg.name]
    return camera.data.output[type][0][..., :3]