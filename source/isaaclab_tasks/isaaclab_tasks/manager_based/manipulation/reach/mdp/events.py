
import torch
import omni.replicator.core as rep
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
# from omni.isaac.core.utils.extensions import enable_extension
import colorsys

def reset_cam(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # get default root state
    # root_states = asset.data.default_root_state[env_ids].clone()
    pos = torch.tensor([[-0.84, 0.994, 1.29855]])
    rot = torch.tensor([[0.41329, 0.29477, -0.4949, -0.70525]])
    root_states = torch.cat([pos, rot, torch.zeros(1, 7)], dim=-1).cuda()

    # velocities
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # poses
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]

    # solve for orientation as a (lookat)
    look_at = torch.tensor([[0.65, 0.05, 0.8]]).cuda()
    direction_vector = -(look_at - positions)
    direction_vector = direction_vector / torch.norm(direction_vector, dim=-1, keepdim=True)
    up_vector = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
    right_vector = torch.cross(up_vector, direction_vector)
    right_vector = right_vector / torch.norm(right_vector, dim=-1, keepdim=True)
    up_vector = torch.cross(direction_vector, right_vector)
    up_vector = up_vector / torch.norm(up_vector, dim=-1, keepdim=True)
    orientations = math_utils.quat_from_matrix(torch.stack([right_vector, up_vector, direction_vector], dim=-1))

    # set into the physics simulation
    asset.set_world_poses(positions, orientations, env_ids, convention="opengl")


# ------------------------------------------------------------
# GLOBAL‑LIGHT RANDOMIZER  (one dome / distant light per stage)
# ------------------------------------------------------------

def randomize_global_light(
    env,                         # ManagerBasedEnv
    env_ids,                     # (ignored – light is global)
    asset_cfg,                   # SceneEntityCfg("light")
    intensity_range=(300, 800),  # lumens
    hue_range=(0.0, 1.0),        # 0‑1  (HSV hue)
    pitch_range=(-15.0, 15.0),   # deg
    yaw_range=(-30.0, 30.0),     # deg
):
    # 1) make sure Replicator is live
    enable_extension("omni.replicator.core")
    import omni.replicator.core as rep

    # 2) absolute USD path of the **single** light prim
    prim_path = env.scene["light"].prim_path        #  -> "/World/light"

    # 3) sample scalars just once
    intens = float(torch.empty(1).uniform_(*intensity_range))
    hue    = float(torch.empty(1).uniform_(*hue_range))

    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)    # full‑sat/value
    pitch   = float(torch.empty(1).uniform_(*pitch_range))
    yaw     = float(torch.empty(1).uniform_(*yaw_range))

    # 4) write attributes through Replicator
    with rep.modify.light(prim_path) as light:
        light.intensity = intens
        light.color     = [r, g, b]
        light.rotation  = [pitch, yaw, 0.0]         # XYZ Euler, degrees

