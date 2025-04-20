# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import yaml

from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from scipy.spatial.transform import Rotation as R

##
# Pre-defined configs
##
from isaaclab_assets import UR16_HIGH_PD_CFG, UR16_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class UR10ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur10
        self.scene.robot = UR16_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = UR16_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # import pdb; pdb.set_trace()
        # self.scene.robot.init_state.joint_pos = {
        #     "shoulder_pan_joint": -0.61,
        #     "shoulder_lift_joint": -1.53589,
        #     "elbow_joint": 1.8326,
        #     "wrist_1_joint": -0.296706,
        #     "wrist_2_joint": 1.0,
        #     "wrist_3_joint": -1.5708,
        # }
        self.scene.robot.init_state.joint_pos = {
            "shoulder_pan_joint": -1.0252,
            "shoulder_lift_joint": -1.5089,
            "elbow_joint": 2.5458,
            "wrist_1_joint": -1.1429,
            "wrist_2_joint": 0.5115,
            "wrist_3_joint": -1.7506,
        }


        self.scene.robot.init_state.pos = (0, 0, 1.020)

        # Convert Euler angles to quaternion using scipy
        r = R.from_euler('xyz', [0, 180, 0], degrees=True)
        self.scene.robot.init_state.rot = tuple(float(x) for x in r.as_quat())

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot", joint_names=[".*"], 
            body_name="wrist_3_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        def add_cube(dims, pose, i):
            return RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Pod_{i+1:02d}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=pose[:3], rot=[1, 0, 0, 0]),
                spawn=sim_utils.CuboidCfg(
                    size=dims,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                ))
        
        with open('./big_collision_primitives_3d.yml') as file:
            world_params = yaml.load(file, Loader=yaml.FullLoader)
        if ('cube' in world_params['world_model']['coll_objs']):
            cube = world_params['world_model']['coll_objs']['cube']
        i = 0
        for obj in cube.keys():
            dims = cube[obj]['dims']
            pose = cube[obj]['pose']
            setattr(self.scene, "pod_prim" + str(i+1), add_cube(dims, pose, i))
            i += 1

        back_wall = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Pod_14",
                # w, x, y, z
                init_state=RigidObjectCfg.InitialStateCfg(pos=[1.0914, 0.0, 0.5], rot=[0.70711, 0.0, 0.0, 0.70711]),
                spawn=sim_utils.CuboidCfg(
                    size=[1, 0.001, 2.5],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        ))

        setattr(self.scene, "pod_prim" + str(14), back_wall)

@configclass
class UR10ReachEnvCfg_PLAY(UR10ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
