# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR

##
# Configuration
##


UR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
"""Configuration of UR-10 arm using implicit actuator models."""


UR16_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/raymond/kaikwan_ws/IsaacLab-Test/ur16e_epick_flattened.usd", # f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur16e/ur16e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        # "arm": ImplicitActuatorCfg(
        #     joint_names_expr=[".*"],
        #     velocity_limit=100.0,
        #     effort_limit=87.0,
        #     stiffness=800.0,
        #     damping=40.0,
        # ),
        "ur_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "ur_elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)



UR16_HIGH_PD_CFG = UR16_CFG.copy()
UR16_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
UR16_HIGH_PD_CFG.actuators["ur_shoulder"].stiffness = 20000.0
UR16_HIGH_PD_CFG.actuators["ur_shoulder"].damping = 100.0
UR16_HIGH_PD_CFG.actuators["ur_elbow"].stiffness = 20000.0
UR16_HIGH_PD_CFG.actuators["ur_elbow"].damping = 100.0

"""Configuration of UR-16e arm using implicit actuator models."""