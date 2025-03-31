# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg

from isaaclab.utils import configclass


@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # resume = True
    # load_run = "2025-03-30_12-56-24"
    # load_checkpoint = "model_9850.pt"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 50
    experiment_name = "franka_lift"
    empirical_normalization = False
    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        rnn_num_layers=2,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    # logger = "wandb"
    # wandb_project = "franka_lift"