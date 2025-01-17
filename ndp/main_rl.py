
import numpy as np
from ndp.arguments import *
import copy
import glob
import os
import time
from collections import deque
from datetime import datetime
import torch.nn.functional as F
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf

from ndp.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr import algo, utils
from ndp.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.envs import make_vec_envs
from ndp.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.model import Policy, DMPPolicy
from ndp.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.storage import RolloutStorage, RolloutStorageDMP
from ndp.pytorch_a2c_ppo_acktr_gail.ppo_train import train as train_ppo
from ndp.pytorch_a2c_ppo_acktr_gail.dmp_train import train as train_dmp
from ndp.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.algo.ppo import PPODMP

from rlkit.core import logger as rlkit_logger
from rlkit.core.eval_util import create_stats_ordered_dict

def dmp_experiment(variant):
    env_name = variant["env_name"]
    env_suite = variant["env_suite"]
    env_kwargs = variant["env_kwargs"]
    seed = variant["seed"]

    log_dir = os.path.expanduser(rlkit_logger.get_snapshot_dir())
    utils.cleanup_log_dir(log_dir)

    device = torch.device("cpu")

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(1)

    envs = make_vec_envs(
        env_suite,
        env_name,
        env_kwargs,
        seed,
        variant["num_processes"],
        variant["rollout_kwargs"]["gamma"],
        rlkit_logger.get_snapshot_dir(),
        device,
        False,
    )

    test_envs = make_vec_envs(
        env_suite,
        env_name,
        env_kwargs,
        seed,
        5,
        None,
        rlkit_logger.get_snapshot_dir(),
        device,
        False,
    )

    dmp_kwargs = variant['dmp_kwargs']
    dmp_kwargs['l'] = variant['num_int_steps'] // dmp_kwargs['T'] + 1

    actor_critic = DMPPolicy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs=dmp_kwargs,
        )
    actor_critic.to(device)

    agent = PPODMP(actor_critic, **variant["algorithm_kwargs"])

    rollouts = RolloutStorageDMP(
        variant["num_steps"],
        variant["num_processes"],
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
        T=dmp_kwargs['T'],
    )

    train_dmp(actor_critic, agent, rollouts, envs, test_envs, device, variant)


def ppo_experiment(args):


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if args.cuda else "cpu")


    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_kwargs = dict()

    env_kwargs['timestep'] = args.timestep

    if "push" in args.env_name:
        env_kwargs['params'] = 'random_goal_unconstrained'

    if "soccer" in args.env_name:
        env_kwargs['params'] = 'random_goal_unconstrained'

    if "faucet" in args.env_name:
        secondary_output = True

    env_kwargs=dict(
            dense=False,
            image_obs=False,
            action_scale=1,
            control_mode="end_effector",
            frame_skip=40,
            target_mode=False,
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=False,
                max_path_length=280,
            ),
            image_kwargs=dict(),
        )
    env_name = "kettle"
    test_envs = make_vec_envs(
        'kitchen',
        env_name,
        env_kwargs,
        args.seed,
        5,
        None,
        args.log_dir,
        device,
        False,
    )
    envs = make_vec_envs(
        'kitchen',
        env_name,
        env_kwargs,
        args.seed,
        args.num_processes,
        args.gamma,
        args.log_dir,
        device,
        False,
    )


    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                          args.gamma, args.log_dir, device, False, env_kwargs=env_kwargs)

    # test_envs = make_vec_envs(args.env_name, args.seed + args.num_processes, args.num_processes,
    #                          None, args.log_dir, device, False, env_kwargs=env_kwargs)

    actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy, 'hidden_size':args.hidden_size, 'hidden_activation':'relu'})
    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)



    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    train_ppo(actor_critic, agent, rollouts, envs, test_envs, args)

if __name__ == '__main__':
    args = get_args_ppo()
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    exp_id = args.expID
    if args.name:
        args.save_dir = path + '/data/' + args.name + '/' + str('{:05d}'.format(exp_id)) + '_' + args.type + '_' + args.env_name + '_s'
    else:
        args.save_dir = path + '/data/' + str('{:05d}'.format(exp_id)) + '_' + args.type + '_' + args.env_name + '_s'
    os.environ["OPENAI_LOGDIR"] = args.save_dir + '/tmp/'
    args.log_dir = args.save_dir + '/tmp/'
    args.num_env_steps = 25000 * args.num_processes * args.num_steps

    args.save_dir += '_T_' + str(args.T) + '_N_' + str(args.N) + '_rd_' + str(args.reward_delay)+ '_az_' + str(args.a_z) + '_cp_' + str(args.clip_param) + '_hs_' + str(args.hidden_size)

    os.makedirs(args.save_dir, exist_ok=True)

    if args.type == 'dmp':
            env_name = args.env_name
            args.env_name += '_pos'
            args.goal_type = 'int_path'
            dmp_experiment(args)

    if args.type == 'ppo':
            env_name = args.env_name
            args.env_name += '_pos'
            ppo_experiment(args)

    if args.type == 'ppo-multi':
            env_name = args.env_name
            args.env_name += '_pos'
            args.goal_type = 'multi_act'
            dmp_experiment(args)
