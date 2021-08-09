import copy
import glob
import os
import time
import sys
from collections import deque
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
from ndp.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr import algo, utils
from ndp.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.algo.ppo import PPODMP
from ndp.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.envs import make_vec_envs
from ndp.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.model import Policy, DMPPolicy
from ndp.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.storage import RolloutStorage, RolloutStorageDMP
import torch.distributions as td
from rad.kitchen_train import compute_path_info
from rlkit.core import logger as rlkit_logger

def train(actor_critic, agent, rollouts, envs, test_envs, device, variant):
    torch.set_num_threads(1)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    episode_rewards = deque(maxlen=50)

    start = time.time()
    num_env_steps = variant['num_env_steps']
    num_steps = variant['num_steps']
    num_processes = variant['num_processes']
    num_updates = int(num_env_steps) // num_steps // num_processes
    epoch_data = dict(distance_train_sample=[],
                      success_train_sample=[],
                      distance_train_det=[],
                      success_train_det=[],
                      distance_test_sample=[],
                      success_test_sample=[],
                      distance_test_det=[],
                      success_test_det=[])

    rollout_infos =  dict(final_distance=[], final_success_rate=[])
    log_interval = variant['eval_interval']
    use_linear_lr_decay = variant['use_linear_lr_decay']
    lr = variant['algorithm_kwargs']['lr']
    T = variant['dmp_kwargs']['T']
    save_interval = log_interval
    num_train_calls = 0
    total_train_expl_time = 0
    total_num_steps = 0
    for j in range(num_updates):
        epoch_start_time = time.time()
        train_expl_st = time.time()
        if use_linear_lr_decay:
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if algo == "acktr" else lr)
        envs.reset()
        all_infos_train = []
        for step in range(num_steps):
            if step % T == 0:
                with torch.no_grad():
                    values, actions, action_log_probs_list, recurrent_hidden_states_lst = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

                action = actions[step % T]
                action_log_probs = action_log_probs_list[step % T]
                recurrent_hidden_states = recurrent_hidden_states_lst[0]
                value = values[:, step % T].view(-1, 1)

            obs, reward, done, infos = envs.step(action)
            all_infos_train.append(infos)

            episode_rewards.append(reward[0].item())
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_probs, value, reward, masks, bad_masks)

            if all(done):
                obs = envs.reset()

        all_infos_traj = []
        for k in range(num_processes):
            ep_infos = []
            for z in range(num_steps):
                if 'bad_transition' in all_infos_train[z][k]:
                    del all_infos_train[z][k]['bad_transition']
                if 'terminal_observation' in all_infos_train[z][k]:
                    del all_infos_train[z][k]['terminal_observation']
                ep_infos.append(all_infos_train[z][k])
            all_infos_traj.append(ep_infos)
        all_infos_train = all_infos_traj
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()[:, 0].view(-1, 1)

        rollouts.compute_returns(next_value, **variant["rollout_kwargs"])


        value_loss, action_loss, dist_entropy, num_calls = agent.update(rollouts)
        num_train_calls += num_calls
        rollouts.after_update()


        success = [1.0*info['success'] for info in infos]
        dist = [info['distance'] for info in infos]
        rollout_infos['final_distance'].append(np.mean(dist))
        rollout_infos['final_success_rate'].append(np.mean(success))

        total_train_expl_time += time.time()-train_expl_st
        if j % log_interval == 0 and len(episode_rewards) > 1:
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            vec_norm = utils.get_vec_normalize(test_envs)
            vec_norm.eval()
            vec_norm.obs_rms = obs_rms

            obs = test_envs.reset()
            all_infos = []
            for step in range(num_steps):
                if step % T == 0:
                    with torch.no_grad():
                        values, actions, action_log_probs_list, recurrent_hidden_states_lst = actor_critic.act(
                            obs, rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step], deterministic=True)

                    action = actions[step % T]
                    action_log_probs = action_log_probs_list[step % T]
                    recurrent_hidden_states = recurrent_hidden_states_lst[0]
                    value = values[:, step % T].view(-1, 1)
                obs, reward, done, infos = test_envs.step(action)
                all_infos.append(infos)

                if all(done):
                    obs = test_envs.reset()
            all_infos_traj = []
            for k in range(5):
                ep_infos = []
                for z in range(num_steps):
                    if 'bad_transition' in all_infos[z][k]:
                        del all_infos[z][k]['bad_transition']
                    if 'terminal_observation' in all_infos[z][k]:
                        del all_infos[z][k]['terminal_observation']
                    ep_infos.append(all_infos[z][k])
                all_infos_traj.append(ep_infos)
            all_infos = all_infos_traj

            dist = [info['distance'] for info in infos]
            success = [1.0*info['success'] for info in infos]
            test_dist = np.mean(dist)
            test_sucess = np.mean(success)
            epoch_data['distance_test_det'].append(test_dist)
            epoch_data['success_test_det'].append(test_sucess)

            rlkit_logger.record_dict({"Average Returns": test_sucess}, prefix="evaluation/")
            rlkit_logger.record_dict({"Average Returns": np.mean(rollout_infos['final_success_rate'])}, prefix="exploration/")
            train_statistics = compute_path_info(all_infos_train)
            statistics = compute_path_info(all_infos)
            rlkit_logger.record_dict(statistics, prefix="evaluation/")
            rlkit_logger.record_dict(train_statistics, prefix="exploration/")
            rlkit_logger.record_tabular("time/total (s)", time.time() - start)
            rlkit_logger.record_tabular("exploration/num steps total", total_num_steps)
            rlkit_logger.record_tabular("trainer/num train calls", num_train_calls)
            rlkit_logger.record_tabular("Epoch", j // variant["eval_interval"])
            rlkit_logger.dump_tabular(with_prefix=False, with_timestamp=False)

            # obs = test_envs.reset()
            # for step in range(num_steps):
            #     if step % T == 0:
            #         with torch.no_grad():
            #             values, actions, action_log_probs_list, recurrent_hidden_states_lst = actor_critic.act(
            #                 obs, rollouts.recurrent_hidden_states[step],
            #                 rollouts.masks[step], deterministic=False)
            #         action = actions[step % T]
            #         action_log_probs = action_log_probs_list[step % T]
            #         recurrent_hidden_states = recurrent_hidden_states_lst[0]
            #         value = values[:, step % T].view(-1, 1)
            #     obs, reward, done, infos = test_envs.step(action)

            #     if all(done):
            #         obs = test_envs.reset()

            # dist = [info['distance'] for info in infos]
            # success = [1.0*info['success'] for info in infos]
            # test_dist = np.mean(dist)
            # test_sucess = np.mean(success)
            # epoch_data['distance_test_sample'].append(test_dist)
            # epoch_data['success_test_sample'].append(test_sucess)



            epoch_data['distance_train_sample'].append(np.mean(rollout_infos['final_distance']))
            epoch_data['success_train_sample'].append(np.mean(rollout_infos['final_success_rate']))
            end = time.time()
            num_epochs = j // log_interval
            total_num_steps = (j + 1) * num_processes * num_steps

            print( "Epochs {}, Updates {}, num timesteps {}, FPS {} \n  final distance {:0.4f}, final success {:0.2f}, final test distance {:0.4f}, final test success {:0.2f} \n"
                .format(num_epochs, j, total_num_steps, int(total_num_steps / (end - start)),
                    np.mean(rollout_infos['final_distance']), np.mean(rollout_infos['final_success_rate']), test_dist, test_sucess))
            sys.stdout.flush()

            rollout_infos =  dict(final_distance=[], final_success_rate=[])


        # if (j % save_interval == 0 or j == num_updates - 1) and save_dir != "":
        #     save_path = save_dir
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass
        #     torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'obs_rms', None),
        #                                             getattr(utils.get_vec_normalize(envs), 'ret_rms', None)],
        #                                                 os.path.join(save_path, "actor_critic.pt"))
        #     torch.save(agent.optimizer.state_dict(), os.path.join(save_path, "agent_optimizer.pt"))
        #     np.save(save_path + '/epoch_data.npy', epoch_data)
