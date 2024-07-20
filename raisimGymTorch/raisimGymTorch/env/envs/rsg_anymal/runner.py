from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.rsg_anymal import NormalSampler
from raisimGymTorch.env.bin.rsg_anymal import RaisimGymEnv
from raisimGymTorch.env.RewardAnalyzer import RewardAnalyzer
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse
import csv


# task specification
task_name = "anymal_locomotion"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
parser.add_argument('-c', '--capture', help='capure data', type=bool, default=False)
parser.add_argument('-cv', '--cvelocity', help='changing velocity', type=bool, default=False)
args = parser.parse_args()
mode = args.mode
weight_path = args.weight
capture_data = args.capture
changing_velocity = args.cvelocity

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
yml = YAML()
cfg = yml.load(open(task_path + "/cfg.yaml", 'r'))
import io
buf = io.BytesIO()
print(yml.dump(cfg['environment'], buf))
print(buf.getvalue())

# create environment from the configuration file
env = VecEnv(RaisimGymEnv(home_path + "/rsc", buf.getvalue()))
env.seed(cfg['seed'])

# shortcuts
ob_dim = env.num_obs + 1 if changing_velocity else env.num_obs # for target velocity
act_dim = env.num_acts
num_threads = cfg['environment']['num_threads']

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           1.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])
# tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              )

reward_analyzer = RewardAnalyzer(env, ppo.writer)

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)


real_pos_file_path = os.path.join(saver.data_dir, "pos_data.csv")
obs_file_path = os.path.join(saver.data_dir, "obs_data.csv")
action_file_path = os.path.join(saver.data_dir, "action_data.csv")

with open(obs_file_path, 'w', newline='') as obs_file, open(action_file_path, 'w', newline='') as action_file, open(real_pos_file_path, 'w', newline='') as position_file:
    obs_writer = csv.writer(obs_file)
    action_writer = csv.writer(action_file)
    position_writer = csv.writer(position_file)

    # Write headers
    if capture_data:
        obs_headers = ['update', 'step'] + [f'obs_{i}' for i in range(ob_dim)]
        action_headers = ['update', 'step'] + [f'action_{i}' for i in range(act_dim)]
        position_headers = ['update', 'step'] + \
            ['LF_HAA','LF_HFE','LF_KFE', 'RF_HAA','RF_HFE','RF_KFE', 'LH_HAA','LH_HFE','LH_KFE',  'RH_HAA','RH_HFE','RH_KFE'] + \
            [f'pos_{i}' for i in range(3)] + \
            [f'rot_{i}' for i in range(3)] + \
            ['LF_HAA_angVel','LF_HFE_angVel','LF_KFE_angVel', 'RF_HAA_angVel','RF_HFE_angVel','RF_KFE_angVel', 'LH_HAA_angVel','LH_HFE_angVel','LH_KFE_angVel',  'RH_HAA_angVel','RH_HFE_angVel','RH_KFE_angVel'] + \
            ['LF_HAA_force','LF_HFE_force','LF_KFE_force', 'RF_HAA_force','RF_HFE_force','RF_KFE_force', 'LH_HAA_force','LH_HFE_force','LH_KFE_force',  'RH_HAA_force','RH_HFE_force','RH_KFE_force']
        obs_writer.writerow(obs_headers)
        action_writer.writerow(action_headers)
        position_writer.writerow(position_headers)

    for update in range(1000000):
        start = time.time()
        env.reset()
        reward_sum = 0
        done_sum = 0
        average_dones = 0.

        if update % cfg['environment']['eval_every_n'] == 0:
            print("Visualizing and evaluating the current policy")
            torch.save({
                'actor_architecture_state_dict': actor.architecture.state_dict(),
                'actor_distribution_state_dict': actor.distribution.state_dict(),
                'critic_architecture_state_dict': critic.architecture.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
            }, saver.data_dir+"/full_"+str(update)+'.pt')
            # we create another graph just to demonstrate the save/load method
            loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
            loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

            env.turn_on_visualization()
            env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

            for step in range(1000):
                with torch.no_grad():
                    frame_start = time.time()
                    obs = env.observe(False)
                    position = env.getPosition()
                    orientation = env.getOrientation()
                    jointAngles = env.getJointAngles()
                    jointAngularVelocities = env.getJointAngularVelocities()
                    jointGeneralizedForces = env.getJointGeneralizedForces()
                    targetVelocity = env.getTargetVelocity()
                    if changing_velocity:
                        obs = np.hstack((obs, targetVelocity))
                    action = loaded_graph.architecture(torch.from_numpy(obs).cpu())
                    reward, dones = env.step(action.cpu().detach().numpy())
                    reward_analyzer.add_reward_info(env.get_reward_info())

                    # logging data
                    if capture_data:
                        obs_writer.writerow([update, step] + obs[0].tolist())
                        action_writer.writerow([update, step] + action[0].tolist())
                        position_writer.writerow(
                            [update, step] + 
                            jointAngles[0].tolist() + 
                            position[0].tolist() + 
                            orientation[0].tolist() +
                            jointAngularVelocities[0].tolist() + 
                            jointGeneralizedForces[0].tolist())

                    frame_end = time.time()
                    wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                    if wait_time > 0.:
                        time.sleep(wait_time)

            env.stop_video_recording()
            env.turn_off_visualization()

            reward_analyzer.analyze_and_plot(update)
            env.reset()
            env.save_scaling(saver.data_dir, str(update))

        # actual training
        for step in range(n_steps):
            obs = env.observe()
            position = env.getPosition()
            orientation = env.getOrientation()
            jointAngles = env.getJointAngles()
            jointAngularVelocities = env.getJointAngularVelocities()
            jointGeneralizedForces = env.getJointGeneralizedForces()
            targetVelocity = env.getTargetVelocity()
            if changing_velocity:
                obs = np.hstack((obs, targetVelocity))
            action = ppo.act(obs)
            reward, dones = env.step(action)
            ppo.step(value_obs=obs, rews=reward, dones=dones)
            done_sum = done_sum + np.sum(dones)
            reward_sum = reward_sum + np.sum(reward)

            # logging data
            if capture_data and update % cfg['environment']['eval_every_n'] != 0:
                obs_writer.writerow([update, step] + obs[0].tolist())
                action_writer.writerow([update, step] + action[0].tolist())
                position_writer.writerow(
                    [update, step] + 
                    jointAngles[0].tolist() + 
                    position[0].tolist() + 
                    orientation[0].tolist() +
                    jointAngularVelocities[0].tolist() + 
                    jointGeneralizedForces[0].tolist())
            
        # take st step to get value obs
        obs = env.observe()
        targetVelocity = env.getTargetVelocity()
        if changing_velocity:
            obs = np.hstack((obs, targetVelocity))
        ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
        average_ll_performance = reward_sum / total_steps
        average_dones = done_sum / total_steps
        avg_rewards.append(average_ll_performance)

        actor.update()
        actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to(device))

        # curriculum update. Implement it in Environment.hpp
        env.curriculum_callback()

        end = time.time()

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('{:<40} {:>6}'.format("Target Velocity: ", '{:0.2f}'.format(env.getTargetVelocity()[0,0])))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
        print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                        * cfg['environment']['control_dt'])))
        print('----------------------------------------------------\n')
