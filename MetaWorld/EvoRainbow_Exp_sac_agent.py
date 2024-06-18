import numpy as np
import torch
from models import flatten_mlp, Private_tanh_gaussian_actor
from replay_buffer import replay_buffer
from utils import get_action_info, reward_recorder
from datetime import datetime
import copy, os
"""
The sac is modified to train the sawyer environment

"""
from torch.nn import functional as F
import torch.nn as nn


from ES import sepCEM

import math
import random

# the soft-actor-critic agent
class sac_agent:
    def __init__(self, env, eval_env, args, our_wandb):
        self.our_wandb= our_wandb
        self.args = args
        self.env = env
        # create eval environment
        self.eval_env = eval_env
        # observation space
        # build up the network that will be used.
        self.total_eval_num = 0
        self.ea_better = 0
        self.pop = []

        for _ in range(args.pop_size):
            self.pop.append(Private_tanh_gaussian_actor(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.args.hidden_size,self.args.log_std_min, self.args.log_std_max))

        self.CEM = sepCEM(self.pop[0].get_size(), mu_init=self.pop[0].get_params(),
                          sigma_init=args.sigma_init, damp=args.damp,
                          damp_limit=args.damp_limit,
                          pop_size=args.pop_size, antithetic=not args.pop_size % 2, parents=args.pop_size // 2,
                          elitism=args.elitism)

        self.qf1 = flatten_mlp(self.env.observation_space.shape[0], self.args.hidden_size, self.env.action_space.shape[0])
        self.qf2 = flatten_mlp(self.env.observation_space.shape[0], self.args.hidden_size, self.env.action_space.shape[0])
        # set the target q functions
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        # build up the policy network

        self.actor_net = Private_tanh_gaussian_actor(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.args.hidden_size,self.args.log_std_min, self.args.log_std_max)
        # define the optimizer for them
        self.qf1_optim = torch.optim.Adam(self.qf1.parameters(), lr=self.args.q_lr)
        self.qf2_optim = torch.optim.Adam(self.qf2.parameters(), lr=self.args.q_lr)
        # the optimizer for the policy network
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=self.args.p_lr)
        # entorpy target
        self.target_entropy = -1 * self.env.action_space.shape[0]

        self.log_alpha = torch.zeros(1, requires_grad=True, device='cuda' if self.args.cuda else 'cpu')
        # define the optimizer
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.p_lr)
        # define the replay buffer
        self.buffer = replay_buffer(self.args.buffer_size)
        # get the action max
        self.action_max = self.env.action_space.high
        # if use cuda, put tensor onto the gpu
        if self.args.cuda:
            self.actor_net.cuda()
            self.qf1.cuda()
            self.qf2.cuda()
            self.target_qf1.cuda()
            self.target_qf2.cuda()
        # get the reward recorder and success recorder
        self.reward_recorder = reward_recorder(10)
        self.success_recorder = reward_recorder(10)
        # automatically create the folders to save models
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.previous_print = 0
        self.total_steps = 0
        self.ep_num = 0
        self.RL2EA = False

        self.rl_index = None

    def evluate(self, actor_net):
        ep_steps = 0
        ep_reward = 0
        done = False
        obs, _ = self.env.reset()
        while not done:
            ep_steps += 1
            with torch.no_grad():
                obs_tensor = self._get_tensor_inputs(obs)
                pi = actor_net(obs_tensor)
                action = get_action_info(pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                action = action.cpu().numpy()[0]
            # input the actions into the environment
            obs_, reward, done, info = self.env.step(self.action_max * action)
            self.reward_recorder.add_rewards(reward)
            ep_reward += reward
            # store the samples
            self.buffer.add(obs, action, reward, obs_, float(done))
            # reassign the observations
            obs = obs_
            if done:
                self.reward_recorder.start_new_episode()
                self.success_recorder.add_rewards(info['success'])
                self.success_recorder.start_new_episode()

        return ep_reward, ep_steps



    # train the agent
    def learn(self):
        global_timesteps = 0
        # before the official training, do the initial exploration to add episodes into the replay buffer
        self._initial_exploration(exploration_policy=self.args.init_exploration_policy) 
        # reset the environment
        while self.total_steps  < self.args.total_timesteps:

            es_params = self.CEM.ask(self.args.pop_size)
            if not self.RL2EA:
                for i in range(self.args.pop_size):
                    self.pop[i].set_params(es_params[i])
            else:
                for i in range(self.args.pop_size):
                    if i != self.rl_index:
                        self.pop[i].set_params(es_params[i])
                    else:
                        es_params[i] = self.pop[i].get_params()
                        # self.pop[i].actor.set_params(es_params[i])
            self.RL2EA = False

            total_ep_reward = 0
            fitness = np.zeros(len(self.pop))
            for index, actor_net in enumerate(self.pop):
                ep_reward, ep_steps = self.evluate(actor_net)
                total_ep_reward +=ep_steps
                fitness[index] += ep_reward


            print("Fitness", fitness)
            self.CEM.tell(es_params, fitness)

            # start to collect samples
            ep_reward, ep_steps = self.evluate(self.actor_net)
            total_ep_reward += ep_steps

            self.total_steps += total_ep_reward


            best_index = np.argmax(fitness)
            print("best index ", best_index, np.max(fitness), " RL index ", self.rl_index,
                  fitness[self.rl_index])
            if self.args.EA_tau > 0.0:
                # perform soft update
                for param, target_param in zip(self.pop[best_index].parameters(),self.actor_net.parameters()):
                    target_param.data.copy_(self.args.EA_tau * param.data + (1 - self.args.EA_tau) * target_param.data)

            if self.total_steps - self.previous_print > self.args.display_interval:
                # start to do the evaluation
                EA_mean_rewards, EA_mean_success = self._evaluate_agent(self.pop[np.argmax(fitness)])

                self.our_wandb.log(
                    {'EA_Rewards': EA_mean_rewards, 'EA_Success': EA_mean_success,'time_steps': self.total_steps})
                print('[{}] Frames: {}, EA Rewards: {:.3f}, Success: {:.3f}'.format(
                        datetime.now(), \
                        self.total_steps, EA_mean_rewards, EA_mean_success))

            #print("current ", self.total_steps ,  self.ep_num , ep_reward, info['success'])
            # after collect the samples, start to update the network
            for _ in range(self.args.update_cycles * total_ep_reward):
                qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss = self._update_newtork(self.pop + [self.actor_net])
                # update the target network
                if global_timesteps % self.args.target_update_interval == 0:
                    self._update_target_network(self.target_qf1, self.qf1)
                    self._update_target_network(self.target_qf2, self.qf2)
                global_timesteps += 1

            # Replace any index different from the new elite
            replace_index = np.argmin(fitness)
            # if replace_index == elite_index:
            #     replace_index = (replace_index + 1) % len(self.pop)
            self.rl_to_evo(self.actor_net, self.pop[replace_index])
            self.RL2EA = True
            self.rl_index = replace_index
            # self.evolver.rl_policy = replace_index
            print('Sync from RL --> Nevo')


            if self.total_steps - self.previous_print >  self.args.display_interval:
                self.previous_print = self.total_steps
                # start to do the evaluation

                mean_rewards, mean_success = self._evaluate_agent(self.actor_net)

                self.total_eval_num +=1.0
                if mean_success > EA_mean_success:
                    self.ea_better +=1.0

                self.our_wandb.log(
                {'EA_better_ratio':self.ea_better/self.total_eval_num ,'Rewards': np.max([mean_rewards, EA_mean_rewards]), 'Success': np.max([EA_mean_success, mean_success]),  'RL_Rewards': mean_rewards, 'RL_Success': mean_success, 'T_Reward': self.reward_recorder.mean, 'Q_loss': qf1_loss ,  'Actor_loss': actor_loss, 'Alpha_loss':alpha_loss, 'Alpha':alpha, 'time_steps': self.total_steps })
                print('[{}] Frames: {}, RL ewards: {:.3f}, Success: {:.3f}, T_Reward: {:.3f}, QF1: {:.3f}, QF2: {:.3f}, AL: {:.3f}, Alpha: {:.3f}, AlphaL: {:.3f}'.format(datetime.now(), \
                        self.total_steps , mean_rewards, mean_success, self.reward_recorder.mean, qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss))

                torch.save(self.actor_net.state_dict(), self.model_path + '/model.pt' if self.args.random_init else self.model_path + '/fixed_model.pt')
                if mean_success == 1:
                    torch.save(self.actor_net.state_dict(), self.model_path + '/best_model.pt' if self.args.random_init else self.model_path + '/fixed_best_model.pt')

    def rl_to_evo(self, rl_agent, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_agent.parameters()):
            target_param.data.copy_(param.data)

    # do the initial exploration by using the uniform policy
    def _initial_exploration(self, exploration_policy='gaussian'):
        # get the action information of the environment
        obs,_ = self.env.reset()
        for _ in range(self.args.init_exploration_steps):
            if exploration_policy == 'uniform':
                action = np.random.uniform(-1, 1, (self.env.action_space.shape[0], ))
            elif exploration_policy == 'gaussian':
                # the sac does not need normalize?
                with torch.no_grad():
                    obs_tensor = self._get_tensor_inputs(obs)
                    # generate the policy
                    pi = self.actor_net(obs_tensor)
                    action = get_action_info(pi).select_actions(reparameterize=False)
                    action = action.cpu().numpy()[0]
            # input the action input the environment
            obs_, reward, done, _ = self.env.step(self.action_max * action)
            # store the episodes
            self.buffer.add(obs, action, reward, obs_, float(done))
            obs = obs_
            if done:
                # if done, reset the environment
                obs, _ = self.env.reset()
        print("Initial exploration has been finished!")
    # get tensors
    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(0)
        return obs_tensor
    
    # update the network
    def _update_newtork(self, all_actors):
        # smaple batch of samples from the replay buffer
        obses, actions, rewards, obses_, dones = self.buffer.sample(self.args.batch_size)
        # preprocessing the data into the tensors, will support GPU later
        obses = torch.tensor(obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        actions = torch.tensor(actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        rewards = torch.tensor(rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        obses_ = torch.tensor(obses_, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = torch.tensor(1 - dones, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        # start to update the actor network
        pis = self.actor_net(obses)
        actions_info = get_action_info(pis, cuda=self.args.cuda)
        actions_, pre_tanh_value = actions_info.select_actions(reparameterize=True)
        log_prob = actions_info.get_log_prob(actions_, pre_tanh_value)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        # use the automatically tuning
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        # get the param
        alpha = self.log_alpha.exp()
        # get the q_value for new actions
        q_actions_ = torch.min(self.qf1(obses, actions_), self.qf2(obses, actions_))
        actor_loss = (alpha * log_prob - q_actions_).mean()


        # q value function loss
        q1_value = self.qf1(obses, actions)
        q2_value = self.qf2(obses, actions)
        with torch.no_grad():

            pis_next = self.actor_net(obses_)
            actions_info_next = get_action_info(pis_next, cuda=self.args.cuda)
            actions_next_, pre_tanh_value_next = actions_info_next.select_actions(reparameterize=True)
            log_prob_next = actions_info_next.get_log_prob(actions_next_, pre_tanh_value_next)
            target_q_value_next = torch.min(self.target_qf1(obses_, actions_next_), self.target_qf2(obses_, actions_next_)) - alpha * log_prob_next
            target_q_value = self.args.reward_scale * rewards + inverse_dones * self.args.gamma * target_q_value_next 
        qf1_loss = (q1_value - target_q_value).pow(2).mean()
        qf2_loss = (q2_value - target_q_value).pow(2).mean()
        # qf1
        self.qf1_optim.zero_grad()
        qf1_loss.backward()
        self.qf1_optim.step()
        # qf2
        self.qf2_optim.zero_grad()
        qf2_loss.backward()
        self.qf2_optim.step()
        # policy loss

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return qf1_loss.item(), qf2_loss.item(), actor_loss.item(), alpha.item(), alpha_loss.item()
    
    # update the target network
    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    # evaluate the agent
    def _evaluate_agent(self, actor_net):
        total_reward = 0
        total_success = 0
        for _ in range(self.args.eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0 
            success_flag = False
            while True:
                with torch.no_grad():
                    obs_tensor = self._get_tensor_inputs(obs)

                    pi = actor_net(obs_tensor)
                    action = get_action_info(pi, cuda=self.args.cuda).select_actions(exploration=False, reparameterize=False)
                    action = action.detach().cpu().numpy()[0]
                # input the action into the environment
                obs_, reward, done, info = self.eval_env.step(self.action_max * action)
                episode_reward += reward
                success_flag = success_flag or info['success']
                if done:
                    break
                obs = obs_
            total_reward += episode_reward
            total_success += 1 if success_flag else 0
        return total_reward / self.args.eval_episodes, total_success / self.args.eval_episodes
