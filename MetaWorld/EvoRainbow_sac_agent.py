import numpy as np
import torch
from models import flatten_mlp, Shared_tanh_gaussian_actor
from replay_buffer import replay_buffer
from utils import get_action_info, reward_recorder
from datetime import datetime
import copy, os
"""
The sac is modified to train the sawyer environment

"""
from torch.nn import functional as F
import torch.nn as nn


class Policy_Value_Network(nn.Module):

    def __init__(self, state_dim, action_dim, representation_dim, cuda):
        super(Policy_Value_Network, self).__init__()

        self.ls = 256
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.representation_dim = representation_dim

        self.policy_size = self.ls * self.action_dim + self.action_dim

        l1 = 256
        l2 = 256
        l3 = l2
        self.l1 = l1
        # Construct input interface (Hidden Layer 1)
        self.policy_w_l1 = nn.Linear(self.ls + 1, self.representation_dim)
        self.policy_w_l2 = nn.Linear(self.representation_dim, self.representation_dim)
        self.policy_w_l3 = nn.Linear(self.representation_dim, self.representation_dim)

        self.policy_b_l1 = nn.Linear(self.ls + 1, self.representation_dim)
        self.policy_b_l2 = nn.Linear(self.representation_dim, self.representation_dim)
        self.policy_b_l3 = nn.Linear(self.representation_dim, self.representation_dim)

        # self.w_state_l1 = nn.Linear(args.state_dim, l1)
        # self.w_action_l1 = nn.Linear(args.action_dim, l1)

        input_dim = self.state_dim + self.action_dim


        self.w_l1 = nn.Linear(input_dim + self.representation_dim * 2, l1)
        # Hidden Layer 2

        self.w_l2 = nn.Linear(l1, l2)

        # Out
        self.w_out = nn.Linear(l3, 1)
        self.w_out.weight.data.mul_(0.1)
        self.w_out.bias.data.mul_(0.1)

        self.policy_w_l4 = nn.Linear(self.ls + 1, self.representation_dim)
        self.policy_w_l5 = nn.Linear(self.representation_dim, self.representation_dim)
        self.policy_w_l6 = nn.Linear(self.representation_dim, self.representation_dim)

        self.policy_b_l4 = nn.Linear(self.ls + 1, self.representation_dim)
        self.policy_b_l5 = nn.Linear(self.representation_dim, self.representation_dim)
        self.policy_b_l6 = nn.Linear(self.representation_dim, self.representation_dim)
        # self.w_state_l1 = nn.Linear(args.state_dim, l1)
        # self.w_action_l1 = nn.Linear(args.action_dim, l1)
        self.w_l3 = nn.Linear(input_dim + self.representation_dim * 2, l1)
        # Hidden Layer 2

        self.w_l4 = nn.Linear(l1, l2)

        # Out
        self.w_out_2 = nn.Linear(l3, 1)
        self.w_out_2.weight.data.mul_(0.1)
        self.w_out_2.bias.data.mul_(0.1)

        self.to('cuda' if cuda else 'cpu')

    def forward(self, input, param_w, param_b):
        reshape_param_w = param_w.reshape([-1, self.ls + 1])
        reshape_param_b = param_b.reshape([-1, self.ls + 1])

        out_p = F.leaky_relu(self.policy_w_l1(reshape_param_w))
        out_p = F.leaky_relu(self.policy_w_l2(out_p))
        out_p = self.policy_w_l3(out_p)
        out_p = out_p.reshape([-1, self.action_dim, self.representation_dim])

        out_b = F.leaky_relu(self.policy_b_l1(reshape_param_b))
        out_b = F.leaky_relu(self.policy_b_l2(out_b))
        out_b = self.policy_b_l3(out_b)
        out_b = out_b.reshape([-1, self.action_dim, self.representation_dim])

        out_p = torch.cat([out_b, out_p], -1)

        out_p = torch.mean(out_p, dim=1)

        # Hidden Layer 1 (Input Interface)
        concat_input = torch.cat((input, out_p), 1)

        # Hidden Layer 2
        out = self.w_l1(concat_input)

        out = F.leaky_relu(out)
        out = self.w_l2(out)
        out = F.leaky_relu(out)

        # Output interface
        out_1 = self.w_out(out)

        out_p = F.leaky_relu(self.policy_w_l4(reshape_param_w))
        out_p = F.leaky_relu(self.policy_w_l5(out_p))
        out_p = self.policy_w_l6(out_p)
        out_p = out_p.reshape([-1, self.action_dim, self.representation_dim])

        out_b = F.leaky_relu(self.policy_b_l4(reshape_param_b))
        out_b = F.leaky_relu(self.policy_b_l5(out_b))
        out_b = self.policy_b_l6(out_b)
        out_b = out_b.reshape([-1, self.action_dim, self.representation_dim])

        out_p = torch.cat([out_b, out_p], -1)
        out_p = torch.mean(out_p, dim=1)
        # Hidden Layer 1 (Input Interface)
        concat_input = torch.cat((input, out_p), 1)

        # Hidden Layer 2
        out = self.w_l3(concat_input)

        out = F.leaky_relu(out)
        out = self.w_l4(out)
        out = F.leaky_relu(out)

        # Output interface
        out_2 = self.w_out_2(out)

        return out_1, out_2

from ES import sepCEM

class shared_state_embedding(nn.Module):
    def __init__(self, state_dim, cuda):
        super(shared_state_embedding, self).__init__()
        self.state_dim = state_dim
        l1 = l2 = l3 = 256

        # Construct Hidden Layer 1
        self.w_l1 = nn.Linear(self.state_dim, l1)
        # Hidden Layer 2
        self.w_l2 = nn.Linear(l1, l2)

        self.w_l3 = nn.Linear(l2, l3)
        # Init
        self.to('cuda' if cuda else 'cpu')

    def forward(self, state):
        # Hidden Layer 1
        out = self.w_l1(state)
        out = out.relu()

        # Hidden Layer 2
        out = self.w_l2(out)
        out = out.relu()

        out = self.w_l3(out)
        out = out.relu()

        return out

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
            self.pop.append(Shared_tanh_gaussian_actor(self.env.action_space.shape[0], self.args.hidden_size,self.args.log_std_min, self.args.log_std_max))

        self.CEM = sepCEM(self.pop[0].get_size(), mu_init=self.pop[0].get_params(),
                          sigma_init=args.sigma_init, damp=args.damp,
                          damp_limit=args.damp_limit,
                          pop_size=args.pop_size, antithetic=not args.pop_size % 2, parents=args.pop_size // 2,
                          elitism=args.elitism)

        self.Mu_agent = Shared_tanh_gaussian_actor(self.env.action_space.shape[0], self.args.hidden_size,self.args.log_std_min, self.args.log_std_max)

        self.PVN = Policy_Value_Network(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.args.policy_representation_dim, self.args.cuda)
        self.target_PVN = copy.deepcopy(self.PVN)
        self.pvn_optim = torch.optim.Adam(self.PVN.parameters(), lr=self.args.q_lr)

        self.qf1 = flatten_mlp(self.env.observation_space.shape[0], self.args.hidden_size, self.env.action_space.shape[0])
        self.qf2 = flatten_mlp(self.env.observation_space.shape[0], self.args.hidden_size, self.env.action_space.shape[0])
        # set the target q functions
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        # build up the policy network

        self.state_embedding = shared_state_embedding(self.env.observation_space.shape[0], self.args.cuda)
        self.actor_net = Shared_tanh_gaussian_actor(self.env.action_space.shape[0], self.args.hidden_size,self.args.log_std_min, self.args.log_std_max)
        # define the optimizer for them
        self.qf1_optim = torch.optim.Adam(self.qf1.parameters(), lr=self.args.q_lr)
        self.qf2_optim = torch.optim.Adam(self.qf2.parameters(), lr=self.args.q_lr)
        # the optimizer for the policy network
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=self.args.p_lr)
        self.state_embedding_optimizer = torch.optim.Adam(self.state_embedding.parameters(), lr=self.args.p_lr)
        # entorpy target
        self.target_entropy = -1 * self.env.action_space.shape[0]

        #print("????", self.args.cuda )

        self.log_alpha = torch.zeros(1, requires_grad=True, device='cuda' if self.args.cuda else 'cpu')
        # define the optimizer
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.p_lr)
        # define the replay buffer
        self.buffer = replay_buffer(self.args.buffer_size)
        # get the action max
        self.action_max = self.env.action_space.high
        # if use cuda, put tensor onto the gpu
        if self.args.cuda:
            self.state_embedding.cuda()
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
                s_z = self.state_embedding.forward(obs_tensor)
                pi = actor_net(s_z)
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



    def evluate_with_critic(self, actor_net):
        ep_steps = 0
        ep_reward = 0
        done = False
        n_step_discount_reward = 0.0
        obs,_ = self.env.reset()

        while not done:
            ep_steps += 1
            with torch.no_grad():
                obs_tensor = self._get_tensor_inputs(obs)
                s_z = self.state_embedding.forward(obs_tensor)
                pi = actor_net(s_z)
                action = get_action_info(pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                action = action.cpu().numpy()[0]

            # input the actions into the environment
            obs_, reward, done, info = self.env.step(self.action_max * action)
            n_step_discount_reward += math.pow(0.99, ep_steps) * reward
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


            if self.args.H <= ep_steps:
                next_state = torch.FloatTensor(np.array([obs_])).to('cuda' if self.args.cuda else 'cpu')
                next_s_z = self.state_embedding.forward(next_state)
                next_pi = actor_net(next_s_z)
                next_action = get_action_info(next_pi, cuda=self.args.cuda).select_actions(reparameterize=False)

                next_Q1 = self.qf1(next_state, next_action)
                next_state_Q = next_Q1.cpu().data.numpy().flatten()
                n_step_discount_reward += math.pow(0.99, ep_steps) * next_state_Q[0]
                break

        return n_step_discount_reward, ep_steps

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
                random_num_num = random.random()
                if random_num_num< self.args.theta:
                    ep_reward, ep_steps = self.evluate(actor_net)
                    total_ep_reward +=ep_steps
                    fitness[index] += ep_reward
                else :
                    ep_reward, ep_steps = self.evluate_with_critic(actor_net)
                    total_ep_reward += ep_steps
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

                self.Mu_agent.set_params(self.CEM.mu)
                Mu_mean_rewards, Mu_mean_success = self._evaluate_agent(self.Mu_agent)

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
                    self._update_target_network(self.target_PVN, self.PVN)
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
                {'EA_better_ratio':self.ea_better/self.total_eval_num ,'Rewards': np.max([mean_rewards, EA_mean_rewards, Mu_mean_rewards]), 'Success': np.max([EA_mean_success, mean_success, Mu_mean_success]),  'RL_Rewards': mean_rewards, 'RL_Success': mean_success, 'T_Reward': self.reward_recorder.mean, 'Q_loss': qf1_loss ,  'Actor_loss': actor_loss, 'Alpha_loss':alpha_loss, 'Alpha':alpha, 'time_steps': self.total_steps })
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
                    s_z = self.state_embedding.forward(obs_tensor)
                    pi = self.actor_net(s_z)
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
        s_z = self.state_embedding.forward(obses)
        pis = self.actor_net(s_z)
        actions_info = get_action_info(pis, cuda=self.args.cuda)
        actions_, pre_tanh_value = actions_info.select_actions(reparameterize=True)
        log_prob = actions_info.get_log_prob(actions_, pre_tanh_value)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        index = random.sample(list(range(self.args.pop_size + 1)), self.args.K)

        EA_actions_list = []
        EA_log_prob_list = []
        for ind in index:
            actor = all_actors[ind]
            EA_pis = actor(s_z)
            EA_actions_info = get_action_info(EA_pis, cuda=self.args.cuda)
            EA_actions_, EA_pre_tanh_value = EA_actions_info.select_actions(reparameterize=True)
            EA_log_prob = EA_actions_info.get_log_prob(EA_actions_, EA_pre_tanh_value)
            alpha_loss += -(self.log_alpha * (EA_log_prob + self.target_entropy).detach()).mean()
            EA_actions_list.append(EA_actions_)
            EA_log_prob_list.append(EA_log_prob)
            # use the automatically tuning
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        # get the param
        alpha = self.log_alpha.exp()
        # get the q_value for new actions
        q_actions_ = torch.min(self.qf1(obses, actions_), self.qf2(obses, actions_))
        actor_loss = (alpha * log_prob - q_actions_).mean()

        for _,  ind in enumerate(index):
            actor = all_actors[ind]
            std_param = [actor.mean.weight, actor.mean.bias]
            mu_param = [actor.log_std.weight, actor.log_std.bias]
            std_param = nn.utils.parameters_to_vector(std_param).data.cpu().numpy()
            mu_param = nn.utils.parameters_to_vector(mu_param).data.cpu().numpy()
            std_param = torch.FloatTensor(std_param).to('cuda' if self.args.cuda else 'cpu')
            std_param = std_param.repeat(len(obses), 1)
            mu_param = torch.FloatTensor(mu_param).to('cuda' if self.args.cuda else 'cpu')
            mu_param = mu_param.repeat(len(obses), 1)
            EA_q1, EA_q2 = self.PVN.forward(torch.cat([obses, EA_actions_list[_]], -1), mu_param, std_param)
            EA_q_actions_ = torch.min(EA_q1, EA_q2)
            EA_actor_loss = (alpha * EA_log_prob_list[_] - EA_q_actions_).mean()
            actor_loss += EA_actor_loss

        next_s_z = self.state_embedding.forward(obses_).detach()

        Pvn_loss_total = 0
        for ind in index:
            actor = all_actors[ind]
            std_param = [actor.mean.weight, actor.mean.bias]
            mu_param = [actor.log_std.weight, actor.log_std.bias]
            std_param = nn.utils.parameters_to_vector(std_param).data.cpu().numpy()
            mu_param = nn.utils.parameters_to_vector(mu_param).data.cpu().numpy()
            std_param = torch.FloatTensor(std_param).to('cuda' if self.args.cuda else 'cpu')
            std_param = std_param.repeat(len(obses), 1)
            mu_param = torch.FloatTensor(mu_param).to('cuda' if self.args.cuda else 'cpu')
            mu_param = mu_param.repeat(len(obses), 1)
            pvn_q1, pvn_q2 = self.PVN.forward(torch.cat([obses, actions], -1), mu_param, std_param)

            with torch.no_grad():
                pis_next = actor(next_s_z)
                actions_info_next = get_action_info(pis_next, cuda=self.args.cuda)
                actions_next_, pre_tanh_value_next = actions_info_next.select_actions(reparameterize=True)
                log_prob_next = actions_info_next.get_log_prob(actions_next_, pre_tanh_value_next)
                target_pvn_q1, target_pvn_q2 = self.target_PVN.forward(torch.cat([obses_, actions_next_], -1), mu_param, std_param)
                target_q_value_next = torch.min(target_pvn_q1, target_pvn_q2) - alpha * log_prob_next
                target_q_value = self.args.reward_scale * rewards + inverse_dones * self.args.gamma * target_q_value_next
            pvn_loss = (pvn_q1 - target_q_value).pow(2).mean() +  (pvn_q2 - target_q_value).pow(2).mean()
            Pvn_loss_total += pvn_loss

        self.pvn_optim.zero_grad()
        Pvn_loss_total.backward()
        self.pvn_optim.step()

        # q value function loss
        q1_value = self.qf1(obses, actions)
        q2_value = self.qf2(obses, actions)
        with torch.no_grad():

            pis_next = self.actor_net(next_s_z)
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
        self.state_embedding_optimizer.zero_grad()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.state_embedding_optimizer.step()
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
                    s_z = self.state_embedding.forward(obs_tensor)
                    pi = actor_net(s_z)
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
