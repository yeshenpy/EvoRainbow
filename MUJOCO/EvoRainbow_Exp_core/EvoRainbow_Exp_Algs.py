import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
from EvoRainbow_Exp_core.parameters import Parameters
from EvoRainbow_Exp_core import replay_memory
from EvoRainbow_Exp_core.mod_utils import is_lnorm_key
import numpy as np

from sklearn.utils import shuffle

def get_negative_expectation(q_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        #
        Eq = F.softplus(-q_samples) + q_samples - log_2  # Note JSD will be shifted
        # Eq = F.softplus(q_samples) #+ q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        q_samples = torch.clamp(q_samples, -1e6, 9.5)

        # print("neg q samples ",q_samples.cpu().data.numpy())
        Eq = torch.exp(q_samples - 1.)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        assert 1 == 2

    if average:
        return Eq.mean()
    else:
        return Eq


def get_positive_expectation(p_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)  # Note JSD will be shifted
        # Ep =  - F.softplus(-p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples

    elif measure == 'RKL':

        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        assert 1 == 2

    if average:
        return Ep.mean()
    else:
        return Ep


def fenchel_dual_loss(l, m, measure=None):
    '''Computes the f-divergence distance between positive and negative joint distributions.
    Note that vectors should be sent as 1x1.
    Divergences supported are Jensen-Shannon `JSD`, `GAN` (equivalent to JSD),
    Squared Hellinger `H2`, Chi-squeared `X2`, `KL`, and reverse KL `RKL`.
    Args:
        l: Local feature map.
        m: Multiple globals feature map.
        measure: f-divergence measure.
    Returns:
        torch.Tensor: Loss.
    '''
    N, units = l.size()

    # Outer product, we want a N x N x n_local x n_multi tensor.
    u = torch.mm(m, l.t())

    # Since we have a big tensor with both positive and negative samples, we need to mask.
    mask = torch.eye(N).to(l.device)
    n_mask = 1 - mask
    # Compute the positive and negative score. Average the spatial locations.
    E_pos = get_positive_expectation(u, measure, average=False)
    E_neg = get_negative_expectation(u, measure, average=False)
    MI = (E_pos * mask).sum(1)  # - (E_neg * n_mask).sum(1)/(N-1)
    # Mask positive and negative terms for positive and negative parts of loss
    E_pos_term = (E_pos * mask).sum(1)
    E_neg_term = (E_neg * n_mask).sum(1) / (N - 1)
    loss = E_neg_term - E_pos_term
    return loss, MI


class MINE(nn.Module):
    def __init__(self, action_dim, z_dim, measure="JSD"):
        super(MINE, self).__init__()
        self.measure = measure
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.nonlinearity = F.leaky_relu
        self.l1_a = nn.Linear(self.action_dim, 64)
        self.l1_z = nn.Linear(z_dim, 64)
        self.l1 =  nn.Linear(128, 128)

        self.l2 = nn.Linear(self.z_dim, 128)

    def forward(self, action, s_z,next_s_z, params=None):

        em_1_a = self.nonlinearity(self.l1_a(action), inplace=True)
        em_1_z = self.nonlinearity(self.l1_z(s_z), inplace=True)

        em_1 = self.nonlinearity(self.l1(torch.cat([em_1_a,em_1_z],-1)), inplace=True)

        em_2 = self.nonlinearity(self.l2(next_s_z), inplace=True)
        two_agent_embedding = [em_1, em_2]
        loss, MI = fenchel_dual_loss(two_agent_embedding[0], two_agent_embedding[1], measure=self.measure)
        return loss, MI


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class GeneticAgent:
    def __init__(self, args: Parameters):

        self.args = args
        self.actor = Actor(args)
        self.old_actor = Actor(args)
        self.temp_actor = Actor(args)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.buffer = replay_memory.ReplayMemory(self.args.individual_bs, args.device)
        self.loss = nn.MSELoss()

    def keep_consistency(self, z_old, z_new):
        target_action = self.old_actor.select_action_from_z(z_old).detach()
        current_action = self.actor.select_action_from_z(z_new)
        delta = (current_action - target_action).abs()
        dt = torch.mean(delta ** 2)
        self.actor_optim.zero_grad()
        dt.backward()
        self.actor_optim.step()
        return dt.data.cpu().numpy()

    def keep_consistency_with_other_agent(self, z_old, z_new, other_actor):
        target_action = other_actor.select_action_from_z(z_old).detach()
        current_action = self.actor.select_action_from_z(z_new)
        delta = (current_action - target_action).abs()
        dt = torch.mean(delta ** 2)
        self.actor_optim.zero_grad()
        dt.backward()
        self.actor_optim.step()
        return dt.data.cpu().numpy()

    def update_parameters(self, batch, p1, p2, critic):
        state_batch, _, _, _, _ = batch

        p1_action = p1(state_batch)
        p2_action = p2(state_batch)
        p1_q = critic.Q1(state_batch, p1_action).flatten()
        p2_q = critic.Q1(state_batch, p2_action).flatten()

        eps = 0.0
        action_batch = torch.cat((p1_action[p1_q - p2_q > eps], p2_action[p2_q - p1_q >= eps])).detach()
        state_batch = torch.cat((state_batch[p1_q - p2_q > eps], state_batch[p2_q - p1_q >= eps]))
        actor_action = self.actor(state_batch)

        # Actor Update
        self.actor_optim.zero_grad()
        sq = (actor_action - action_batch)**2
        policy_loss = torch.sum(sq) + torch.mean(actor_action**2)
        policy_mse = torch.mean(sq)
        policy_loss.backward()
        self.actor_optim.step()

        return policy_mse.item()

class shared_state_embedding(nn.Module):
    def __init__(self, args):
        super(shared_state_embedding, self).__init__()
        self.args = args
        l1 = 400
        l2 = args.ls
        l3 = l2

        # Construct Hidden Layer 1
        self.w_l1 = nn.Linear(args.state_dim, l1)
        if self.args.use_ln: self.lnorm1 = LayerNorm(l1)

        # Hidden Layer 2
        self.w_l2 = nn.Linear(l1, l2)
        if self.args.use_ln: self.lnorm2 = LayerNorm(l2)
        # Init
        self.to(self.args.device)

    def forward(self, state):
        # Hidden Layer 1
        out = self.w_l1(state)
        if self.args.use_ln: out = self.lnorm1(out)
        out = out.tanh()

        # Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = out.tanh()

        return out

from copy import deepcopy
import math
def to_numpy(var):
    return var.data.numpy()

import math
class Actor(nn.Module):
    def __init__(self, args, init=False):
        super(Actor, self).__init__()
        self.args = args

        self.state_embedding = shared_state_embedding(args)

        l1 = args.ls; l2 = args.ls; l3 = l2
        # Out
        #array = torch.empty(l3, args.action_dim)
        # torch.rand()
        #nn.init.kaiming_uniform_(array, a=math.sqrt(5))
        #self.policy_embedding = torch.nn.Parameter(torch.tensor(array, dtype=torch.float32, requires_grad=True))
        #self.policy_embedding.data.mul_(0.1)
        self.w_out = nn.Linear(l3, args.action_dim)
        # Init
        if init:
            self.w_out.weight.data.mul_(0.1)
            self.w_out.bias.data.mul_(0.1)

        self.to(self.args.device)

    def forward(self, input):
        s_z = self.state_embedding.forward(input)
        action = self.w_out(s_z).tanh()
        return action
        #action = torch.matmul(s_z, self.policy_embedding).tanh()
        #return action
    def select_action_from_z(self,s_z):

        action = self.w_out(s_z).tanh()
        #action = torch.matmul(s_z, self.policy_embedding).tanh()
        return action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.args.device)
        return self.forward(state).cpu().data.numpy().flatten()

    def get_novelty(self, batch):
        state_batch, action_batch, _, _, _ = batch
        novelty = torch.mean(torch.sum((action_batch - self.forward(state_batch))**2, dim=-1))
        return novelty.item()

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            # if torch.cuda.is_available():
            #     param.data.copy_(torch.from_numpy(
            #         params[cpt:cpt + tmp]).view(param.size()).cuda())
            # else:
            param.data.copy_(torch.from_numpy(
                params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def get_grads(self):
        """
        Returns the current gradient
        """
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in self.parameters()]))

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]

    # function to return current pytorch gradient in same order as genome's flattened parameter vector
    def extract_grad(self):
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count + sz] = param.grad.view(-1)
            count += sz
        return pvec.detach().clone()

    # function to grab current flattened neural network weights
    def extract_parameters(self):
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count + sz] = param.view(-1)
            count += sz
        return pvec.detach().clone()

    # function to inject a flat vector of ANN parameters into the model's current neural network weights
    def inject_parameters(self, pvec):
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            raw = pvec[count:count + sz]
            reshaped = raw.view(param.size())
            param.data.copy_(reshaped.data)
            count += sz

    # count how many parameters are in the model
    def count_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            count += param.numel()
        return count


class Critic(nn.Module):

    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args

        l1 = 400;
        l2 = 300;
        l3 = l2

        # Construct input interface (Hidden Layer 1)
        self.w_l1 = nn.Linear(args.state_dim+args.action_dim, l1)
        # Hidden Layer 2

        self.w_l2 = nn.Linear(l1, l2)
        if self.args.use_ln:
            self.lnorm1 = LayerNorm(l1)
            self.lnorm2 = LayerNorm(l2)

        # Out
        self.w_out = nn.Linear(l3, 1)
        self.w_out.weight.data.mul_(0.1)
        self.w_out.bias.data.mul_(0.1)

        self.w_l3 = nn.Linear(args.state_dim+args.action_dim, l1)
        # Hidden Layer 2
        self.w_l4 = nn.Linear(l1, l2)
        if self.args.use_ln:
            self.lnorm3 = LayerNorm(l1)
            self.lnorm4 = LayerNorm(l2)

        # Out
        self.w_out_2 = nn.Linear(l3, 1)
        self.w_out_2.weight.data.mul_(0.1)
        self.w_out_2.bias.data.mul_(0.1)

        self.to(self.args.device)

    def forward(self, input, action):

        # Hidden Layer 1 (Input Interface)
        concat_input = torch.cat([input,action],-1)

        out = self.w_l1(concat_input)
        if self.args.use_ln:out = self.lnorm1(out)

        out = F.leaky_relu(out)
        # Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = F.leaky_relu(out)
        # Output interface
        out_1 = self.w_out(out)

        out_2 = self.w_l3(concat_input)
        if self.args.use_ln: out_2 = self.lnorm3(out_2)
        out_2 = F.leaky_relu(out_2)

        # Hidden Layer 2
        out_2 = self.w_l4(out_2)
        if self.args.use_ln: out_2 = self.lnorm4(out_2)
        out_2 = F.leaky_relu(out_2)

        # Output interface
        out_2 = self.w_out_2(out_2)

        return out_1, out_2

    def Q1(self, input, action):

        concat_input = torch.cat([input, action], -1)

        out = self.w_l1(concat_input)
        if self.args.use_ln:out = self.lnorm1(out)

        out = F.leaky_relu(out)
        # Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = F.leaky_relu(out)
        # Output interface
        out_1 = self.w_out(out)
        return out_1




class Policy_Value_Network(nn.Module):

    def __init__(self, args):
        super(Policy_Value_Network, self).__init__()
        self.args = args

        self.policy_size = self.args.ls * self.args.action_dim + self.args.action_dim

        l1 = 400; l2 = 300; l3 = l2
        self.l1 = l1
        # Construct input interface (Hidden Layer 1)

        if self.args.use_ln:
            self.lnorm1 = LayerNorm(l1)
            self.lnorm2 = LayerNorm(l2)
            self.lnorm3 = LayerNorm(l1)
            self.lnorm4 = LayerNorm(l2)
        self.policy_w_l1 = nn.Linear(self.args.ls + 1, self.args.pr)
        self.policy_w_l2 = nn.Linear(self.args.pr, self.args.pr)
        self.policy_w_l3 = nn.Linear(self.args.pr, self.args.pr)
        # self.w_state_l1 = nn.Linear(args.state_dim, l1)
        # self.w_action_l1 = nn.Linear(args.action_dim, l1)
        if self.args.OFF_TYPE == 1 :
            input_dim = self.args.state_dim + self.args.action_dim
        else:
            input_dim = self.args.ls

        self.w_l1 = nn.Linear(input_dim + self.args.pr, l1)
        # Hidden Layer 2

        self.w_l2 = nn.Linear(l1, l2)


        # Out
        self.w_out = nn.Linear(l3, 1)
        self.w_out.weight.data.mul_(0.1)
        self.w_out.bias.data.mul_(0.1)

        self.policy_w_l4 = nn.Linear(self.args.ls + 1, self.args.pr)
        self.policy_w_l5 = nn.Linear(self.args.pr, self.args.pr)
        self.policy_w_l6 = nn.Linear(self.args.pr, self.args.pr)
        # self.w_state_l1 = nn.Linear(args.state_dim, l1)
        # self.w_action_l1 = nn.Linear(args.action_dim, l1)
        self.w_l3 = nn.Linear(input_dim + self.args.pr, l1)
        # Hidden Layer 2

        self.w_l4 = nn.Linear(l1, l2)

        # Out
        self.w_out_2 = nn.Linear(l3, 1)
        self.w_out_2.weight.data.mul_(0.1)
        self.w_out_2.bias.data.mul_(0.1)

        self.to(self.args.device)

    def forward(self,  input,param):
        reshape_param = param.reshape([-1,self.args.ls + 1])

        out_p = F.leaky_relu(self.policy_w_l1(reshape_param))
        out_p = F.leaky_relu(self.policy_w_l2(out_p))
        out_p = self.policy_w_l3(out_p)
        out_p = out_p.reshape([-1,self.args.action_dim,self.args.pr])
        out_p = torch.mean(out_p,dim=1)

        # Hidden Layer 1 (Input Interface)
        concat_input = torch.cat((input,out_p), 1)

        # Hidden Layer 2
        out = self.w_l1(concat_input)
        if self.args.use_ln: out = self.lnorm1(out)
        out = F.leaky_relu(out)
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = F.leaky_relu(out)

        # Output interface
        out_1 = self.w_out(out)

        out_p = F.leaky_relu(self.policy_w_l4(reshape_param))
        out_p = F.leaky_relu(self.policy_w_l5(out_p))
        out_p = self.policy_w_l6(out_p)
        out_p = out_p.reshape([-1, self.args.action_dim, self.args.pr])
        out_p = torch.mean(out_p, dim=1)

        # Hidden Layer 1 (Input Interface)
        concat_input = torch.cat((input, out_p), 1)

        # Hidden Layer 2
        out = self.w_l3(concat_input)
        if self.args.use_ln: out = self.lnorm3(out)
        out = F.leaky_relu(out)

        out = self.w_l4(out)
        if self.args.use_ln: out = self.lnorm4(out)
        out = F.leaky_relu(out)

        # Output interface
        out_2 = self.w_out_2(out)

        
        return out_1, out_2

    def Q1(self, input, param):
        reshape_param = param.reshape([-1, self.args.ls + 1])

        out_p = F.leaky_relu(self.policy_w_l1(reshape_param))
        out_p = F.leaky_relu(self.policy_w_l2(out_p))
        out_p = self.policy_w_l3(out_p)
        out_p = out_p.reshape([-1, self.args.action_dim, self.args.pr])
        out_p = torch.mean(out_p, dim=1)

        # Hidden Layer 1 (Input Interface)

        # out_state = F.elu(self.w_state_l1(input))
        # out_action = F.elu(self.w_action_l1(action))
        concat_input = torch.cat((input, out_p), 1)

        # Hidden Layer 2
        out = self.w_l1(concat_input)
        if self.args.use_ln: out = self.lnorm1(out)
        out = F.leaky_relu(out)
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = F.leaky_relu(out)

        # Output interface
        out_1 = self.w_out(out)
        return out_1

import random

def caculate_prob(score):

    X = (score - np.min(score))/(np.max(score)-np.min(score) + 1e-8)
    max_X = np.max(X)

    exp_x = np.exp(X-max_X)
    sum_exp_x = np.sum(exp_x)
    prob = exp_x/sum_exp_x
    return prob

class TD3(object):
    def __init__(self, args):
        self.args = args
        self.max_action = 1.0
        self.device = args.device
        self.actor = Actor(args, init=True)
        self.actor_target = Actor(args, init=True)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.mine = MINE(self.args.action_dim,args.ls)


        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(args).to(self.device)
        self.critic_target = Critic(args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=1e-3)

        self.buffer = replay_memory.ReplayMemory(args.individual_bs, args.device)

        #self.max_action = max_action


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()



    def train(self,evo_times,all_fitness, all_gen , on_policy_states, on_policy_params, on_policy_discount_rewards,on_policy_actions,replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2, train_OFN_use_multi_actor= False,all_actor = None):
        actor_loss_list =[]
        critic_loss_list =[]
        pre_loss_list = []
        pv_loss_list = [0.0]
        keep_c_loss = [0.0]
        
        
        
        select_prob = caculate_prob(all_fitness)
        
        for it in range(iterations):

            x, y, u, r, d, _ ,_= replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
          #  next_action = torch.FloatTensor(next_u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
          #  policy_embeddings = torch.FloatTensor(policy_embeddings).to(self.device)
            
            #print("x 0", x[0])
            #print("x 1", x[1])


            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
#            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            next_action = (self.actor_target.forward(next_state)+noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()
            
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
 
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
            self.critic_optimizer.step()
            critic_loss_list.append(critic_loss.cpu().data.numpy().flatten())

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor.forward(state)).mean()
                # Optimize the actor

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # for param, target_param in zip(self.state_embedding.parameters(), self.state_embedding_target.parameters()):
                #     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                actor_loss_list.append(actor_loss.cpu().data.numpy().flatten())
                pre_loss_list.append(0.0)

            # if len(replay_buffer.storage) < 1024:
            #     batch = replay_buffer.sample(len(replay_buffer.storage))
            # else:
            #     batch = replay_buffer.sample(1024)
            # state_batch, _, _, _, _, _, _ = batch
            # state_batch = torch.FloatTensor(state_batch).to(self.args.device)
            # z_new = self.state_embedding.forward(state_batch).detach()
            # z_old = self.old_state_embedding.forward(state_batch).detach()
            # for gen in all_gen:
            #     loss = gen.keep_consistency(z_old, z_new)
            #     keep_c_loss.append(loss)

        return np.mean(actor_loss_list) , np.mean(critic_loss_list), np.mean(pre_loss_list),np.mean(pv_loss_list), np.mean(keep_c_loss)



def fanin_init(size, fanin=None):
    v = 0.008
    return torch.Tensor(size).uniform_(-v, v)

def actfn_none(inp): return inp

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class OUNoise:

    def __init__(self, action_dimension, scale=0.3, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
