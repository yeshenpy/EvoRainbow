import numpy as np
import gym, torch
import os
cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
import time 
import random
import psutil

import argparse
import pickle
from EvoRainbow_core.parameters import Parameters

parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choices: (Swimmer-v2) (HalfCheetah-v2) (Hopper-v2) ' +
                                 '(Walker2d-v2) (Ant-v2)', required=True, type=str)
parser.add_argument('-seed', help='Random seed to be used', type=int, default=7)
parser.add_argument('-pr', help='pr', type=int, default=128)
parser.add_argument('-pop_size', help='pop_size', type=int, default=10)

parser.add_argument('-disable_cuda', help='Disables CUDA', action='store_true')
parser.add_argument('-render', help='Render gym episodes', action='store_true')
parser.add_argument('-sync_period', help="How often to sync to population", type=int)
parser.add_argument('-novelty', help='Use novelty exploration', action='store_true')
parser.add_argument('-proximal_mut', help='Use safe mutation', action='store_true')
parser.add_argument('-distil', help='Use distilation crossover', action='store_true')
parser.add_argument('-distil_type', help='Use distilation crossover. Choices: (fitness) (distance)',
                    type=str, default='fitness')
parser.add_argument('-EA', help='Use ea', action='store_true')
parser.add_argument('-RL', help='Use rl', action='store_true')
parser.add_argument('-detach_z', help='detach_z', action='store_true')
parser.add_argument('-random_choose', help='Use random_choose', action='store_true')

parser.add_argument('-per', help='Use Prioritised Experience Replay', action='store_true')
parser.add_argument('-use_all', help='Use all', action='store_true')

parser.add_argument('-intention', help='intention', action='store_true')

parser.add_argument('-mut_mag', help='The magnitude of the mutation', type=float, default=0.05)
parser.add_argument('-tau', help='tau', type=float, default=0.005)

parser.add_argument('-prob_reset_and_sup', help='prob_reset_and_sup', type=float, default=0.05)
parser.add_argument('-frac', help='frac', type=float, default=0.1)


parser.add_argument('-TD3_noise', help='tau', type=float, default=0.2)
parser.add_argument('-mut_noise', help='Use a random mutation magnitude', action='store_true')
parser.add_argument('-verbose_mut', help='Make mutations verbose', action='store_true')
parser.add_argument('-verbose_crossover', help='Make crossovers verbose', action='store_true')
parser.add_argument('-logdir', help='Folder where to save results', type=str, required=True)
parser.add_argument('-opstat', help='Store statistics for the variation operators', action='store_true')
parser.add_argument('-opstat_freq', help='Frequency (in generations) to store operator statistics', type=int, default=1)
parser.add_argument('-save_periodic', help='Save actor, critic and memory periodically', action='store_true')
parser.add_argument('-next_save', help='Generation save frequency for save_periodic', type=int, default=200)
parser.add_argument('-K', help='K', type=int, default=5)
parser.add_argument('-OFF_TYPE', help='OFF_TYPE', type=int, default=1)
parser.add_argument('-num_evals', help='num_evals', type=int, default=1)

parser.add_argument('-version', help='version', type=int, default=1)
parser.add_argument('-time_steps', help='time_steps', type=int, default=1)
parser.add_argument('-test_operators', help='Runs the operator runner to test the operators', action='store_true')
parser.add_argument('-EA_actor_alpha', help='EA_actor_alpha', type=float, default=1.0)
parser.add_argument('-state_alpha', help='state_alpha', type=float, default=1.0)
parser.add_argument('-actor_alpha', help='actor_alpha', type=float, default=1.0)
parser.add_argument('-theta', help='theta', type=float, default=0.5)

parser.add_argument('-gamma', help='gamma', type=float, default=0.99)


parser.add_argument('-scale', help='scale', type=float, default=1.0)

parser.add_argument('-EA_tau', help='EA_tau', type=float, default=0.3)

parser.add_argument('-Soft_Update', default=1, type=int)
parser.add_argument('-Value_Function', default=1, type=int)
# CEM args
# parser.add_argument('-pop_size', help='pop_size', type=int, default=10)     ### 种群大小
parser.add_argument('-n_grad', default=3, type=int)                         ### 种群中的RL actor个数
parser.add_argument('-sigma_init', default=1e-3, type=float)                ### 初始sigma大小
parser.add_argument('-damp', default=1e-3, type=float)                      ### 噪声大小
parser.add_argument('-damp_limit', default=1e-5, type=float)                ### 噪声衰减因子
parser.add_argument('-elitism', action='store_true')                        ### 保护1st精英
parser.add_argument('-mult_noise', action='store_true')                     ### 好像没有用到??????????????????????????

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
parameters = Parameters(parser)  # Inject the cla arguments in the parameters object

# Create Env
#env = utils.NormalizedActions(gym.make(parameters.env_name))
env = gym.make(parameters.env_name)
print("env.action_space.low",env.action_space.low, "env.action_space.high",env.action_space.high)
parameters.action_dim = env.action_space.shape[0]
parameters.state_dim = env.observation_space.shape[0]

# Write the parameters to a the info file and print them
parameters.write_params(stdout=True)

# Seed
os.environ['PYTHONHASHSEED']= str(parameters.seed)
env.seed(parameters.seed)
torch.manual_seed(parameters.seed)
np.random.seed(parameters.seed)
random.seed(parameters.seed)

from EvoRainbow_core import mod_utils as utils, agent

env.action_space.seed(parameters.seed)

if __name__ == "__main__":

    # Create Agent
    agent = agent.Agent(parameters, env)
    print('Running', parameters.env_name, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)

    next_save = parameters.next_save; time_start = time.time()
    while agent.num_frames <= parameters.num_frames:
        stats = agent.train()
        if stats['rho'] is not None:
            parameters.wandb.log(
                {'spearmanr_c': stats['rho'], 'elite_spearmanr_c': stats['rho_elite'], 'num_frames': agent.num_frames,
                 'inter_spearmanr_c': stats['rho_interaction'],
                 'elite_inter_spearmanr_c': stats['elite_rho_interaction'],
                 'fake_inter_spearmanr_c': stats['rho_of_fake_and_interaction'],
                 'elite_fake_inter_spearmanr_c': stats['elite_rho_of_fake_and_interaction']})
        if stats['log_wandb']:
            elite_rate = stats['elite_rate']
            win_rate = stats['win_rate']
            dis_rate = stats['dis_rate']
            best_train_fitness = stats['best_train_fitness']
            erl_score = stats['test_score']
            mu_score = stats['mu_score']
            # elite_index = stats['elite_index']
            RL_reward = stats['RL_reward']
            policy_gradient_loss = stats['pg_loss']
            behaviour_cloning_loss = stats['bc_loss']
            population_novelty = stats['pop_novelty']
            current_q = stats['current_q']
            target_q = stats['target_q']
            pre_loss = stats['pre_loss']
            before_rewards = stats['before_rewards']
            add_rewards = stats['add_rewards']
            l1_before_after = stats['l1_before_after']
            keep_c_loss = stats['keep_c_loss']
            pvn_loss = stats['pvn_loss']
            min_fintess = stats['min_fintess']
            best_old_fitness = stats['best_old_fitness']
            new_fitness = stats['new_fitness']
            previous_info = stats['previous_info']
            current_info = stats['current_info']

            print('#Games:', agent.num_games, '#Frames:', agent.num_frames,
                  ' Train_Max:', '%.2f'%best_train_fitness if best_train_fitness is not None else None,
                  ' Mu_Score:', '%.2f'%mu_score,
                  ' Test_Score:','%.2f'%erl_score if erl_score is not None else None,
                  ' ENV:  '+ parameters.env_name,
                  ' RL Reward:', '%.2f'%RL_reward,
                  ' PG Loss:', '%.4f' % policy_gradient_loss)
    
            # elite = agent.evolver.selection_stats['elite']/agent.evolver.selection_stats['total']
            # selected = agent.evolver.selection_stats['selected'] / agent.evolver.selection_stats['total']
            # discarded = agent.evolver.selection_stats['discarded'] / agent.evolver.selection_stats['total']
    
            print()

            min_fintess = stats['min_fintess']
            best_old_fitness = stats['best_old_fitness']
            new_fitness = stats['new_fitness']
            best_reward = np.max([RL_reward,erl_score])
            #print(previous_info, current_info)
            if previous_info is not None :

                parameters.wandb.log(
                    {'Pop_mean': current_info[0], 'Pop_max':current_info[1], 'Pop_min': current_info[2],'Pop_improve': current_info[0] - previous_info[0], 'Pop_max_update':current_info[1]- previous_info[1],'elite': elite_rate, 'selected': win_rate, 'discarded': dis_rate,'best_reward': best_reward, 'add_rewards': add_rewards,
                     'pvn_loss': pvn_loss, 'keep_c_loss': keep_c_loss, 'l1_before_after': l1_before_after,
                     'mu_score': mu_score,
                     'pre_loss': pre_loss, 'num_frames': agent.num_frames, 'num_games': agent.num_games,
                     'erl_score': erl_score, 'RL_reward': RL_reward,
                     'policy_gradient_loss': policy_gradient_loss, 'population_novelty': population_novelty,
                     'best_train_fitness': best_train_fitness, 'behaviour_cloning_loss': behaviour_cloning_loss})
            else :
                parameters.wandb.log(
                    {'Pop_mean': current_info[0], 'Pop_max': current_info[1], 'Pop_min': current_info[2],
                     'elite': elite_rate, 'selected': win_rate,
                     'discarded': dis_rate, 'best_reward': best_reward, 'add_rewards': add_rewards,
                     'pvn_loss': pvn_loss, 'keep_c_loss': keep_c_loss, 'l1_before_after': l1_before_after,
                     'mu_score': mu_score,
                     'pre_loss': pre_loss, 'num_frames': agent.num_frames, 'num_games': agent.num_games,
                     'erl_score': erl_score, 'RL_reward': RL_reward,
                     'policy_gradient_loss': policy_gradient_loss, 'population_novelty': population_novelty,
                     'best_train_fitness': best_train_fitness, 'behaviour_cloning_loss': behaviour_cloning_loss})

                # Save Policy
        if agent.num_games > next_save:
            next_save += parameters.next_save
            torch.save(agent.muagent.actor.state_dict(), os.path.join(parameters.save_foldername,
                                                                               str(parameters.seed) + '_mu_actor_'+str(agent.num_games)+'.pkl'))
            torch.save(agent.rl_agent.state_embedding.state_dict(), os.path.join(parameters.save_foldername,
                                                                               str(parameters.seed) + '_evo_share_'+str(agent.num_games)+'.pkl'))
            torch.save(agent.rl_agent.actor.state_dict(), os.path.join(parameters.save_foldername,
                                                                               str(parameters.seed) + '_evo_rl_actor_'+str(agent.num_games)+'.pkl'))
            if parameters.save_periodic:
                save_folder = os.path.join(parameters.save_foldername, 'models')
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                actor_save_name = os.path.join(save_folder, 'evo_net_actor_{}.pkl'.format(next_save))
                critic_save_name = os.path.join(save_folder, 'evo_net_critic_{}.pkl'.format(next_save))
                buffer_save_name = os.path.join(save_folder, 'champion_buffer_{}.pkl'.format(next_save))

                #torch.save(agent.pop[elite_index].actor.state_dict(), actor_save_name)
                #torch.save(agent.rl_agent.critic.state_dict(), critic_save_name)
                with open(buffer_save_name, 'wb+') as buffer_file:
                    pickle.dump(agent.rl_agent.buffer, buffer_file)

            print("Progress Saved")











