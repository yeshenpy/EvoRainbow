import argparse

# define the arguments that will be used in the SAC
def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--env-name', type=str, default='sawyer_faucet_open', help='the environment name')
    parse.add_argument('--total-timesteps', type=int, default=int(1e7), help='the total num of timesteps to be run')
    parse.add_argument('--cuda', action='store_true', help='use GPU do the training')
    parse.add_argument('--seed', type=int, default=123, help='the random seed to reproduce results')
    parse.add_argument('--hidden-size', type=int, default=256, help='the size of the hidden layer')
    parse.add_argument('--train-loop-per-epoch', type=int, default=1, help='the training loop per epoch')
    parse.add_argument('--q-lr', type=float, default=3e-4, help='the learning rate')
    parse.add_argument('--p-lr', type=float, default=3e-4, help='the learning rate of the actor')
    parse.add_argument('--n-epochs', type=int, default=int(3e3), help='the number of total epochs')
    parse.add_argument('--epoch-length', type=int, default=int(1e3), help='the lenght of each epoch')
    parse.add_argument('--n-updates', type=int, default=int(1e3), help='the number of training updates execute')
    parse.add_argument('--init-exploration-steps', type=int, default=int(1e4), help='the steps of the initial exploration')
    parse.add_argument('--init-exploration-policy', type=str, default='uniform', help='the inital exploration policy')
    parse.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the replay buffer')
    parse.add_argument('--batch-size', type=int, default=128, help='the batch size of samples for training')
    parse.add_argument('--reward-scale', type=float, default=1, help='the reward scale')
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parse.add_argument('--log-std-max', type=float, default=2, help='the maximum log std value')
    parse.add_argument('--log-std-min', type=float, default=-20, help='the minimum log std value')
    parse.add_argument('--entropy-weights', type=float, default=0.2, help='the entropy weights')
    parse.add_argument('--tau', type=float, default=5e-3, help='the soft update coefficient')
    parse.add_argument('--target-update-interval', type=int, default=1, help='the interval to update target network')
    parse.add_argument('--update-cycles', type=int, default=1, help='how many updates apply in the update')
    parse.add_argument('--eval-episodes', type=int, default=10, help='the episodes that used for evaluation')
    parse.add_argument('--display-interval', type=int, default=int(1e4), help='the display interval')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the place to save models')
    parse.add_argument('--reg', type=float, default=1e-3, help='the reg term')
    parse.add_argument('--reward-shaping', action='store_true', help='if use the reward shaping to train the agent')
    parse.add_argument('--action-repeat', type=int, default=10, help='repeat the action for n times')
    parse.add_argument('--episode-length', type=int, default=150, help='the length of one episode')
    parse.add_argument('--random-init', action='store_true', help='if use random init in the episode')
    parse.add_argument('--render', action='store_true', help='if render the env')
    parse.add_argument('--obs-type', type=str, default='plain', help='which obs type')


    parse.add_argument('--pop_size', type=int, default=5, help='the size of population')
    parse.add_argument('--policy_representation_dim', type=int, default=32, help='the size of policy representation')
    parse.add_argument('--K', type=int, default=1, help='Sample actor')
    parse.add_argument('--H', type=int, default=30, help='rollout step')
    parse.add_argument('--theta', type=float, default=0.8, help='the prob of using H step bootstrap')
    parse.add_argument('--EA_tau', type=float, default=0.1, help='EA tau')


    parse.add_argument('-sigma_init', default=1e-3, type=float)  ### 初始sigma大小
    parse.add_argument('-damp', default=1e-3, type=float)  ### 噪声大小
    parse.add_argument('-damp_limit', default=1e-5, type=float)  ### 噪声衰减因子
    parse.add_argument('-elitism', action='store_true')  ### 保护1st精英
    return parse.parse_args()
