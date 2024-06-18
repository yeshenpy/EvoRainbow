import numpy as np
import torch
from utils import env_wrapper
from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place import SawyerReachPushPickPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerFaucetOpenEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerSweepEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerBinPickingEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerWindowOpenEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerBasketballEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerSweepIntoGoalEnv
from arguments import get_args
from models import tanh_gaussian_actor
import time


TASKS = {'sawyer_faucet_open': SawyerFaucetOpenEnv,
        'pick_place': SawyerReachPushPickPlaceEnv, 
        'sweep': SawyerSweepEnv,
        'pick_bin': SawyerBinPickingEnv,
        'open_window': SawyerWindowOpenEnv,
        'basketball': SawyerBasketballEnv,
        'sweep_into_hole': SawyerSweepIntoGoalEnv,
        }

if __name__ == '__main__':
    args = get_args()
    # the environment
    #env = TASKS[args.env_name]()
    env = TASKS[args.env_name](random_init=args.random_init, obs_type=args.obs_type)
    env = env_wrapper(env, args)
    # start to build the network
    net = tanh_gaussian_actor(env.observation_space.shape[0], env.action_space.shape[0], args.hidden_size, args.log_std_min, args.log_std_max)
    # start to load the models
    model_path = args.save_dir + args.env_name + '/model.pt' if args.random_init else args.save_dir + args.env_name + '/fixed_best_model.pt'
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    # start to test
    for ep in range(5):
        obs = env.reset()
        reward_sum = 0
        while True:
            if args.render:
                env.render()
                #time.sleep(0.05)
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                mean, _ = net(obs_tensor)
                action = torch.tanh(mean).numpy().squeeze() 
            # use the action
            obs, reward, done, info = env.step(action * env.action_space.high)
            reward_sum += reward
            if done: break
        print('episode: {}, reward: {}, success: {}'.format(ep, reward_sum, info['success']))
