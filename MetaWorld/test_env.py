import time
import numpy as np
import argparse
from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place import SawyerReachPushPickPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerBasketballEnv
from utils import env_wrapper
from arguments import get_args

# function that closes the render window
def sample_sawyer_pick_and_place(args):
    env = SawyerBasketballEnv(obs_type='with_goal', random_init=True)
    env = env_wrapper(env, args)
    for ep in range(6):
        obs = env.reset()
        print('ep: {}, goal: {}'.format(ep, obs[6:]))
        for t in range(20):
            #env.render()
            obs, reward, done, info = env.step(env.action_space.sample())
            print("episode: {}, timestep: {}, done: {}, goal: {}, goal_info: {}".format(ep, t, done, obs[6:], info['goal']))
            if done: break
 #   glfw.destroy_window(env.viewer.window)
if __name__ == '__main__':
    args = get_args()
    sample_sawyer_pick_and_place(args)
