import os
import torch
import wandb
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
os.environ["WANDB_API_KEY"] = ""
#os.environ["WANDB_MODE"] = "offline"
torch.set_num_threads(cpu_num)
import numpy as np
import random
from arguments import get_args
from EvoRainbow_sac_agent import sac_agent
from utils import env_wrapper
import utils


if __name__ == '__main__':
    args = get_args()
    # build the environment

    name = "EvoRainbow_k_" + str(args.K) + "_"+str(args.EA_tau)+"_CEM_"+ str(args.damp) + "_"+ str(args.damp_limit)+"_SAC_Env_"+ str(args.H) + "_" + str(args.theta) +"_" + str(args.pop_size) + "_" + str(args.policy_representation_dim) + "_" + str(args.batch_size) + "_" + str(args.env_name) + "_steps_" + str(args.total_timesteps)
    our_wandb = wandb.init(project="MetaWorld-v2", name=name)
    # env = TASKS[args.env_name]()

    env = utils.make_metaworld_env(args, args.seed)
    env = env_wrapper(env, args)
    # create the eval env
    # eval_env = TASKS[args.env_name]()
    eval_env = utils.make_metaworld_env(args, args.seed + 100)
    eval_env = env_wrapper(eval_env, args)
    # set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    # set the seed of torch
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # create the agent
    sac_trainer = sac_agent(env, eval_env, args, our_wandb)
    sac_trainer.learn()
