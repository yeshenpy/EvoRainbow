import pprint
import torch
import os
import wandb
os.environ["WANDB_API_KEY"] = ""
#os.environ["WANDB_MODE"] = "offline"
class Parameters:
    def __init__(self, cla, init=True):
        if not init:
            return
        cla = cla.parse_args()

        # Set the device to run on CUDA or CPU
        if not cla.disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Render episodes
        self.render = cla.render
        self.env_name = cla.env
        self.save_periodic = cla.save_periodic

        # Number of Frames to Run
        self.num_frames = 1005000

        # Synchronization
        if cla.env == 'HalfCheetah-v2' or cla.env == 'Hopper-v2' or cla.env == 'Ant-v2' or cla.env == 'Walker2d-v2' or cla.env == "Humanoid-v2":
            self.rl_to_ea_synch_period = 1
        else:
            self.rl_to_ea_synch_period = 10
        
        # Overwrite sync from command line if value is passed
        if cla.sync_period is not None:
            self.rl_to_ea_synch_period = cla.sync_period

        # Novelty Search
        self.ns = cla.novelty
        self.ns_epochs = 10

        # Model save frequency if save is active
        self.next_save = cla.next_save

        # DDPG params
        self.use_ln = True
        self.gamma = cla.gamma
        self.tau = cla.tau
        self.seed = cla.seed
        self.batch_size = 128
        self.frac_frames_train = 1.0
        self.use_done_mask = True
        self.buffer_size = 1000000
        self.ls = 300

        # Prioritised Experience Replay
        self.per = cla.per
        self.replace_old = True
        self.alpha = 0.7
        self.beta_zero = 0.5
        self.learn_start = (1 + self.buffer_size / self.batch_size) * 2
        self.total_steps = self.num_frames

        # ========================================== NeuroEvolution Params =============================================

        # Num of trials
        self.num_evals = 1
        if cla.num_evals is not None:
            self.num_evals = cla.num_evals

        # Elitism Rate
        self.elite_fraction = 0.2
        # Number of actors in the population
        self.pop_size = cla.pop_size

        self.n_grad = cla.n_grad
        self.sigma_init = cla.sigma_init
        self.damp = cla.damp
        self.damp_limit = cla.damp_limit
        self.elitism = cla.elitism
        self.mult_noise = cla.mult_noise

        # Mutation and crossover
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9
        self.mutation_mag = cla.mut_mag
        self.mutation_noise = cla.mut_noise
        self.mutation_batch_size = 256
        self.proximal_mut = cla.proximal_mut
        self.distil = cla.distil
        self.distil_type = cla.distil_type
        self.verbose_mut = cla.verbose_mut
        self.verbose_crossover = cla.verbose_crossover

        # Genetic memory size
        self.individual_bs = 8000
        self.intention = cla.intention
        # Variation operator statistics
        self.opstat = cla.opstat
        self.opstat_freq = cla.opstat_freq
        self.test_operators = cla.test_operators

        # Save Results
        self.state_dim = None  # To be initialised externally
        self.action_dim = None  # To be initialised externally
        self.random_choose = cla.random_choose
        self.EA = cla.EA
        self.RL = cla.RL 
        self.K = cla.K 
        self.state_alpha = cla.state_alpha
        self.detach_z = cla.detach_z
        self.actor_alpha = cla.actor_alpha
        self.TD3_noise = cla.TD3_noise
        self.pr = cla.pr
        self.use_all = cla.use_all 
        self.OFF_TYPE = cla.OFF_TYPE
        self.prob_reset_and_sup = cla.prob_reset_and_sup
        self.frac = cla.frac
        self.EA_actor_alpha = cla.EA_actor_alpha
        self.theta = cla.theta
        self.time_steps = cla.time_steps
        self.init_steps = 10000
        self.scale = 1.0
        self.Soft_Update = cla.Soft_Update
        self.Value_Function = cla.Value_Function
        self.EA_tau = cla.EA_tau
        self.name = "EvoRainbow_H_step_Value_Function_"+str(self.Value_Function)+"_Theta_"+ str(self.theta)+ "_"+ str(self.gamma)+ "_EA_guide_RL_"+ str(self.Soft_Update) + "_"+ str(self.EA_tau) +"_Add_CEM_use_Re2_"+ str(self.sigma_init)+ "_from_"+str(self.damp) +"_to_"+str(self.damp_limit)+"_H_alpha_" +str(self.time_steps)  +"_K_" + str(self.K) + "_"  + str(self.env_name)

        self.wandb = wandb.init(project="Revisiting",name=self.name)

        self.wandb.config.rl_to_ea_synch_period = self.rl_to_ea_synch_period
        self.wandb.config.env = cla.env
        self.wandb.config.tau = self.tau

        self.wandb.config.gamma = self.gamma
        self.wandb.config.num_evals = self.num_evals
        self.wandb.config.elite_fraction = self.elite_fraction
        self.wandb.config.crossover_prob = self.crossover_prob
        self.wandb.config.mutation_prob = self.mutation_prob
        self.wandb.config.mutation_batch_size = self.mutation_batch_size
        self.wandb.config.distil = self.distil
        self.wandb.config.proximal_mut = self.proximal_mut

        self.save_foldername = cla.logdir + "/"+self.name
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

    def write_params(self, stdout=True):
        # Dump all the hyper-parameters in a file.
        params = pprint.pformat(vars(self), indent=4)
        if stdout:
            print(params)

        with open(os.path.join(self.save_foldername, 'info.txt'), 'a') as f:
            f.write(params)