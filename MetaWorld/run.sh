
nohup   python train_EvoRainbow.py --env-name='push-back-v2' --EA_tau=1.0  -damp=1e-4 --K=1 --H=100 --theta=0.2 --total-timesteps=1000000  --seed 1  --eval-episodes 20 > ./logs/direct_use_critic1.log 2>&1 &

nohup   python train_EvoRainbow_Exp.py --env-name='push-back-v2' --EA_tau=0.3 -damp=1e-5  --total-timesteps=1000000  --seed 1  --eval-episodes 20 > ./logs/direct_use_critic1.log 2>&1 &
