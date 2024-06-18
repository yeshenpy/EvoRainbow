# Meta World Single Tasks

## Requirement
- torch == 1.2.0
- metaworld

## Install Metaworld (use old version!)
```
https://github.com/rlworkgroup/metaworld.git
cd metaworld/
git checkout 2361d353d0895d5908156aec71341d4ad09dd3c2
pip install -e .
```

## Train the agent
```
python train_network.py --cuda --env-name='pick_place' --total-timesteps=1500000
```

## Demo
Download pretrained models from [here](https://drive.google.com/file/d/19zdmws5rFrH_2KjAl4GnwrtpeBxgwPIG/view?usp=sharing).
```
python demo.py --env-name='pick_place' --render
```
## Results
**Note**: the result is ploted by using 5 random seeds and using smooth function to make plot look better (the success rate of `pick_place` is nearly binary: 0 or 1). 
![results](figures/results.png)


