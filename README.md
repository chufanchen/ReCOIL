# Dual RL: ReCOIL - Imitation Learning from expert data + suboptimal offline data.

### [**[Project Page](https://hari-sikchi.github.io/dual-rl/)**] 

### [**[Paper](https://arxiv.org/abs/2302.08560)**] 



Official code base for **[Dual RL: Unification and New Methods for Reinforcement and Imitation Learning](https://arxiv.org/abs/2302.08560)** by [Harshit Sikchi](https://hari-sikchi.github.io/), [Qinqing Zheng](https://enosair.github.io/), [Amy Zhang](https://www.ece.utexas.edu/people/faculty/amy-zhang), and [Scott Niekum](https://people.cs.umass.edu/~sniekum/).



This repository contains code for **ReCOIL** framework for Imitation Learning proposed in our paper.


Please refer to instructions inside the **offline** folder to get started with installation and running the code.


## How to run the code

### Install dependencies

Create an empty conda environment and follow the commands below.

```bash
conda create -n dvl python=3.9

conda install -c conda-forge cudnn

pip install --upgrade pip

# Install 1 of the below jax versions depending on your CUDA version
## 1. CUDA 12 installation
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

## 2. CUDA 11 installation
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


pip install -r requirements.txt

```

Also, see other configurations for CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

### Example training code

```
python train_offline_imitate.py --env_name=<D4RL env name> --config=configs/<corresponding-config>  --eval_episodes=10 --eval_interval=5000 --temp=<see paper> --beta=<see paper> --exp_name=<experiment name> --expert_trajectories=<optional> --seed=4 
```

For example:

```
python train_offline_imitate.py --env_name=hopper-random-v2 --config=configs/mujoco_config.py  --eval_episodes=10 --eval_interval=5000 --temp=5.0
```

### Hyperparameter tuning guide

A general offline way to tune hyperparamters for dual RL methods is to keep increasing pessimism (increase temp; try 2.5,5.0,10.0,20.0) until the Q functions no longer diverge. Some environments where Q function diverges, adding layernorm can help too. I have not observed layernorm to reliably help across all environments, hence it is kept off by default in the config.



### Citation
```
@misc{sikchi2023dual,
      title={Dual RL: Unification and New Methods for Reinforcement and Imitation Learning}, 
      author={Harshit Sikchi and Qinqing Zheng and Amy Zhang and Scott Niekum},
      year={2023},
      eprint={2302.08560},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Acknowledgement and Reference
This code was built on top of the IQL codebase [here](https://github.com/ikostrikov/implicit_q_learning).



## Questions
Please feel free to email us if you have any questions. 

Harshit Sikchi ([hsikchi@utexas.edu](mailto:hsikchi@utexas.edu?subject=[GitHub]%ReCOIL))
