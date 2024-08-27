# ReCOIL - Imitation Learning from expert data + suboptimal offline data.

## How to run the code

### Install dependencies

These are the same setup instructions as in [Implicit Q-Learning](https://github.com/ikostrikov/implicit_q_learning).



```
conda env create -f environment.yml

conda install -c conda-forge cudnn

pip install --upgrade pip

# Install 1 of the below jax versions depending on your CUDA version
## 1. CUDA 12 installation
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

## 2. CUDA 11 installation
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


```

Also, see other configurations for CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

### Example training code

```
python train_offline_imitate.py --env_name=<D4RL env name> --config=configs/<corresponding-config>  --eval_episodes=10 --eval_interval=5000 --temp=<see paper> --beta=<see paper> --exp_name=<experiment name> --expert_trajectories=<optional> --seed=4 
```


### 
This code was built on top of the IQL codebase [here](https://github.com/ikostrikov/implicit_q_learning).
