from typing import Dict

import flax.linen as nn
import gym
import numpy as np
from PIL import Image

def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int, verbose: bool = False) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        observation, done = env.reset(), False

        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)

        for k in stats.keys():
            stats[k].append(info['episode'][k]) 
            if verbose:
                v = info['episode'][k]
                print(f'{k}:{v}')

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats



def evaluate_and_save_video(agent: nn.Module, env: gym.Env, log_folder,
             num_episodes: int, verbose: bool = False) -> Dict[str, float]:
    import cv2
    num_frames = 1000
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = 30
    video_file = log_folder +'/'+ 'policy'+'.mp4' #'/path/to/output/video.mp4'
    stats = {'return': [], 'length': []}
    print("Saving the video to ", video_file)
    for j in range(1):
        
        observation, done = env.reset(), False
        if j==0:
            width, height = env.render(mode='rgb_array').shape[:2]
            writer = cv2.VideoWriter(video_file, fourcc, fps, (width, height))
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation , _, done, info = env.step(action)
            if j==0:
                img = env.render(mode='rgb_array')
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                writer.write(img)
        for k in stats.keys():
            stats[k].append(info['episode'][k]) 
            if verbose:
                v = info['episode'][k]
                print(f'{k}:{v}')
    writer.release()
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats


def evaluate_and_save_video_adroit(agent: nn.Module, env: gym.Env, log_folder,
             num_episodes: int, verbose: bool = False) -> Dict[str, float]:
    import skvideo.io
    video_file = log_folder +'/'+ 'policy'+'.gif' #'/path/to/output/video.mp4'
    stats = {'return': [], 'length': []}
    print("Saving the video to ", video_file)
    frame_size=(640,480)
    arrs = []
    imgs = []
    for j in range(10):
        observation, done = env.reset(), False
        
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation , _, done, info = env.step(action)
            curr_frame = env.sim.render(width=frame_size[0], height=frame_size[1],
                                             mode='offscreen', camera_name="fixed", device_id=0)
            arrs.append(curr_frame[::-1,:,:])
            imgs.append(Image.fromarray(curr_frame[::-1,:,:]))

        for k in stats.keys():
            stats[k].append(info['episode'][k]) 
            if verbose:
                v = info['episode'][k]
                print(f'{k}:{v}')
    # skvideo.io.vwrite( video_file, np.asarray(arrs))
    # import ipdb;ipdb.set_trace()
    # imgs = np.array(imgs)
    imgs[0].save(video_file, save_all=True,
        append_images=imgs[1:], duration=45, loop=0)
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats


def evaluate_and_save_video_kitchen(agent: nn.Module, env: gym.Env, log_folder,
             num_episodes: int, verbose: bool = False) -> Dict[str, float]:
    import skvideo.io
    video_file = log_folder +'/'+ 'policy'+'.gif' #'/path/to/output/video.mp4'
    stats = {'return': [], 'length': []}
    print("Saving the video to ", video_file)
    frame_size=(640,480)
    arrs = []
    imgs = []
    for j in range(2):
        observation, done = env.reset(), False
        
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation , _, done, info = env.step(action)
            # import ipdb;ipdb.set_trace()
            curr_frame = env.render(mode='rgb_array')
            # curr_frame = env.sim.render(width=frame_size[0], height=frame_size[1])
            arrs.append(curr_frame)
            imgs.append(Image.fromarray(curr_frame))

        for k in stats.keys():
            stats[k].append(info['episode'][k]) 
            if verbose:
                v = info['episode'][k]
                print(f'{k}:{v}')
    # skvideo.io.vwrite( video_file, np.asarray(arrs))
    # imgs = np.array(imgs)
    imgs[0].save(video_file, save_all=True,
        append_images=imgs[1:], duration=45, loop=0)
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats

def evaluate_ant(agent: nn.Module, env: gym.Env,
             num_episodes: int, offline_min, offline_max, verbose: bool = False) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        # import ipdb; ipdb.set_trace()
        observation[:2] = (observation[:2] - offline_min) / (offline_max - offline_min)
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)

        for k in stats.keys():
            stats[k].append(info['episode'][k]) 
            if verbose:
                v = info['episode'][k]
                print(f'{k}:{v}')

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
