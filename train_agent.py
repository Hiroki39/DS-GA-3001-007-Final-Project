import gym
import disneyenv

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO


def get_train_env():
    train_env = gym.make("disneyenv/Disney-v0", train=True)
    return train_env


def get_eval_env():
    eval_env = gym.make("disneyenv/Disney-v0", train=False)
    return eval_env


train_env = VecMonitor(SubprocVecEnv([lambda: get_train_env()
                                      for i in range(32)], start_method="fork"), filename="./monitor_logs/train", info_keywords=("current_date",))
eval_env = VecMonitor(SubprocVecEnv([lambda: get_eval_env()
                                     for i in range(32)], start_method="fork"), filename="./monitor_logs/eval", info_keywords=("current_date",))

model = PPO("MultiInputPolicy", train_env, verbose=1, device="cpu")


eval_callback = EvalCallback(eval_env, best_model_save_path="./eval_results/",
                             log_path="./eval_results/", eval_freq=5000, n_eval_episodes=15)

model.learn(total_timesteps=1000000, callback=eval_callback)
