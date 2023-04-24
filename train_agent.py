import gym
import disneyenv

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, ProgressBarCallback
from stable_baselines3 import PPO


def get_train_env():
    train_env = gym.make("disneyenv/Disney-v0", train=True)
    train_env = Monitor(train_env, filename="./monitor_logs/train",
                        info_keywords=("current_date",))
    return train_env


def get_eval_env():
    eval_env = gym.make("disneyenv/Disney-v0", train=False)
    eval_env = Monitor(eval_env, filename="./monitor_logs/eval",
                       info_keywords=("current_date",))
    return eval_env


train_env = DummyVecEnv([get_train_env])
eval_env = DummyVecEnv([get_eval_env])

model = PPO("MultiInputPolicy", train_env, verbose=1, device="cpu")


eval_callback = EvalCallback(eval_env, best_model_save_path="./eval_results/",
                             log_path="./eval_results/", eval_freq=5000, n_eval_episodes=15)

pb_callback = ProgressBarCallback()

model.learn(total_timesteps=100000, callback=CallbackList(
    [eval_callback, pb_callback]))
