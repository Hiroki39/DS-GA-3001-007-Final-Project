import gym
import disneyenv

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO, DQN, A2C
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str,
                    choices=["ppo", "dqn", "a2c"], default="ppo")
args = parser.parse_args()


def get_train_env():
    train_env = gym.make("disneyenv/Disney-v0", train=True)
    return train_env


def get_eval_env():
    eval_env = gym.make("disneyenv/Disney-v0", train=False)
    return eval_env


train_env = VecMonitor(SubprocVecEnv([lambda: get_train_env()
                                      for i in range(32)], start_method="fork"), filename=f"./monitor_logs_{args.algo}/train", info_keywords=("current_date",))
train_env.seed(42)

eval_env = VecMonitor(SubprocVecEnv([lambda: get_eval_env()
                                     for i in range(32)], start_method="fork"), filename=f"./monitor_logs_{args.algo}/eval", info_keywords=("current_date",))
eval_env.seed(42)

if args.algo == "ppo":
    model = PPO("MultiInputPolicy", train_env, verbose=1, device="cpu")
elif args.algo == "dqn":
    model = DQN("MultiInputPolicy", train_env, verbose=1, device="cpu")
elif args.algo == "a2c":
    model = A2C("MultiInputPolicy", train_env, verbose=1, device="cpu")


eval_callback = EvalCallback(eval_env, best_model_save_path=f"./eval_results_{args.algo}/",
                             log_path=f"./eval_results_{args.algo}/", eval_freq=5000, n_eval_episodes=15)

model.learn(total_timesteps=1000000, callback=eval_callback)
