import gym
import disneyenv

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, DQN, A2C
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str,
                    choices=["ppo", "dqn", "a2c"], default="ppo")
args = parser.parse_args()


def get_train_env(agent_id):
    train_env = gym.make("disneyenv/Disney-v0", train=True, agent_id=agent_id)
    return train_env


def get_eval_env():
    eval_env = gym.make("disneyenv/Disney-v0", train=False)
    return eval_env


train_env = VecMonitor(SubprocVecEnv([lambda i=i: get_train_env(i)
                                      for i in range(64)], start_method="fork"), filename=f"./monitor_logs/{args.algo}/train", info_keywords=("current_date", "agent_id"))

eval_env = Monitor(get_eval_env(
), filename=f"./monitor_logs/{args.algo}/eval", info_keywords=("current_date",))

if args.algo == "ppo":
    model = PPO("MultiInputPolicy", train_env, verbose=1, device="cpu")
elif args.algo == "dqn":
    model = DQN("MultiInputPolicy", train_env, verbose=1, device="cpu")
elif args.algo == "a2c":
    model = A2C("MultiInputPolicy", train_env, verbose=1, device="cpu")

eval_callback = EvalCallback(eval_env, best_model_save_path=f"./eval_results/{args.algo}/",
                             log_path=f"./eval_results/{args.algo}/", eval_freq=5000, n_eval_episodes=15)

model.learn(total_timesteps=10000000, callback=eval_callback)
