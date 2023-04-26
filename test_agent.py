import gym
import csv
import numpy as np
import disneyenv

from stable_baselines3 import PPO, DQN, A2C
import argparse


class RandomAgent:

    def __init__(self, env):
        self.action_space = env.action_space

    def predict(self, obs, deterministic=True):
        action = self.action_space.sample()
        return action, None


class GreedyAgent:

    def __init__(self, env):
        self.action_space = env.action_space
        self.adjacency_matrix = env.adjacency_matrix
        self.landID_arr = env.ridesinfo.landID.to_numpy()
        self.reward_arr = env.ridesinfo.popularity.apply(
            lambda x: 5 if type(x) != str else env.reward_dict[x]).to_numpy()
        self.ride_duration_arr = env.ridesinfo.duration_min.to_numpy()

    def predict(self, obs, deterministic=True):
        indicies = np.where(
            (obs["operationStatus"] + ~obs["pastActions"]) == 2)[0]
        if len(indicies) == 0:
            return self.action_space.n - 1, None

        land_travel_times = env.adjacency_matrix[obs["currentLand"], :]
        travel_times_arr = land_travel_times[self.landID_arr]

        travel_times_arr[travel_times_arr == 0] = 1

        time_arr = obs["waitTime"][indicies] + \
            self.ride_duration_arr[indicies] + travel_times_arr[indicies]
        reward_arr = self.reward_arr[indicies]

        tmp = np.argmax(reward_arr / time_arr)

        action = indicies[tmp]

        return action, None


parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str,
                    choices=["ppo", "dqn", "a2c", "greedy", "random"], default="ppo")
args = parser.parse_args()

algo = args.algo

env = gym.make("disneyenv/Disney-v0", train=False)

if algo == "ppo":
    model = PPO.load(f"./eval_results/{algo}/best_model.zip")
elif algo == "dqn":
    model = DQN.load(f"./eval_results/{algo}/best_model.zip")
elif algo == "a2c":
    model = A2C.load(f"./eval_results/{algo}/best_model.zip")
elif algo == "greedy":
    model = GreedyAgent(env)
elif algo == "random":
    model = RandomAgent(env)

with open(f'test_logs/{algo}.csv', 'w') as f:

    fieldnames = ['current_date', 'current_time', 'current_location', 'current_land',
                  'wait_duration', 'ride_duration', 'travel_duration', 'ride_reward', 'travel_reward', 'agent_id']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(15):

        obs = env.reset()

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            writer.writerow(info)

            if done:
                break
