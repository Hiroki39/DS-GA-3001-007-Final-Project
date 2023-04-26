import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


ppo_data = np.load('eval_results/ppo/evaluations.npz')
a2c_data = np.load('eval_results/a2c/evaluations.npz')
dqn_data = np.load('eval_results/dqn/evaluations.npz')

ppo_train_df = pd.read_csv('monitor_logs/ppo/train.monitor.csv', skiprows=1)
a2c_train_df = pd.read_csv('monitor_logs/a2c/train.monitor.csv', skiprows=1)
dqn_train_df = pd.read_csv('monitor_logs/dqn/train.monitor.csv', skiprows=1)

ppo_df = pd.read_csv('test_logs/ppo.csv')
a2c_df = pd.read_csv('test_logs/a2c.csv')
dqn_df = pd.read_csv('test_logs/dqn.csv')
greedy_df = pd.read_csv('test_logs/greedy.csv')
random_df = pd.read_csv('test_logs/random.csv')

ppo_df["reward"] = ppo_df['travel_reward'] + ppo_df['ride_reward']
a2c_df["reward"] = a2c_df['travel_reward'] + a2c_df['ride_reward']
dqn_df["reward"] = dqn_df['travel_reward'] + dqn_df['ride_reward']
random_df["reward"] = random_df['travel_reward'] + random_df['ride_reward']
greedy_df["reward"] = greedy_df['travel_reward'] + greedy_df['ride_reward']

random_avg = random_df.groupby('current_date')['reward'].sum().mean()
greedy_avg = greedy_df.groupby('current_date')['reward'].sum().mean()

sns.set_style("whitegrid")

# Produce Training Reward Progress Plot

plt.plot(ppo_train_df["r"], label="PPO", alpha=0.8)
plt.plot(a2c_train_df["r"], label="A2C", alpha=0.8)
plt.legend()
plt.xlabel("Training Episode")
plt.ylabel("Episode Reward")
plt.title("Training Reward Over Time")
plt.save("images/train_reward_progress.png")

plt.clf()

# Produce Evaluation Reward Progress Plot

plt.ticklabel_format(useOffset=False)
plt.plot(ppo_data["timesteps"], ppo_data["results"].mean(
    axis=1), label="PPO", marker='o')
plt.plot(a2c_data["timesteps"], a2c_data["results"].mean(
    axis=1), label="A2C", marker='v')
plt.xticks(ppo_data["timesteps"], labels=ppo_data["timesteps"], rotation=45)

# draw random and greedy baselines
plt.axhline(y=random_avg, color='r', linestyle='--', label="Random Baseline")
plt.axhline(y=greedy_avg, color='g', linestyle='--', label="Greedy Baseline")

plt.xlabel("Timestep")
plt.ylabel("Average Reward")
plt.legend()
plt.title("Average Evaluation Reward")
plt.save("images/eval_reward_progress.png")

plt.clf()

df.groupby('current_date')['reward'].sum()


def get_durations(df, bar_chart=False):
    result = np.zeros([4, 15])
    df["valid_ride_duration"] = df["ride_duration"] * (df.ride_reward > 0)
    df = df[["current_date", "wait_duration", "ride_duration", "travel_duration",
             "valid_ride_duration"]].groupby("current_date").agg("sum")

    if not bar_chart:
        result[0] = df.valid_ride_duration.to_numpy()
        result[1] = df.wait_duration.to_numpy()
        result[2] = df.travel_duration.to_numpy()
        result[3] = df.ride_duration + result[1] + result[2]

    else:
        result = np.zeros(3)
        sum_time = df.ride_duration.to_numpy() + df.wait_duration.to_numpy() + \
            df.travel_duration.to_numpy()
        result[0] = (df.ride_duration.to_numpy()/sum_time).sum()
        result[1] = (df.wait_duration.to_numpy()/sum_time).sum()
        result[2] = (df.travel_duration.to_numpy()/sum_time).sum()

    return result


time_dict = {}
for i in ["a2c", "dqn", "ppo"]:
    df = pd.read_csv("./test_logs/" + i + ".csv")
    time_dict[i] = get_durations(df)

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
plt.title("Percentage of valid ride time in testing episodes")
plt.plot(np.arange(15), time_dict["ppo"][0] /
         time_dict["ppo"][3], label="ppo", marker="o")
plt.plot(np.arange(15), time_dict["a2c"][0] /
         time_dict["a2c"][3], label="a2c", marker="v")
plt.plot(np.arange(15), time_dict["dqn"][0] /
         time_dict["dqn"][3], label="dqn", marker="s")
plt.xlabel("Episode")
plt.ylabel("Percentage of time")

plt.legend()

names = ["DQN", "A2C", "PPO"]
r = [time_dict[key][0] for key in time_dict.keys()]
w = [time_dict[key][1] for key in time_dict.keys()]
t = [time_dict[key][2] for key in time_dict.keys()]

plt.figure(figsize=(10, 5))

plt.title("Time breakdown for agents in testing episodes")
plt.barh(names, r, label="ride time (both valid and invalid)", color="mediumblue")

plt.barh(names, w, left=r, label="wait time", color="skyblue")

plt.barh(names, t, left=np.array(w)+np.array(r),
         label="travel time", color="cyan")
plt.xlabel("Time")
plt.legend()


def plot_test_reward(arr_reward, agent_name, markermap, linemap):
    sns.set_style("whitegrid")

    plt.figure(figsize=(7, 9))

    plt.title("Reward for agents in test episodes")
    for i, __ in enumerate(arr_reward):
        plt.plot(np.arange(len(arr_reward[i])), arr_reward[i],
                 label=agent_name[i], marker=markermap[i], linestyle=linemap[i])
        # plt.axhline(y = np.mean(arr_reward[i]),ls = "dashed")

    plt.xlabel("Days")
    plt.ylabel("Reward")
    plt.ylim([-100, 600])
    plt.legend()
    plt.show()


agent_list = []
name = ["ppo", "a2c", "dqn", "random", "greedy"]

# cmap = ["red","green","pink","orange","blue"]
markermap = ["o", "v", "s", "x", "x"]
linemap = ["-", "-", "-", ":", ":"]

for i in name:
    df = pd.read_csv("./test_logs/"+i+".csv")
    df["step_reward"] = df["ride_reward"]-df["travel_reward"]
    agent_list += [df[["current_date", "step_reward"]
                      ].groupby("current_date").agg("sum").step_reward.to_numpy()]

plot_test_reward(agent_list, name, markermap, linemap)
plt.title("Mean reward in test episode")
plt.xlabel("Mean Reward")
plt.barh(name, np.mean(agent_list, axis=1), color=[
         'purple', 'tab:red', 'green', 'tab:orange', 'tab:blue'][::-1])
