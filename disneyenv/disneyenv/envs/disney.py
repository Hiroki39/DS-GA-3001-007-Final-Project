import numpy as np
import pandas as pd
from collections import OrderedDict

import warnings
import gym
from datetime import datetime, timedelta
from gym.spaces import Discrete, Box, Dict, MultiBinary


warnings.filterwarnings('ignore')


class DisneyEnv(gym.Env):
    def __init__(self, train, eval_days=15, agent_id=0):

        # Dataframe for extracting data
        self.waitTime = pd.read_csv(
            "disneyenv/disneyenv/envs/data/disneyRideTimes.csv")
        self.waitTimeMax = self.waitTime.waitMins.max()
        self.waitTime["dateTime"] = pd.to_datetime(self.waitTime["dateTime"])
        self.waitTime["date"] = self.waitTime["dateTime"].dt.date
        self.waitTime = self.waitTime.set_index(["rideID", "dateTime"])

        self.weather = pd.read_csv(
            "disneyenv/disneyenv/envs/data/hourlyWeather.csv")
        self.weather["dateTime"] = pd.to_datetime(self.weather["dateTime"])
        self.weather["date"] = self.weather["dateTime"].dt.date
        self.weather = self.weather.set_index("dateTime")
        self.tempRange = [
            self.weather.feelsLikeF.min(), self.weather.feelsLikeF.max()]

        self.ridesinfo = pd.read_csv(
            "disneyenv/disneyenv/envs/data/rideDuration.csv")
        self.rides = self.ridesinfo["id"].unique()
        self.train = train

        np.random.seed(42 + agent_id)
        self.agent_id = agent_id

        if train:
            self.avalible_dates = np.sort(
                self.waitTime["date"].unique())[:-eval_days]
        else:
            # for evaluation, we only use the last eval_days days
            self.avalible_dates = np.sort(
                self.waitTime["date"].unique())[-eval_days:]
            self.eval_idx = 0

        # Action space
        # len(self.rides) indicates wait for 10 min
        self.__all_actions = np.arange(len(self.rides) + 1)
        self.action_space = Discrete(len(self.__all_actions))

        # walking time obtained via google map api
        self.adjacency_matrix = np.load(
            "disneyenv/disneyenv/envs/data/walking_time.npy")

        # waitTime and feelsLikeF are normalized
        # progress indicates how much of today's time has passed
        self.observation_space = Dict(
            {
                "waitTime": Box(low=0, high=1, shape=(len(self.rides),), dtype=np.float64),
                "operationStatus": MultiBinary(len(self.rides)),
                "currentLand": Discrete(len(self.adjacency_matrix)),
                "progress": Box(low=0, high=1, shape=(1,), dtype=np.float64),
                "rainStatus": Discrete(6),
                "feelsLikeF": Box(low=0, high=1, shape=(1,), dtype=np.float64),
                "pastActions": MultiBinary(len(self.rides))
            }
        )

        # reward
        self.reward_dict = {
            "MAA": 20,
            "MIA": 10,
            "HD": 20
        }

        # convert rain status to 0-5
        self.rain_mapping = {
            0: 0,
            5: 1,
            6: 2,
            14: 3,
            10: 4,
            2: 5
        }

    def __get_observation(self):

        # for each attraction, return its wait time
        input_waitTime_index = pd.MultiIndex.from_arrays(
            [np.array(self.rides), np.repeat(self.current_time, len(self.rides))])

        waitTime, operationStatus = self.retrieve_closest_prior_info(
            input_waitTime_index, self.waitTime, "waitTime")

        # weather
        rainStatus, feelsLikeF = self.retrieve_closest_prior_info(
            [self.current_time], self.weather_today, "weather")

        # normalize waitTime and feelsLikeF
        waitTime = waitTime / self.waitTimeMax
        feelsLikeF = (feelsLikeF - self.tempRange[0]) / \
            (self.tempRange[1] - self.tempRange[0])

        # compute progress -- (current - 8am) / (10pm - 8am)
        progress = ((self.current_time - datetime.combine(self.current_time.date(),
                    datetime.min.time())).total_seconds() - 28800) / 50400

        self.observation = OrderedDict([
            ("waitTime", waitTime),
            ("operationStatus", operationStatus),
            ("currentLand", self.current_land),
            ("progress", [progress]),
            ("rainStatus", rainStatus),
            ("feelsLikeF", [feelsLikeF]),
            ("pastActions", self.past_actions)
        ])

        return self.observation

    def retrieve_closest_prior_info(self, input_index, target_df, event_type: str):

        # get iloc of the target indexes
        target_ilocs = target_df.index.get_indexer(input_index, method="ffill")
        # index of valid ilocs
        valid_targets = np.where(target_ilocs != -1)[0]

        if event_type == "waitTime":
            # make sure that the rideID is the same
            valid_targets = valid_targets[
                target_df.index[target_ilocs[valid_targets]].get_level_values(
                    0) == input_index[valid_targets].get_level_values(0)]

            waitTime = np.repeat(np.nan, len(input_index))
            waitTime[valid_targets] = target_df.iloc[target_ilocs[valid_targets]
                                                     ].waitMins.values

            # if the waitTime is nan, assume it is 0
            waitTime = np.nan_to_num(waitTime)

            # 0 for not operating, 1 for operating
            operationStatus = np.repeat(0, len(input_index))
            operationStatus[valid_targets] = (
                target_df.iloc[target_ilocs[valid_targets]].status == "Operating")

            return waitTime, operationStatus

        elif event_type == "weather":

            # on average, the feelsLikeF is 68
            if len(valid_targets) == 0:
                return 0, 68

            else:
                rainstatus = target_df.iloc[target_ilocs[0]].rainStatus
                feelsLikeF = target_df.iloc[target_ilocs[0]].feelsLikeF

                return self.rain_mapping[rainstatus], feelsLikeF

    def reset(self):
        '''
        OBSERVATIONS:
        Waittime: Waittime for 110 rides [106]
        Distance: Distance to 18 lands [18]
        Weather: temperature and precipitation [2]
        Past Actions: A vector of 0 and 1 showing rides haven't been done [106]
        '''
        # reset past actions: 0 visits to any of the rides
        self.past_actions = np.zeros(len(self.rides), dtype=bool)

        if self.train:
            # choose a date with available data, sample with replacement
            while True:
                # initialize the date and location
                self.current_date = np.random.choice(self.avalible_dates)

                # locate the date
                self.waitTime_today = self.waitTime[self.waitTime.date == self.current_date].copy(
                )

                if (self.waitTime_today is None) or (len(self.waitTime_today) == 0):
                    continue

                break
        else:
            self.current_date = self.avalible_dates[self.eval_idx]

            # locate the date
            self.waitTime_today = self.waitTime[self.waitTime.date == self.current_date].copy(
            )

            self.eval_idx = (self.eval_idx + 1) % len(self.avalible_dates)

        # filter the weather data
        self.weather_today = self.weather[self.weather.date == self.current_date].copy(
        )

        # start from 8:00 am
        self.current_time = datetime(
            self.current_date.year, self.current_date.month, self.current_date.day, 8, 0)

        # The location of disney gallery, which locates at the entrance of the disneyland
        self.current_location = 61
        self.current_land = self.ridesinfo.iloc[self.current_location].landID

        # reset the reward
        self.current_reward = 0

        # initialize the observation
        self.observation = self.__get_observation()

        # print(f"A new day! Today is {self.current_date.strftime('%Y-%m-%d')}")

        return self.observation

    def step(self, action: int):  # action is the index of the ride. Not the ride ID

        if action == len(self.rides):
            # do nothing and wait for 10 minutes
            self.current_time += timedelta(minutes=10)

            wait_duration = 10
            ride_duration = 0
            travel_duration = 0

            ride_reward = 0
            travel_reward = 0

        else:

            # visit the ride
            travel_duration = self.adjacency_matrix[self.current_land][self.ridesinfo.iloc[action].landID]

            # if ride and current location in same land, assume 1 min travel
            if travel_duration == 0:
                travel_duration = 1

            # get actual waiting time and operation status upon arrival
            self.current_time += timedelta(minutes=travel_duration)
            waitTime, operationStatus = self.retrieve_closest_prior_info(pd.MultiIndex.from_arrays(
                [[self.rides[action]], [self.current_time]]), self.waitTime_today, "waitTime")

            # if the ride is operating
            if operationStatus[0]:

                # assign reward based on popularity, no reward if the ride has been done before
                if not self.past_actions[action]:

                    popularity = self.ridesinfo.iloc[action]["popularity"]

                    ride_reward = self.reward_dict[popularity] if type(
                        popularity) == str else 5

                    # update past action record
                    self.past_actions[action] = True

                else:
                    ride_reward = 0

                # get the wait and ride duration
                wait_duration = waitTime[0]
                ride_duration = self.ridesinfo.duration_min[action]

                # update the current time
                self.current_time += timedelta(
                    minutes=wait_duration + ride_duration)

            else:
                # if the ride is not operating, apply a penalty
                ride_reward = -5

                wait_duration = 0
                ride_duration = 0

            # apply small penalty for walking
            travel_reward = -0.2 * travel_duration

            # update current location
            self.current_location = action
            self.current_land = self.ridesinfo.iloc[action].landID

        # compute the final reward
        reward = ride_reward + travel_reward

        # get next observation
        self.observation = self.__get_observation()

        info = {
            "current_date": self.current_date,
            "current_time": self.current_time,
            "current_location": self.current_location,
            "current_land": self.current_land,
            "wait_duration": wait_duration,
            "ride_duration": ride_duration,
            "travel_duration": travel_duration,
            "ride_reward": ride_reward,
            "travel_reward": travel_reward,
            "agent_id": self.agent_id,
        }

        # the visit is over if the time is after 10:00 pm
        terminated = self.current_time > datetime(
            self.current_date.year, self.current_date.month, self.current_date.day, 22, 0)

        # update reward
        self.current_reward += reward

        # if terminated:
        #     print(f"The day is over! The reward is {self.current_reward}")

        return self.observation, reward, terminated, info

    def close():
        pass
