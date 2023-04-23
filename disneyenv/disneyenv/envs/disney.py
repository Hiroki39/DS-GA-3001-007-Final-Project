import numpy as np
import pandas as pd
from collections import OrderedDict

import warnings
import gym
from datetime import datetime, timedelta
from gym.spaces import Discrete, Box, Dict, MultiBinary

from scipy.spatial.distance import squareform, pdist
import geopy.distance


warnings.filterwarnings('ignore')


class DisneyEnv(gym.Env):
    def __init__(self, **kwargs):

        # Dataframe for extracting data
        self.waitTime = pd.read_csv(
            "disneyenv/disneyenv/envs/data/disneyRideTimes.csv")
        self.waitTimeMax = self.waitTime.waitMins.max()
        self.waitTime.waitMins = self.waitTime.waitMins / self.waitTimeMax
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
        self.weather.feelsLikeF = (
            self.weather.feelsLikeF - self.tempRange[0]) / (self.tempRange[1] - self.tempRange[0])

        self.ridesinfo = pd.read_csv(
            "disneyenv/disneyenv/envs/data/rideDuration.csv")
        self.rides = self.ridesinfo["id"].unique()
        self.avalible_dates = self.waitTime["date"].unique()
        self.observation = None

        # Action space
        # len(self.rides) indicates wait for 10 min
        self.__all_actions = np.arange(len(self.rides) + 1)

        # adjacency matrix
        landLocation = pd.read_csv(
            "disneyenv/disneyenv/envs/data/landLocation.csv")
        walking_speed = 0.0804672  # km/min

        self.adjacency_matrix = squareform(pdist(landLocation[[
                                           "longitude", "latitude"]], lambda u, v: geopy.distance.geodesic(u, v).km/walking_speed))

        # The current date we are in
        self.current_date = None
        self.current_time = None
        self.current_location = None
        self.current_land = None
        self.waitTime_today = None
        self.weather_today = None
        self.past_actions = None
        self.current_reward = None

        # Mandatory field for inheriting gym.Env
        # self.observation_space = spaces.Discrete(231)
        self.observation_space = Dict(
            {
                "waitTime": Box(low=0, high=1, shape=(len(self.rides),), dtype=np.float64),
                "operationStatus": MultiBinary(len(self.rides)),
                "currentLand": Discrete(len(self.adjacency_matrix)),
                "rainStatus": Discrete(6),
                "feelsLikeF": Box(low=0, high=1, shape=(1,), dtype=np.float64),
                "pastActions": MultiBinary(len(self.rides))
            }
        )

        self.action_space = Discrete(len(self.__all_actions))

        # reward
        self.reward_dict = {
            "MAA": 20,
            "MIA": 10,
            "HD": 20
        }

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

        # input_waitTime_index = input_waitTime_index.set_levels(
        #     input_waitTime_index.get_level_values(0).astype(np.int64), level=0)

        waitTime, operationStatus = self.retrieve_closest_prior_info(
            input_waitTime_index, self.waitTime, "waitTime")

        # weather
        rainStatus, feelsLikeF = self.retrieve_closest_prior_info(
            [self.current_time], self.weather_today, "weather")

        self.observation = OrderedDict([
            ("waitTime", waitTime),
            ("operationStatus", operationStatus),
            ("currentLand", self.current_land),
            ("rainStatus", rainStatus),
            ("feelsLikeF", [feelsLikeF]),
            ("pastActions", self.past_actions)
        ])

        return self.observation

    def retrieve_closest_prior_info(self, input_index, target_df, event_type: str):

        # get iloc of the target indexes
        if event_type == "waitTime":
            print(input_index.dtypes, target_df.index.dtypes)
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

        while True:
            # initialize the date and location
            self.current_date = np.random.choice(self.avalible_dates)

            # locate the date
            self.waitTime_today = self.waitTime[self.waitTime.date == self.current_date].copy(
            )

            if (self.waitTime_today is None) or (len(self.waitTime_today) == 0):
                continue

            break

        # start from 8:00 am
        self.current_time = datetime(
            self.current_date.year, self.current_date.month, self.current_date.day, 8, 0)
        # sample without replacement
        # self.avalible_dates.remove(self.current_date)

        # The location of disney gallery, which locates at the entrance of the disneyland
        self.current_location = 61
        self.current_land = self.ridesinfo.iloc[self.current_location].landID

        self.weather_today = self.weather[self.weather.date == self.current_date].copy(
        )

        self.current_reward = 0

        self.observation = self.__get_observation()

        print("A new day! Today is " +
              self.current_date.strftime("%Y-%m-%d"))

        return self.observation

    def step(self, action: int):  # action is the index of the ride. Not the ride ID

        # REWARD
        if action == len(self.rides):
            # do nothing and wait for 10 minutes
            reward = 0
            travel_duration = 0
            wait_duration = 10
            ride_duration = 0
        elif not self.observation["operationStatus"][action]:
            # visit a ride that is not operating
            reward = 0
            travel_duration = self.adjacency_matrix[self.ridesinfo.iloc[action]
                                                    .landID][self.ridesinfo.iloc[self.current_location].landID]
            wait_duration = 10
            ride_duration = 0

            # apply small penalty for walking
            reward -= travel_duration * 0.1

        else:
            popularity = self.ridesinfo.iloc[action]["popularity"]

            # assign reward based on popularity
            reward = self.reward_dict[popularity] if type(
                popularity) == str else 5

            # no reward if the ride has been done
            reward = 0 if self.past_actions[action] else reward

            # update past actions
            self.past_actions[action] |= 1

            travel_duration = self.adjacency_matrix[self.ridesinfo.iloc[action]
                                                    .landID][self.ridesinfo.iloc[self.current_location].landID]
            # self.observation is a attribute since we need to use it here
            # scale back to normal time scale
            wait_duration = self.observation["waitTime"][action] * \
                self.waitTimeMax
            ride_duration = self.ridesinfo.duration_min[action]

            # apply small penalty for walking
            reward -= travel_duration * 0.1

        # compute next timestamp
        self.current_time += timedelta(minutes=(travel_duration +
                                       wait_duration + ride_duration))
        self.observation = self.__get_observation()

        info = {}
        terminated = self.current_time.hour > 22

        # update location
        if action != len(self.rides):
            self.current_location = action
            self.current_land = self.ridesinfo.iloc[self.current_location].landID

        # update reward
        self.current_reward += reward

        if terminated:
            print("The day is over! The reward is " + str(self.current_reward))

        return self.observation, reward, terminated, info

    def denormalize_temperature(self, temperature):
        return (temperature*(self.tempRange[1]-self.tempRange[0])) + self.tempRange[0]

    def denormalize_wait_time(self, wait_time):
        return wait_time * self.waitTimeMax

    def close():
        pass
