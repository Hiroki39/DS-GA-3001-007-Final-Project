import numpy as np
import pandas as pd


import gym
from datetime import datetime, timedelta
from gym.spaces import Discrete, Box

from scipy.spatial.distance import squareform, pdist
import geopy.distance


def locate_event(time: datetime, time_array, event_type: str):
    # time is a datetime object

    if len(time_array) == 0:  # There is no waittime for a ride at the day
        return 999

    delta_time_array = np.array([(time-i).total_seconds() for i in time_array])
    # only use negative index values
    delta_time_array[delta_time_array > 0] = 999

    if event_type == "waittime":
        threshold_time = 1800
    elif event_type == "weather":
        threshold_time = 4800  # different event update time at different interval
    else:
        raise ValueError("event_type can only be 'waitime' or 'weather'")

    # if the time selected is less than threshold
    if np.min(np.abs(delta_time_array)) < threshold_time:
        return np.argmin(np.abs(delta_time_array))
    # the selected time is so far from the actual time
    else:
        return 999


class DisneyEnv(gym.Env):
    def __init__(self, **kwargs):

        # Dataframe for extracting data
        self.waittime = pd.read_csv(
            "disneyenv/disneyenv/envs/data/disneyRideTimes.csv")
        self.waittime["dateTime"] = pd.to_datetime(self.waittime["dateTime"])
        self.waittime["date"] = self.waittime["dateTime"].dt.date

        self.weather = pd.read_csv(
            "disneyenv/disneyenv/envs/data/hourlyWeather.csv")
        self.weather["dateTime"] = pd.to_datetime(self.weather["dateTime"])
        self.weather["date"] = self.weather["dateTime"].dt.date

        self.ridesinfo = pd.read_csv(
            "disneyenv/disneyenv/envs/data/rideDuration.csv")
        self.rides = self.ridesinfo["id"].unique()
        self.avalible_dates = self.waittime["date"].unique()
        self.observation = None

        # Action space
        # -1 indicates wait for 10 min
        self.__all_actions = np.append(np.arange(len(self.rides)), -1)

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
        self.waittime_today = None
        self.weather_today = None
        self.past_actions = None
        self.current_reward = None

        # Mandatory field for inheriting gym.Env
        # self.observation_space = spaces.Discrete(231)
        self.observation_space = Box(
            low=-10, high=1000, shape=(231,), dtype=np.float64)
        # self.action_space = Discrete(
        #     len(self.__all_actions) - 1, start=-1)
        self.action_space = Discrete(len(self.__all_actions) - 1)

        # reward
        self.reward_dict = {
            "MAA": 20,
            "MIA": 10,
            "HD": 20
        }

    def __get_observation(self):
        # for each attraction, return its wait time
        waittime = np.array([])
        for ride_id in self.rides:
            event = locate_event(
                self.current_time, self.waittime_today[self.waittime_today.rideID == ride_id]["dateTime"], "waittime")
            if event == 999:
                t = 999
            else:
                t = self.waittime_today[self.waittime_today.rideID == ride_id].iloc()[
                    event]["waitMins"]
                if np.isnan(t):
                    t = 999
            waittime = np.append(waittime, t)

        # distance
        distance = self.adjacency_matrix[self.ridesinfo.iloc()[
            self.current_location].landID]

        # weather
        event = locate_event(
            self.current_time, self.weather_today.dateTime, "weather")
        if event == 999:
            weather = [0, 50]
        else:
            weather = [self.weather_today.iloc()[event]["rainStatus"],
                       self.weather_today.iloc()[event]["feelsLikeF"]]

        self.observation = np.hstack(
            [waittime, distance, weather, self.past_actions])
        return self.observation

    def reset(self):
        '''
        OBSERVATIONS: 
        Waittime: Waittime for 110 rides [106]
        Distance: Distance to 18 lands [18]
        Weather: temperature and precipitation [2]
        Past Actions: A vector of 0 and 1 showing rides haven't been done [106]
        '''
        # reset past actions: 0 visits to any of the rides
        self.past_actions = np.zeros(len(self.rides))

        while True:
            # initialize the date and location
            self.current_date = np.random.choice(self.avalible_dates)

            # locate the date
            self.waittime_today = self.waittime[self.waittime.date == self.current_date].copy(
            )

            if (self.waittime_today is None) or (len(self.waittime_today) == 0):
                continue

            break

        # start from 8:00 am
        self.current_time = datetime(
            self.current_date.year, self.current_date.month, self.current_date.day, 8, 0)
        # sample without replacement
        # self.avalible_dates.remove(self.current_date)

        # The location of disney gallery, which locates at the entrance of the disneyland
        self.current_location = 61

        self.weather_today = self.weather[self.weather.date == self.current_date].copy(
        )

        self.current_reward = 0

        self.observation = self.__get_observation()

        print("A new day! Today is " +
              self.current_date.strftime("%Y-%m-%d"))
        return self.observation

    def step(self, action: int):  # action is the index of the ride. Not the ride ID

        valid_action = True
        # print(self.observation[action])
        # REWARD
        if action == -1:
            reward = 0
        elif (self.observation[action] == 999) or (np.isnan(self.observation[action])):
            reward = -50
            valid_action = False
        else:
            popularity = self.ridesinfo.iloc()[action]["popularity"]
            if type(popularity) == str:
                reward = self.reward_dict[popularity]
            else:  # popularity is none
                reward = 1

            # discount reward
            reward /= (2**(self.past_actions[action]))

        # STATE
        # update pass actions
        if action == -1:  # wait
            travel_duration = 0
            wait_duration = 10
            ride_duration = 0
        elif not valid_action:  # choose a ride but it doesn't open
            travel_duration = self.adjacency_matrix[self.ridesinfo.iloc[action]
                                                    .landID][self.ridesinfo.iloc[self.current_location].landID]
            wait_duration = 10
            ride_duration = 0
        else:
            assert action != -1
            self.past_actions[action] += 1
            # Next time stamp
            travel_duration = self.adjacency_matrix[self.ridesinfo.iloc[action]
                                                    .landID][self.ridesinfo.iloc[self.current_location].landID]
            # self.observation is a attribute since we need to use it here
            wait_duration = self.observation[action]
            ride_duration = self.ridesinfo.duration_min[action]

        self.current_time += timedelta(minutes=(travel_duration +
                                       wait_duration+ride_duration))
        self.observation = self.__get_observation()
        info = {}
        terminated = self.current_time.hour > 22

        # update location
        if action != -1:
            self.current_location = action

        if terminated:
            print("The day is over! The reward is " + str(self.current_reward))

        return self.observation, reward, terminated, info
    '''
    def get_action_space(self):
        # method that returns possible actions.
        if self.observation is None:
            self.action_space.n = 1
            return np.array([])
        else:
            possible_action = []
            for index,content in enumerate(self.__all_actions):
                if content == -1:
                    possible_action += [content]
                elif (self.observation[index] != 999) and (~np.isnan(self.observation[index])):
                    possible_action += [content]

            self.action_space.n = len(possible_action)
            return np.array(possible_action)
    '''

    def close():
        pass
