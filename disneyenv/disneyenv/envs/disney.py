import numpy as np
import pandas as pd

import gymnasium as gym
from datetime import datetime
from gymnasium import error, spaces, utils


def locate_event(time: datetime, time_array):
    # time is a datetime object
    delta_time_array = np.array([(time-i).total_seconds() for i in time_array])
    # only use negative index values
    delta_time_array[(delta_time_array > 0)] = np.inf
    return np.argmin(np.abs(delta_time_array))


def find_distance(location: int):
    '''
    adjacency_matrix will store the distance from one place to all other places

    '''
    return adjacency_matrix[location]


class DisneyEnv(gym.Env):
    def __init__(self):

        # Dataframe for extracting data
        self.waittime_df = pd.read_csv("./data/disneyRideTimes.csv")
        self.weather_df = pd.read_csv("./data/hourlyWeather.csv")
        self.rides_df = pd.read_csv("./data/rideDuration.csv")
        self.parksched_df = pd.read_csv("./data/parkSchedules.csv")

        # process the data
        self.waittime_df["dateTime"] = pd.to_datetime(
            self.waittime_df["dateTime"])
        self.weather_df["dateTime"] = pd.to_datetime(
            self.weather_df["dateTime"])
        self.rides = self.rides_df["id"].values
        self.avalible_dates = self.waittime_df["dateTime"].dt.date.unique()

        # The current day we are in
        self.current_date = np.random.choice(list(self.avalible_dates))
        self.current_time = None
        self.current_location = 17734741
        self.waittime_today = None
        self.weather_today = None

        # track the count of past visits
        self.past_visits = np.zeros(len(self.rides))

        self.observation_space = spaces.Discrete(len(self.rides) + 1)

    def _get_obs(self):
        return {"agent": self._agent_location, "time": self._current_timestamp}

    def reset(self):
        '''
        OBSERVATIONS: 
        Waittime: Waittime for 110 rides [110]
        Distance: Distance to 18 lands [18]
        Weather: temperature and precipitation [2]
        Past Actions: A vector of 0 and 1 showing rides haven't been done [110]
        '''
        # reset past visits
        self.past_visits = np.zeros(len(self.rides))

        # initialize the day and location
        self.current_date = np.random.choice(list(self.avalible_dates))
        self.current_time = datetime.strftime(self.day + " 08:00:00")
        # self.avalible_days.remove(self.current_day)
        # The location of disney gallery, which locates at the entrance of the disneyland
        self.current_location = 17734741

        # locate the day
        self.waittime_today = self.waittime[self.waittime.day ==
                                            self.current_day]
        self.weather_today = self.weather[self.weather.day == self.current_day]

        # for each attraction, return its wait time
        waittime = np.array([])
        for ride_id in self.rides:
            event = locate_event(
                self.current_time, self.waittime_today[self.waittime_today.rideID == ride_id].dateTime)
            waittime = np.append(
                waittime, self.waittime_today.iloc()[event].waitMins)

        # distance
        distance = find_distance(self.current_location)

        # weather
        event = locate_event(self.current_time, self.weather_today.dateTime)
        weather = [self.weather_today.iloc()[event].rainStatus,
                   self.weather_today.iloc()[event].feelsLikeF]

        observation = np.vstack(
            [waittime, distance, weather, self.past_actions])
        return observation

    def step(self, action):

        return observation, reward, terminated, False, info

    def close():
        pass
