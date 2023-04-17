import numpy as np
import pandas as pd

import gymnasium as gym
from datetime import datetime,timedelta
from gymnasium import spaces, utils
import geopy.distance


def locate_event(time: datetime, time_array,event_type:str):
    # time is a datetime object

    if len(time_array) == 0: # There is no waittime for a ride at the day
        return 999
    
    delta_time_array = np.array([(time-i).total_seconds() for i in time_array])
    delta_time_array[delta_time_array > 0] = 999 # only use negative index values

    if event_type == "waittime":
        threshold_time = 1800
    elif event_type == "weather":
        threshold_time = 4800 # different event update time at different interval
    else:
        raise ValueError("event_type can only be 'waitime' or 'weather'")

    # if the time selected is less than threshold
    if np.min(np.abs(delta_time_array)) < threshold_time: 
        return np.argmin(np.abs(delta_time_array))
    # the selected time is so far from the actual time
    else:
        return 999


class DisneyEnv(gym.Env):
    def __init__(self):

        # Dataframe for extracting data
        self.waittime = pd.read_csv("./data/disneyRideTimes.csv")
        self.waittime["dateTime"] = self.waittime["dateTime"].apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
        self.waittime["day"] = [str(i.year) + "-" + str(i.month) + "-" + str(i.day) for i in self.waittime["dateTime"]]

        self.weather = pd.read_csv("./data/hourlyWeather.csv")
        self.weather["dateTime"] = self.weather["dateTime"].apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
        self.weather["day"] = [str(i.year) + "-" + str(i.month) + "-" + str(i.day) for i in self.weather["dateTime"]]


        self.ridesinfo = pd.read_csv("./data/rideDuration.csv")
        self.rides = self.ridesinfo.id.to_numpy()
        self.avalible_days = set(self.waittime.day.unique())
        self.observation = None

        # Action space
        self.__all_actions = np.append(np.arange(len(self.rides)),-1) # -1 indicates wait for 10 min

        # adjacency matrix
        landLocation = pd.read_csv("./data/landLocation.csv")
        self.adjacency_matrix =  np.zeros([17,17])
        walking_speed = 0.0804672 # km/min

        for i in range(len(landLocation)):
            start = [landLocation.loc[i]["longitude"],landLocation.loc[i]["latitude"]]
            for j in range(len(landLocation)):
                destination =  [landLocation.loc[j]["longitude"],landLocation.loc[j]["latitude"]]
                self.adjacency_matrix[i][j] = geopy.distance.geodesic(start,destination).km/walking_speed

        # The current day we are in
        self.current_day = None
        self.current_time = None
        self.current_location = None
        self.waittime_today = None
        self.weather_today = None
        self.past_actions = None

        # Mandatory field for inheriting gym.Env
        self.observation_space = spaces.Discrete(231)
        self.action_space = spaces.Discrete(len(self.__all_actions))


        # reward
        self.reward_dict= {
             "MAA":20,
             "MIA":10,
             "HD":20
        }

        
    def __get_observation(self):
        print(self.waittime_today)
        if self.waittime_today is None:
            return np.array([])

        # for each attraction, return its wait time
        waittime = np.array([])
        for ride_id in self.rides:
            event = locate_event(self.current_time,self.waittime_today[self.waittime_today.rideID == ride_id]["dateTime"],"waittime")
            if event == 999:
                t = 999
            else:
                t = self.waittime_today[self.waittime_today.rideID == ride_id].iloc()[event]["waitMins"]
            waittime = np.append(waittime,t)
        
        
        # distance
        distance = self.adjacency_matrix[self.ridesinfo.iloc()[self.current_location].landID]

        # weather
        event = locate_event(self.current_time,self.weather_today.dateTime,"weather")
        if event == 999:
            weather = [0,50]
        else:
            weather = [self.weather_today.iloc()[event]["rainStatus"],self.weather_today.iloc()[event]["feelsLikeF"]]

        self.observation = np.hstack([waittime,distance,weather,self.past_actions])
        return self. observation


    def reset(self):

        '''
        OBSERVATIONS: 
        Waittime: Waittime for 110 rides [106]
        Distance: Distance to 18 lands [18]
        Weather: temperature and precipitation [2]
        Past Actions: A vector of 0 and 1 showing rides haven't been done [106]
        '''
        # reset past actions
        self.past_actions = np.zeros(len(self.rides))

        # initialize the day and location
        self.current_day = np.random.choice(list(self.avalible_days))
        self.current_time = datetime.strptime(self.current_day+ " 08:00:00","%Y-%m-%d %H:%M:%S")
        self.avalible_days.remove(self.current_day)
        self.current_location =  61 # The location of disney gallery, which locates at the entrance of the disneyland

        # locate the day
        self.waittime_today = self.waittime[self.waittime.day == self.current_day].copy()
        self.weather_today = self.weather[self.weather.day == self.current_day].copy()

        self.observation = self.__get_observation()

        print("A new day! Today is "+self.current_day)
        return self.observation

    def step(self,action:int): # action is the index of the ride. Not the ride ID

        valid_action = True
    
        # REWARD
        if action == -1:
            reward = 0
        elif (self.observation[action] == 999) or (self.observation[action] is None):
            reward = -50
            valid_action = False
        else:
            popularity = self.ridesinfo.iloc()[action]["popularity"]
            if type(popularity) == str:
                reward = self.reward_dict[popularity]
            else: # popularity is none
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
            travel_duration = self.adjacency_matrix[self.ridesinfo.iloc()[action].landID][self.ridesinfo.iloc()[self.current_location].landID]
            wait_duration = 0
            ride_duration = 0
        else:
            assert action != -1
            self.past_actions[action] += 1
            # Next time stamp
            travel_duration = self.adjacency_matrix[self.ridesinfo.iloc()[action].landID][self.ridesinfo.iloc()[self.current_location].landID]
            wait_duration = self.observation[action] # self.observation is a attribute since we need to use it here
            ride_duration = self.ridesinfo.duration_min[action]
        
        self.current_time += timedelta(minutes=(travel_duration+wait_duration+ride_duration))

        # if the day is done
        if self.current_time.hour > 22: # disney closes at 22
            self.observation = None
            terminated = True
            info = None

        else: 
            terminated = False
            info = None

            # get the observation
            self.observation = self.__get_observation()

            # update current location
            if action != -1:
                self.current_location = action

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