from datetime import datetime, timedelta
import numpy as np
import pandas as pd


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


class DisneyEnvironment:
    def __init__(self,waittime_df,weather_df,rides_df,adjacency_matrix):

        # Dataframe for extracting data
        self.waittime = waittime_df
        self.weather = weather_df
        self.ridesinfo = rides_df
        self.rides = rides_df.id.to_numpy()
        self.avalible_days = set(self.waittime.day.unique())
        self.observation = None

        # Action space
        self.__action_space = np.append(np.arange(len(self.rides)),-1) # -1 indicates wait for 10 min

        # adjecency matrix
        self.adjacency_matrix = adjacency_matrix

        # The current day we are in
        self.current_day = None
        self.current_time = None
        self.current_location = None
        self.waittime_today = None
        self.weather_today = None
        self.past_actions = None


        # reward
        self.reward_dict= {
             "MAA":20,
             "MIA":10,
             "HD":20
        }
        
    def __get_observation(self):

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
        Past Actions: A vector of 0 and 1 showing rides haven't been done [110]
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


        # REWARD
        popularity = self.ridesinfo.iloc()[action]["popularity"]
        if type(popularity) == str:
            reward = self.reward_dict[popularity]
        else: # popularity is none
            reward = 1
        
        '''
        discount the reward using past_actions: if a ride is popular, one would ride is several times. 
        Less popular rides would not be rided for several times
        '''
        if action != -1:
            reward /= (2**(self.past_actions[action]))
        
        # STATE
        # update pass actions
        if action != -1:
            self.past_actions[action] += 1

            # Next time stamp
            travel_duration = self.adjacency_matrix[self.ridesinfo.iloc()[action].landID][self.ridesinfo.iloc()[self.current_location].landID]
            wait_duration = self.observation[action] # self.observation is a attribute since we need to use it here
            ride_duration = self.ridesinfo.duration_min[action]
        else:
            travel_duration = 0
            wait_duration = 10
            ride_duration = 0
        
        self.current_time += timedelta(minutes=(travel_duration+wait_duration+ride_duration))

        # if the day is done
        if self.current_time.hour > 22: # disney closes at 22
            self.observation = None
            terminated = True
            info = None
            return self.observation,reward,terminated,info
        else: 
            terminated = False
            info = None

            # get the observation
            self.observation = self.__get_observation()

            # update current location
            self.current_location = action

            return self.observation, reward, terminated, info

    def get_action_space(self):
        # method that returns possible actions.
        if self.observation is None:
            return np.array([])
        else:
            possible_action = []
            for index,content in enumerate(self.__action_space):
                if content == -1:
                    possible_action += [content]
                elif (self.observation[index] != 999) and (~np.isnan(self.observation[index])):
                    possible_action += [content]
            return np.array(possible_action)
                


    def close():
        pass