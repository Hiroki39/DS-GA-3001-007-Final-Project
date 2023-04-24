import numpy as np
import pandas as pd
import googlemaps
from dotenv import load_dotenv
import os

load_dotenv()

gmaps = googlemaps.Client(key=os.environ.get("GOOGLE_MAP_API"))
land_df = pd.read_csv("./disneyenv/disneyenv/envs/data/landLocation.csv")

long_lats = land_df[["longitude", "latitude"]].apply(tuple, axis=1).tolist()

# initialize walking time matrix
walking_time = np.zeros((len(long_lats), len(long_lats)))

for i in range(len(long_lats)):
    for j in range(len(long_lats)):
        res = gmaps.distance_matrix(
            origins=long_lats[i],
            destinations=long_lats[j],
            mode="walking",
        )
        # record the walking time in minutes
        walking_time[i, j] = res["rows"][0]["elements"][0]["duration"]["value"] / 60

# save as a .npy file
np.save("./disneyenv/disneyenv/envs/data/walking_time.npy", walking_time)
