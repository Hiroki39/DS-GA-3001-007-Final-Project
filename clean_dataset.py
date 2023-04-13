import pandas as pd

waittime_df = pd.read_csv("./disneyenv/envs/data/disneyRideTimes.csv")
open_df = waittime_df[waittime_df["status"] == "Operating"]

# remove rides that are not operating on any day
