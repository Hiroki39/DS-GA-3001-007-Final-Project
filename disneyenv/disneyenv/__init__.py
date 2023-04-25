from gym.envs.registration import register

register(
    id="disneyenv/Disney-v0",
    entry_point="disneyenv.envs:DisneyEnv",
    max_episode_steps=500,
)
