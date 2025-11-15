import gymnasium as gym
from vanilla_policy_gradient import VPG

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)

wrapped_env = gym.wrappers.RecordEpisodeStatistics(env)

agent = VPG(8, 4)

agent.train_epoch(wrapped_env, 5)
print("end")