import gymnasium as gym

from vanilla_policy_gradient import VPG

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="human")

agent = VPG(8, 4, learning_rate=0.004)
agent.load("vpg_lunar_lander_agent.pkl")

agent.train_epoch(env, 30)