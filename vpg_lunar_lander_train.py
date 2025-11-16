import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from vanilla_policy_gradient import VPG

num_eps = 100
num_epochs = 200

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)

wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length = num_eps)

agent = VPG(8, 4, learning_rate=0.004)

epoch_rewards = []

for _ in tqdm(range(num_epochs)):
    rewards = agent.train_epoch(wrapped_env, num_eps)
    epoch_rewards.append(rewards)
    agent.set_learning_rate(agent.learning_rate * 0.99)
print("end")

ep_rewards = [reward for ep_rewards in epoch_rewards for reward in ep_rewards]

fig,ax = plt.subplots()

ax.plot(ep_rewards)

plt.show()

agent.save("vpg_lunar_lander_agent.pkl")