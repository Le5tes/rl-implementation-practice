from vpg_competitive import VPGCompetitive
from zilch.zilch_env import ZilchEnv

from torch.distributions.bernoulli import Bernoulli
from tqdm import tqdm

num_players = 2
reward_score_multiplyer = 0
num_eps = 5
num_epochs = 5

env = ZilchEnv(num_players, reward_score_multiplyer, goal_score=2000, end_turn_on_invalid=True)

agent = VPGCompetitive(21,6, distribution = Bernoulli)

for _ in tqdm(range(num_epochs)):
  rewards = agent.train_epoch(env, num_eps)
print("end")
