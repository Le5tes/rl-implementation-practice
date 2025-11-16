from torch import nn, as_tensor, save, load
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np

def build_model(observation_size, action_size):
  return nn.Sequential(
    nn.Linear(observation_size, 256),
    nn.ReLU(),
    nn.Linear(256,256),
    nn.ReLU(),
    nn.Linear(256 , action_size)
  )

class VPG:
  def __init__(self, observation_size, action_size, learning_rate = 0.001):
    self.learning_rate = learning_rate
    self.observation_size = observation_size
    self.action_size = action_size
    self.policy_model = build_model(observation_size, action_size)
    self.optimiser = Adam(self.policy_model.parameters(), lr = self.learning_rate)

  def set_learning_rate(self, learning_rate):
    self.learning_rate = learning_rate 
    self.optimiser = Adam(self.policy_model.parameters(), lr = self.learning_rate)

  def policy(self, observation):
    logits = self.policy_model(observation)
    return Categorical(logits = logits)
  
  def compute_loss(self, transitions):
    state, actions, rewards = tuple(zip(*transitions))
    state, actions, rewards = as_tensor(np.array(state)), as_tensor(actions), as_tensor(rewards)

    log_probs = self.policy(state).log_prob(actions)

    return - (log_probs * rewards).mean()

  
  def train_epoch(self, env, num_episodes):
    
    ep_transitions = []
    for _ in range(num_episodes):
      ep_transitions.append(self.train_episode(env))

    ep_rewards = [transition_batch[0][2] for transition_batch in ep_transitions]

    transitions = [transition for transition_batch in ep_transitions for transition in transition_batch]

    self.optimiser.zero_grad()
    loss = self.compute_loss(transitions)

    loss.backward()
    self.optimiser.step()

    return ep_rewards

  
  def train_episode(self, env):
    (state, _) = env.reset()
    transitions = []

    ep_reward = 0
    finished = False

    while not finished:
      action_distribution = self.policy(as_tensor(state))
      action = action_distribution.sample()
      transitions.append((state,action.item()))
      (state, reward, terminated, truncated, _) = env.step(action.item())
      
      ep_reward += reward
      finished = terminated or truncated


    return [(state, action, ep_reward) for state, action in transitions]
  
  def save(self, filepath):
    save(self.policy_model.state_dict(), filepath)

  def load(self, filepath):
    self.policy_model.load_state_dict(load(filepath, weights_only=True))
    self.optimiser = Adam(self.policy_model.parameters(), lr = self.learning_rate)
