
from vanilla_policy_gradient import VPG
from torch import as_tensor

win_bonus = 100

## An alternative version of the VPG training where it trains the agent to play against itself
class VPGCompetitive(VPG):
  def train_episode(self, env, players = 2):
    (state, _) = env.reset()

    transitions = []
    ep_rewards = [0 for _ in range(players)]
    current_player = 0

    finished = False

    while not finished:
      action_distribution = self.policy(as_tensor(state))
      action = action_distribution.sample()
      transitions.append((state,action.tolist(), current_player))
      (state, reward, terminated, truncated, info) = env.step(action.numpy().astype('int'))

      if info['end_turn']:
        current_player += 1
        # print("VPGCompetative.train_episode next player turn")
        if current_player >= players:
          current_player = 0
      
      ep_rewards[current_player] += reward
      finished = terminated or truncated
      if finished:
        ep_rewards[info['winner']] += win_bonus


    print("VPGCompetative.train_episode end of episode")
    return [(state, action, ep_rewards[current_player]) for state, action, current_player in transitions]


