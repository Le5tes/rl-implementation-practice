import numpy as np

from zilch.zilch_scorer import ZilchScorer

class ZilchEnv:
  def __init__(self, num_players, reward_score_multiplier = 0, goal_score = 10000, end_turn_on_invalid = False):
    self.turn_num = 0
    self.turn_score = 0
    self.goal_score = goal_score
    self.reward_score_multiplier = reward_score_multiplier
    self.current_player = 0
    self.num_players = num_players
    self.total_scores = [0 for _ in range(num_players)]
    self.dice_state = [
      np.array(['free', 'free', 'free', 'free', 'free', 'free']), # free or kept
      np.array([0,0,0,0,0,0]), # number
      np.array([0,0,0,0,0,0]) # score
    ]
    self.zilch_scorer = ZilchScorer()
    self.step_count = 0
    self.end_turn_on_invalid = end_turn_on_invalid

  # Useful for modifying during training - for early epochs set a lower goal score to make it easier!
  def set_goal_score(self, goal_score):
    self.goal_score = goal_score

  def reset(self):
    self.turn_num = 0
    self.current_player = 0
    self.total_scores = [0 for _ in range(self.num_players)]
    self.reset_dice()

    return self.state(), None

  def reset_dice(self):
    self.dice_state[0] = np.array(['free', 'free', 'free', 'free', 'free', 'free'])
    self.dice_state[2] = np.array([0,0,0,0,0,0])
    self.roll()
  
  def state(self):
    highest_competitor_score = max(self.total_scores[:self.current_player] + self.total_scores[self.current_player + 1 :])
    print(f"ZilchEnv.state - self.dice_state: {self.dice_state}")
    return np.concatenate([
      np.array([
        self.total_scores[self.current_player], 
        highest_competitor_score,
        self.turn_score
      ]),
      self.dice_state[0] == 'kept',
      self.dice_state[1],
      self.dice_state[2]
    ], dtype='float32')

  # TODO: any other invalid cases?
  def is_valid(self, action):
    score, scored_dice = self.check_score()

    bank_all = all(action | (self.dice_state[0] == 'kept'))

    # can't bank non-scoring dice (unless banking all remaining dice)
    if not bank_all and any(action & np.invert(scored_dice)):
      return False

    # can't bank all remaining dice if total score would not be above 250
    if bank_all and ((score + self.turn_score) < 250):
      return False

    return True

  def is_finished(self):
    return any(np.array(self.total_scores) >= self.goal_score) and self.current_player == 0

  def next_player(self):
    self.current_player += 1
    if self.current_player >= self.num_players:
      self.turn_num += 1
      self.current_player = 0

  def end_turn(self):
    self.total_scores[self.current_player] += self.turn_score
    # print(f"ZilchEnv.end_turn - player {self.current_player}'s current score: {self.total_scores[self.current_player]}")
    self.next_player()
    # print(f"ZilchEnv.end_turn - player {self.current_player}'s turn")
    return self.reset_dice()

  def check_score(self):
    return self.zilch_scorer.score_dice(self.dice_state[1], self.dice_state[0] == 'free')

  def score_dice(self, action):
    score, dice_scored = self.zilch_scorer.score_dice(self.dice_state[1],action)

    self.dice_state[2] = dice_scored

    return score
  
  def roll(self):
    for i in range(6):
      if self.dice_state[0][i] == 'free':
        self.dice_state[1][i] = np.random.randint(1,6)
    

  # Action - agent is presented dice numbers having been rolled ad choses which ones to keep
  # returns state, reward, terminated, truncated, info
  def step(self, action):
    if not self.is_valid(action):
      reward = -1
      info = {'end_turn': False, 'is_valid_action': False}
      self.step_count += 1
      if self.step_count % 100 == 0:
        print(f"ZilchEnv.step - step count = {self.step_count}, action: {action} (INVALID)")
      if self.end_turn_on_invalid:
        self.end_turn()
        info['end_turn'] = True
      return self.state(), reward, False, False, info
    
    info = {'end_turn': False, 'winner': None, 'is_valid_action': True}
    action = action & (self.dice_state[0] == 'free') # can only keep dice that aren't already kept

    for i, die in enumerate(action):
      if die:
        self.dice_state[0][i] = 'kept'

    dice_score = self.score_dice(action)
    
    self.turn_score += dice_score
    
    if all(self.dice_state[0] == 'kept'):
      if all(self.dice_state[2]):
        self.dice_state[0] = ['free', 'free', 'free', 'free', 'free', 'free']
      else:
        self.end_turn()
        info['end_turn'] = True
    else:
      self.roll()
      score, _ = self.check_score()
      zilched = score == 0
      if zilched:
        dice_score = - self.turn_score
        self.turn_score = 0
        self.end_turn()
        info['end_turn'] = True

    if self.is_finished():
      info['winner'] = np.argmax(self.total_scores)
      print(f"ZilchEnv.step - finished, winner is {info['winner']}")

    self.step_count += 1
    if self.step_count % 100 == 0:
      print(f"ZilchEnv.step - step count = {self.step_count}, action: {action}")

    return self.state(), dice_score * self.reward_score_multiplier, self.is_finished(), False, info
