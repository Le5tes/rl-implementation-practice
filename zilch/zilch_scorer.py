import numpy as np

class ZilchScorer:
  def score_dice(self, dice_numbers, action):
    counts = {1:0,2:0,3:0,4:0,5:0,6:0}
    for i, num in enumerate(dice_numbers):
      if action[i]:
        counts[num] += 1

    all_values = all(np.array([val for val in counts.values()]))
    
    three_twos = sum(np.array([val == 2 for val in counts.values()])) == 3
    
    if all_values or three_twos:
      return 1200, np.array([True for _ in range(6)])
    
    for i, val in counts.items():
      if val >= 3:
        score_multiplier = 10 if i == 1 else i
        score = score_multiplier * 100 * 2 ** (val - 3)
        dice_scored = (dice_numbers == i) & action
        other_score, other_dice_scored = self.score_dice(dice_numbers, action & (dice_numbers != i)) 
        return score + other_score, dice_scored | other_dice_scored
      
    dice_scored = ((dice_numbers == 1) | (dice_numbers == 5)) & action
    score = counts[1] * 100 + counts[5] * 50
    return score, dice_scored
