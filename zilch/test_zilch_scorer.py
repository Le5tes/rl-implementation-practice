from zilch.zilch_scorer import ZilchScorer
import numpy as np

zilch_scorer = ZilchScorer()

def test_zilch_scorer_should_return_1200_and_all_dice_scored_when_is_run():
  action = np.array([1,1,1,1,1,1])
  dice = np.array([1,2,3,4,5,6])

  total_score, dice_scores = zilch_scorer.score_dice(dice, action)

  assert total_score == 1200

  for i in range(6):
    assert dice_scores[i]


def test_zilch_scorer_should_return_1200_and_all_dice_scored_when_is_3_pairs():
  action = np.array([1,1,1,1,1,1])
  dice = np.array([1,1,3,3,6,6])

  total_score, dice_scores = zilch_scorer.score_dice(dice, action)

  assert total_score == 1200

  for i in range(6):
    assert dice_scores[i]

def test_zilch_scorer_should_return_500_and_those_3_scored_if_3_fives():
  action = np.array([1,1,1,1,1,1])
  dice = np.array([5,5,3,5,6,6])

  total_score, scored = zilch_scorer.score_dice(dice, action)
  assert total_score == 500
  assert all(scored == np.array([True, True, False, True, False, False]))

def test_3_1s_2_5s():
  action = np.array([1,1,1,1,1,1])
  dice = np.array([1,6,1,1,5,5])

  total_score, scored = zilch_scorer.score_dice(dice, action)

  assert total_score == 1100
  assert all(scored == np.array([True, False, True, True, True, True]))

def test_4_4s_plus_1_not_included():
  action = np.array([1,1,1,0,1,1])
  dice = np.array([4,6,4,4,4,4])

  total_score, scored = zilch_scorer.score_dice(dice, action)

  assert total_score == 800
  print(scored)
  assert all(scored == np.array([True, False, True, False, True, True]))