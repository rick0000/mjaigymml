import conftest
import pytest
from mjaigymml.rewardpredictor.global_reward_predictor import GlobalRewardPredictor


def test_seat0_index0_to_oya_index0():

    grp = GlobalRewardPredictor()
    scores = [1000, 2000, 3000, 4000]

    moved = grp.change_seat0_index0_to_oya_index0(scores, 0)
    assert moved == [scores[0], scores[1], scores[2], scores[3]]

    moved = grp.change_seat0_index0_to_oya_index0(scores, 1)
    assert moved == [scores[1], scores[2], scores[3], scores[0]]

    moved = grp.change_seat0_index0_to_oya_index0(scores, 2)
    assert moved == [scores[2], scores[3], scores[0], scores[1]]

    moved = grp.change_seat0_index0_to_oya_index0(scores, 3)
    assert moved == [scores[3], scores[0], scores[1], scores[2]]

    with pytest.raises(Exception):
        moved = grp.change_seat0_index0_to_oya_index0(scores, 4)
        moved = grp.change_seat0_index0_to_oya_index0(scores, -1)


def test_oya_index0_to_seat0_index0():
    grp = GlobalRewardPredictor()
    preds = [1000, 2000, 3000, 4000]

    moved = grp.change_oya_index0_to_seat0_index0(preds, 0)
    assert moved == [preds[0], preds[1], preds[2], preds[3]]

    moved = grp.change_oya_index0_to_seat0_index0(preds, 1)
    assert moved == [preds[3], preds[0], preds[1], preds[2]]

    moved = grp.change_oya_index0_to_seat0_index0(preds, 2)
    assert moved == [preds[2], preds[3], preds[0], preds[1]]

    moved = grp.change_oya_index0_to_seat0_index0(preds, 3)
    assert moved == [preds[1], preds[2], preds[3], preds[0]]

    with pytest.raises(Exception):
        moved = grp.change_oya_index0_to_seat0_index0(preds, 4)
        moved = grp.change_oya_index0_to_seat0_index0(preds, -1)
