import conftest
import pytest
from mjaigymml.rewardpredictor.grp_dataset import GrpDataset


def test_rotate_scores():
    scores = [1000, 2000, 3000, 4000]
    grp = GrpDataset(
        honba=0,
        oya=0,
        kyotaku=0,
        before_scores=scores,
        end_scores=scores,
        label_scores=scores,
    )
    assert grp.oya_oriented_ranks == [3, 2, 1, 0]
    assert grp.label_class == 23


def test_prob_convert():
    probs = [0.25] * 24

    scores = [25000, 25000, 25000, 25000]
    grp = GrpDataset(
        honba=0,
        oya=0,
        kyotaku=0,
        before_scores=scores,
        end_scores=scores,
        label_scores=scores,
    )
    each_rank_probs = grp.probs_to_each_ranks(probs)
    assert len(each_rank_probs) == 4
