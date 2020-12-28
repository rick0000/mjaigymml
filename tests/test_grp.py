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
    probs = [1.0/24] * 24

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

    for each_rank_prob in each_rank_probs:
        for p in each_rank_prob:
            assert (p - 0.25) < 10**-9  # float diff

    diff = 2**-4

    probs[0] += diff  # [0,1,2,3] means toimen 3rd, kamicha 4th
    probs[1] -= diff  # [0,1,3,2] means toimen 4th, kamicha 3rd

    # 対面の3位率が上がって4位率が下がる
    # 上家の3位率が下がって4位率が上がる
    new_each_rank_probs = grp.probs_to_each_ranks(probs)

    assert pytest.approx(each_rank_probs[2][2]) == \
        new_each_rank_probs[2][2] - diff
    assert pytest.approx(each_rank_probs[2][3]) == \
        new_each_rank_probs[2][3] + diff

    assert pytest.approx(each_rank_probs[3][2]) == \
        new_each_rank_probs[3][2] + diff
    assert pytest.approx(each_rank_probs[3][3]) == \
        new_each_rank_probs[3][3] - diff
