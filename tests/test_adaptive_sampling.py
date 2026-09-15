import numpy as np
import pytest
from Benchmarks.adaptive import probabilities,resample

def test_score_weighting_and_uniform_control():
    score=np.r_[np.ones(90),np.full(10,100.)]
    p=probabilities(score,'residual')
    assert p[-10:].sum()>.4
    assert np.allclose(probabilities(score,'uniform'),.01)
    assert np.allclose(probabilities(np.zeros(100),'gradient'),.01)

def test_unknown_mode_rejected():
    with pytest.raises(ValueError):probabilities([1,2],'typo')

def test_resampling_replays_exactly_and_retains_coverage():
    initial=np.arange(200).reshape(100,2);candidates=np.arange(200,2200).reshape(1000,2)
    a,info=resample(initial,candidates,np.arange(1000),23,'residual')
    b,other=resample(initial,candidates,np.arange(1000),23,'residual')
    assert np.array_equal(a,b) and a.shape==initial.shape
    assert np.array_equal(a[:50],initial[info['anchor_indices']])
    assert np.array_equal(a[50:],candidates[info['selected_indices']])
    assert len(set(info['selected_indices']))==50
    assert np.array_equal(info['probabilities'],other['probabilities'])

@pytest.mark.parametrize('score',[[np.nan,1],[-1,2],[],[[1,2]]])
def test_invalid_scores_rejected(score):
    with pytest.raises(ValueError):probabilities(score,'residual')
