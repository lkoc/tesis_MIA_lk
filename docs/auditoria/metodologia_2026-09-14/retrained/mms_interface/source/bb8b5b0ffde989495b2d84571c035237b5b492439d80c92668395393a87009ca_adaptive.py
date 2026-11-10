"""Auditable fixed-size adaptive resampling, inspired by RAD (Wu et al., 2022)."""
import numpy as np

def probabilities(scores,mode):
    if mode not in ('uniform','residual','gradient'):raise ValueError('Unknown adaptive sampling mode')
    score=np.asarray(scores,dtype=float)
    if score.ndim!=1 or not len(score) or not np.isfinite(score).all() or np.any(score<0):raise ValueError('Finite nonnegative scores required')
    if mode=='uniform' or score.mean()<=1e-15:return np.full(len(score),1/len(score))
    weights=score/score.mean()+1.
    return weights/weights.sum()

def resample(initial,candidates,scores,seed,mode,fraction=.5):
    if not 0<fraction<=1:raise ValueError('Adaptive fraction must be in (0,1]')
    rng=np.random.default_rng(seed);n=len(initial);count=int(round(n*fraction))
    if len(candidates)<count:raise ValueError('Candidate cloud too small')
    p=probabilities(scores,mode)
    anchors=rng.choice(n,n-count,replace=False);chosen=rng.choice(len(candidates),count,replace=False,p=p)
    cloud=np.vstack([initial[anchors],candidates[chosen]])
    return cloud,dict(anchor_indices=anchors,selected_indices=chosen,probabilities=p)
