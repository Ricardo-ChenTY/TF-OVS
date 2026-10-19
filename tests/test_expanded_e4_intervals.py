"""Check aggregation and pairing, not an assumption that timing is stationary."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_proposal_e4_expanded import bootstrap,interval

def test_equal_target_mean_is_not_image_weighted():
    per,macro=bootstrap([np.full(2,1.),np.full(9,3.)],draws=100)
    assert np.all(macro==2.)
    assert interval(macro)==[2.,2.]

def test_paired_constant_overhead_cancels_image_difficulty():
    baseline=np.array([1.,7.,2.,20.])
    differences=(baseline+.125)-baseline
    _,macro=bootstrap([differences,differences],draws=100)
    assert interval(macro)==[.125,.125]

def test_bootstrap_reproducible_and_non_degenerate():
    arrays=[np.arange(10,dtype=float),np.arange(20,dtype=float)]
    _,a=bootstrap(arrays,draws=500,seed=7)
    _,b=bootstrap(arrays,draws=500,seed=7)
    assert np.array_equal(a,b)
    lo,hi=interval(a)
    assert lo<7.<hi
