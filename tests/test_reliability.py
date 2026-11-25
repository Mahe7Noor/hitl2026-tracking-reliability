"""Guard against fabricated labels, data leakage, and unfair acquisition budgets."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from reliability_al.data import FEATURES,oriented_iou,_valid_hungarian,extract_features
from reliability_al.experiment import Settings,run_experiments,STRATEGIES
ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'data/pilot/repair_store_first18_reliability.csv.gz'


def test_obb_iou_geometry():
    a=np.array([100.,100.,50.,30.,np.pi/5])
    assert oriented_iou(a,a)==pytest.approx(1,rel=1e-4)
    b=np.array([500.,500.,50.,30.,0.])
    assert oriented_iou(a,b)==pytest.approx(0.)


def test_one_to_one_matching_and_iou_threshold():
    # Two predicted boxes compete for a single GT object, one must stay unmatched.
    pairs=_valid_hungarian(np.array([[.81,.78]]),.5)
    assert pairs==[(0,0,.81)]
    assert _valid_hungarian(np.array([[.49,.48]]),.5)==[]


def test_pilot_has_only_real_gt_match_labels():
    df=pd.read_csv(PILOT)
    assert len(df)==243
    assert df.label.isin([0,1]).all()
    assert int(df.label.sum())==101
    assert int((df.label==0).sum())==142
    assert 'matched_gt_person_id' in df
    assert df.loc[df.label==1,'matched_iou'].ge(.5-1.e-6).all()
    assert df.loc[df.label==0,'matched_iou'].isna().all()
    assert not {'label','matched_gt_person_id','matched_iou','track_id','frame','filename','event_id'} & set(FEATURES)
    assert set(df.split)=={'train','test','excluded'}
    assert df.loc[df.split=='train','frame'].max()<=12
    assert df.loc[df.split=='test','frame'].min()>=15
    assert df.event_id.is_unique


def test_fair_reproducible_sampling():
    d=pd.read_csv(PILOT)
    config=Settings(seeds=(17,),initial_positive=4,initial_negative=4,batch_size=3,rounds=2)
    r1,q1=run_experiments(d,config)
    r2,q2=run_experiments(d,config)
    pd.testing.assert_frame_equal(r1,r2)
    pd.testing.assert_frame_equal(q1,q2)
    assert r1.loc[r1['round']==0,'average_precision'].nunique()==1
    assert set(r1.strategy)==set(STRATEGIES)
    assert q1.groupby(['strategy','round']).size().eq(3).all()
    assert not q1.duplicated(['strategy','event_id']).any()
    assert set(r1['total_labels'])=={8,11,14}


def test_zero_class_holdout_rejected():
    from reliability_al.experiment import validate_dataset
    d=pd.read_csv(PILOT)
    d.loc[d.split=='test','label']=1
    with pytest.raises(ValueError,match='lacks one class'):
        validate_dataset(d,Settings(initial_positive=4,initial_negative=4,batch_size=3,rounds=2))
