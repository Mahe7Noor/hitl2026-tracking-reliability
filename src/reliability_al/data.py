"""WEPDTOF tracking reliability: real ground-truth matched/unmatched detections.

A sample is ONE predicted ByteTrack bounding box in an annotated frame.
label=1: predicted OBB matched to one real WEPDTOF person OBB at IoU >= threshold.
label=0: predicted OBB was not matched to a WEPDTOF person OBB at threshold.
This is detection correctness at a chosen IoU threshold, NOT identity correctness.
No artificial labels are created. Ground-truth fields are audit-only, not features.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

#11 features extracted from YOLO OBB + Habbod and Bytetrack results

FEATURES = [
    'confidence', 'log_area', 'aspect', 'age', 'gap', 'speed',
    'size_change', 'angle_change', 'nearest_distance', 'max_overlap',
    'has_history',
]
NEEDED_TRACK_COLUMNS = {
    'frame','filename','track_id','cx','cy','width','height','angle_radians','confidence'
}


def read_tracks(path):
    d = pd.read_csv(path)
    absent = NEEDED_TRACK_COLUMNS - set(d.columns)
    if absent:
        raise ValueError('tracks CSV is missing: '+', '.join(sorted(absent)))
    if d.duplicated(['frame','track_id']).any():
        raise ValueError('Duplicate (frame,track_id) predictions')
    if (d[['width','height']] <= 0).to_numpy().any():
        raise ValueError('Nonpositive predicted box dimensions')
    if d[list(NEEDED_TRACK_COLUMNS - {'filename'})].isna().any().any():
        raise ValueError('Missing prediction numeric data')
    return d.sort_values(['frame','track_id']).reset_index(drop=True)


def _aabb_from_obb(row):
    x,y,w,h,ang = map(float,row)
    dx = (abs(np.cos(ang))*w+abs(np.sin(ang))*h)/2
    dy = (abs(np.sin(ang))*w+abs(np.cos(ang))*h)/2
    return np.array([x-dx,y-dy,x+dx,y+dy])


def _aabb_iou(a,b):
    lo=np.maximum(a[:2],b[:2]); hi=np.minimum(a[2:],b[2:])
    inter=float(np.maximum(0,hi-lo).prod())
    a_area=float(np.maximum(0,a[2:]-a[:2]).prod())
    b_area=float(np.maximum(0,b[2:]-b[:2]).prod())
    return inter / max(1.e-12,a_area+b_area-inter)


def _rect(row):
    x,y,w,h,t = map(float,row)
    return ((x,y),(w,h),float(np.rad2deg(t)))


def oriented_iou(box1,box2):
    """Actual oriented rectangle IoU; angle in radians for both inputs."""
    r1,r2 = _rect(box1),_rect(box2)
    result,poly = cv2.rotatedRectangleIntersection(r1,r2)
    if result == cv2.INTERSECT_NONE or poly is None:
        return 0.0
    intersection=float(abs(cv2.contourArea(cv2.convexHull(poly))))
    a=float(box1[2]*box1[3]);b=float(box2[2]*box2[3])
    return float(np.clip(intersection/max(1.e-12,a+b-intersection),0,1))


def _frame_neighbor_features(d):
    nearest=np.full(len(d),100.,dtype=float)
    overlap=np.zeros(len(d),dtype=float)
    for _,g in d.groupby('frame',sort=False):
        ix=g.index.to_numpy(); n=len(ix)
        if n<=1: continue
        xy=g[['cx','cy']].to_numpy(float)
        areas=np.sqrt((g.width*g.height).to_numpy(float).clip(1.e-6))
        dist=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=2)
        np.fill_diagonal(dist,np.inf)
        nearest[ix]=np.minimum(dist.min(axis=1)/areas,100.)
        rots=g[['cx','cy','width','height','angle_radians']].to_numpy(float)
        boxes=np.stack([_aabb_from_obb(row) for row in rots])
        for i in range(n):
            overlap[ix[i]]=max((_aabb_iou(boxes[i],boxes[j]) for j in range(n) if j!=i),default=0.)
    out=d.copy(); out['nearest_distance']=nearest;out['max_overlap']=overlap
    return out


def extract_features(tracks):
    """Eleven prediction-only features; no GT or labels used in these features."""
    d=_frame_neighbor_features(tracks)
    ordered=d.sort_values(['track_id','frame']).copy()
    grp=ordered.groupby('track_id',sort=False)
    prev={c:grp[c].shift(1) for c in ['frame','cx','cy','width','height','angle_radians']}
    area=(ordered.width*ordered.height).clip(lower=1.e-6)
    past_area=(prev['width']*prev['height']).clip(lower=1.e-6)
    gap=(ordered.frame-prev['frame']).fillna(0)
    ordered['log_area']=np.log(area)
    ordered['aspect']=np.minimum(ordered.width,ordered.height)/np.maximum(ordered.width,ordered.height)
    ordered['age']=grp.cumcount()+1
    ordered['gap']=gap
    ordered['speed']=(np.hypot(ordered.cx-prev['cx'],ordered.cy-prev['cy'])/
                     np.sqrt(area)/gap.replace(0,np.nan)).fillna(0.)
    ordered['size_change']=abs(np.log(area/past_area)).fillna(0.)
    delta=ordered.angle_radians-prev['angle_radians']
    ordered['angle_change']=abs((delta+np.pi/4)%(np.pi/2)-np.pi/4).fillna(0.)
    ordered['has_history']=prev['frame'].notna().astype(int)
    return ordered.sort_index().reset_index(drop=True)


def read_ground_truth(annotation_dir):
    root=Path(annotation_dir)
    if not root.is_dir(): raise FileNotFoundError('Missing annotation directory: '+str(root))
    refs={}
    for path in sorted(root.glob('*.json')):
        js=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(js,dict) or 'image' not in js or 'annotations' not in js:
            raise ValueError('Invalid per-frame WEPDTOF JSON: '+str(path))
        key=js['image'].get('file_name')
        if not key or key in refs: raise ValueError('Missing/duplicate annotated filename: '+str(key))
        anns=js['annotations']
        for ann in anns:
            bb=ann['bbox']
            if len(bb)!=5: raise ValueError('Expected WEPDTOF 5-value oriented boxes')
            if float(bb[2])<=0 or float(bb[3])<=0:raise ValueError('Invalid GT dimensions')
        refs[key]=anns
    if not refs:raise ValueError('No WEPDTOF frame annotations found in '+str(root))
    return refs


def _valid_hungarian(iou_matrix,threshold):
    """One-to-one matching maximizing valid count, then IoU (GT rows, pred cols)."""
    if not iou_matrix.size:return []
    valid=iou_matrix >= threshold
    # Important: prefer the maximum NUMBER of valid matches before IoU sum.
    score=np.where(valid,100. + iou_matrix,0.)
    rows,cols=linear_sum_assignment(-score)
    return [(int(i),int(j),float(iou_matrix[i,j])) for i,j in zip(rows,cols) if valid[i,j]]


def build_dataset(tracks_path, annotations_dir, *, scene='repair_store', mapping_path=None,
                  iou_threshold=.5, iou_mode='obb', train_end=620,test_start=700,
                  require_all_frames=False):
    """GT-matched predictions, with a temporal embargo between train/test.

    For full Repair Store use train_end=620/test_start=700. For 18-frame local
    proof-of-pipeline use train_end=12/test_start=15. Never use the latter pilot
    as an independent estimate of full-scene performance.
    """
    if not (0 < iou_threshold <= 1):raise ValueError('iou_threshold must be (0,1]')
    if iou_mode not in ('obb','aabb'):raise ValueError('iou_mode must be obb/aabb')
    if not (1<=train_end<test_start-1):raise ValueError('Use a nonempty temporal embargo')
    tracks=read_tracks(tracks_path)
    d=extract_features(tracks)
    gts=read_ground_truth(annotations_dir)
    frames=d[['frame','filename']].drop_duplicates()
    if mapping_path:
        mapping=pd.read_csv(mapping_path)
        if not {'frame','filename'}.issubset(mapping.columns):raise ValueError('frame_mapping.csv lacks frame,filename')
        frames=mapping[['frame','filename']].copy()
        if frames.frame.duplicated().any():raise ValueError('Duplicate frame in mapping')
        # Do not allow a mapping that refers to the wrong image for a prediction.
        sample=tracks[['frame','filename']].drop_duplicates()
        check=sample.merge(frames,on='frame',how='left',suffixes=('_tracks','_mapping'))
        if check.filename_mapping.isna().any() or (check.filename_tracks!=check.filename_mapping).any():
            raise ValueError('Frame mapping does not agree with predictions')
    if frames.frame.duplicated().any():raise ValueError('Frame has more than one filename')
    annotated=frames[frames.filename.isin(gts)]
    if require_all_frames and len(annotated)!=len(frames):
        raise ValueError(f'Only {len(annotated)} of {len(frames)} predicted frames have GT. Full GT needed.')
    d=d[d.frame.isin(annotated.frame)].copy()
    d['matched_gt_person_id']=pd.NA
    d['matched_iou']=np.nan
    d['label']=0
    group_positions=d.groupby('frame',sort=False).groups
    frame_to_filename=dict(annotated.itertuples(index=False,name=None))
    matched_count=0; unmatched_count=0
    for frame,indices in group_positions.items():
        ix=np.asarray(list(indices))
        f=d.loc[ix]
        name=frame_to_filename[frame]
        anns=gts[name]
        pred=f[['cx','cy','width','height','angle_radians']].to_numpy(float)
        gt=np.array([z['bbox'] for z in anns],dtype=float).reshape(-1,5)
        if len(gt):gt[:,4]=np.deg2rad(gt[:,4])
        sims=np.zeros((len(gt),len(pred)),dtype=float)
        for i,ga in enumerate(gt):
            for j,pb in enumerate(pred):
                if iou_mode=='obb':sims[i,j]=oriented_iou(ga,pb)
                else:sims[i,j]=_aabb_iou(_aabb_from_obb(ga),_aabb_from_obb(pb))
        pairs=_valid_hungarian(sims,iou_threshold)
        for i,j,score in pairs:
            d.loc[ix[j],'label']=1
            d.loc[ix[j],'matched_gt_person_id']=str(anns[i].get('person_id',''))
            d.loc[ix[j],'matched_iou']=score
        matched_count+=len(pairs)
        unmatched_count+=len(ix)-len(pairs)
    d['split']=np.select([d.frame<=train_end,d.frame>=test_start],['train','test'],default='excluded')
    d['scene']=scene
    d['event_id']=(d.scene.astype(str)+'_'+d.frame.astype(str)+'_'+d.track_id.astype(str))
    d=d[['scene','split','event_id','frame','filename','track_id',*FEATURES,
         'label','matched_gt_person_id','matched_iou']].reset_index(drop=True)
    audit={
        'source':'real_habbof_bytetrack_predictions_and_wepdtof_annotations',
        'target':'label=1 predicted OBB matched to GT OBB; label=0 unmatched prediction (not identity correctness)',
        'matching':f'one-to-one {iou_mode.upper()} IoU >= {iou_threshold}, maximum valid assignments first',
        'annotated_frames':int(len(annotated)),
        'prediction_frames':int(len(frames)),
        'prediction_rows_in_annotated_frames':int(len(d)),
        'matched_predictions':int(matched_count),
        'unmatched_predictions':int(unmatched_count),
        'train_end':train_end,'test_start':test_start,
        'train_test_gap':list(range(train_end+1,test_start)),
        'split_labels':{name:{str(v):int(k) for v,k in sub.label.value_counts().items()}
                        for name,sub in d.groupby('split')},
        'limitations':[
            'A GT-unmatched prediction might still depict a person but fail the IoU threshold.',
            'Missed GT persons (false negatives) are not in the prediction-row classifier.',
            'Person identity is NOT a classifier feature or a classification target here.',
            'Same-scene temporal train/test blocks remain correlated; scene-disjoint holdout preferred.',
        ],
    }
    return d,audit


def save_dataset(d,audit,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(path,index=False,compression='gzip' if path.suffix=='.gz' else None)
    name=path.with_suffix('').with_suffix('.metadata.json') if path.suffix=='.gz' else path.with_suffix('.metadata.json')
    name.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    return name
