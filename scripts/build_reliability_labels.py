#!/usr/bin/env python3
"""Generate real GT-matched/unmatched detection labels from WEPDTOF + ByteTrack."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from reliability_al.data import build_dataset,save_dataset
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--tracks',required=True)
p.add_argument('--annotations',required=True)
p.add_argument('--mapping')
p.add_argument('--scene',default='repair_store')
p.add_argument('--output',default='data/real/repair_store_reliability.csv.gz')
p.add_argument('--iou',type=float,default=0.5)
p.add_argument('--iou-mode',choices=['obb','aabb'],default='obb')
p.add_argument('--train-end',type=int,default=620)
p.add_argument('--test-start',type=int,default=700)
p.add_argument('--require-all-frames',action='store_true')
a=p.parse_args()
d,summary=build_dataset(a.tracks,a.annotations,scene=a.scene,mapping_path=a.mapping,
     iou_threshold=a.iou,iou_mode=a.iou_mode,train_end=a.train_end,
     test_start=a.test_start,require_all_frames=a.require_all_frames)
meta=save_dataset(d,summary,a.output)
print('Ground-truth frames:',summary['annotated_frames'],'/',summary['prediction_frames'])
print('Predictions matched to GT:',summary['matched_predictions'])
print('Predictions unmatched:',summary['unmatched_predictions'])
print('Train/test distribution:')
print(d.groupby(['split','label']).size().to_string())
print('Saved:',a.output,'and',meta)
