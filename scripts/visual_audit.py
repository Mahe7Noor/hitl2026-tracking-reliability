#!/usr/bin/env python3
"""Create visual overlays to spot-check WEPDTOF GT vs ByteTrack match labels.

Blue OBB = GT-matched prediction; orange = unmatched prediction; green = GT.
Requires original image files and tracking CSV on yourAI; does not change labels.
"""
import argparse
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--tracks', required=True, help='Original ByteTrack tracks.csv with OBB geometry')
p.add_argument('--labels', required=True, help='Generated full reliability labels CSV.gz')
p.add_argument('--images', required=True, help='WEPDTOF original frames')
p.add_argument('--annotations', required=True, help='Per-frame WEPDTOF JSON annotations')
p.add_argument('--frames', default='1,204,590,713', help='Comma-separated 1-based frame numbers')
p.add_argument('--out', default='results/quality_audit')
a = p.parse_args()

tracks = pd.read_csv(a.tracks)
labels = pd.read_csv(a.labels)
key = ['frame', 'track_id']
if tracks.duplicated(key).any() or labels.duplicated(key).any():
    raise SystemExit('Duplicate frame/track_id rows detected')
if not set(key).issubset(tracks.columns) or not set(key+['label']).issubset(labels.columns):
    raise SystemExit('Missing required track/label fields')
joined = tracks.merge(labels[key+['label','matched_iou']], on=key, how='inner', validate='one_to_one')
if len(joined) != len(labels):
    raise SystemExit('Some labels have no corresponding track prediction')
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)


def contour(cx, cy, w, h, angle_degrees):
    rect = ((float(cx),float(cy)), (float(w),float(h)), float(angle_degrees))
    return np.int32(np.round(cv2.boxPoints(rect))).reshape((-1,1,2))

for frame in [int(x.strip()) for x in a.frames.split(',') if x.strip()]:
    f = joined[joined.frame==frame]
    if f.empty:
        raise SystemExit(f'No predictions for frame {frame}')
    filename = f.filename.iloc[0]
    image = cv2.imread(str(Path(a.images)/filename))
    if image is None:
        raise SystemExit(f'Cannot read image {filename}')
    gt_path = Path(a.annotations)/Path(filename).with_suffix('.json').name
    if not gt_path.exists():
        raise SystemExit(f'Missing GT JSON {gt_path}')
    gt = json.loads(gt_path.read_text(encoding='utf-8'))['annotations']
    for ann in gt:
        cx,cy,w,h,deg = ann['bbox']
        cv2.polylines(image,[contour(cx,cy,w,h,deg)],True,(38,175,72),1,cv2.LINE_AA)
    for r in f.itertuples():
        color = (226,132,48) if r.label==1 else (42,145,245)
        cv2.polylines(image,[contour(r.cx,r.cy,r.width,r.height,np.rad2deg(r.angle_radians))],True,color,2,cv2.LINE_AA)
        desc=f'ID {r.track_id} L{int(r.label)}'
        if pd.notna(r.matched_iou): desc+=f' IoU{r.matched_iou:.2f}'
        cv2.putText(image,desc,(max(5,int(r.cx-40)),max(16,int(r.cy-20))),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
    cv2.putText(image,f'Frame {frame} | GT green | matched blue | unmatched orange',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),2,cv2.LINE_AA)
    dest=out/f'frame_{frame:06d}_match_overlay.png'
    cv2.imwrite(str(dest),image)
    print(dest)
