#!/usr/bin/env python3
"""Check that FULL data, five-seed curves and saved notebook outputs exist.

Run from project root on yourAI after rerunning and saving BOTH notebooks.
Does not check scientific truth of annotations: that needs independent visual auditing.
"""
from pathlib import Path
import sys
import json
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[1]
issues = []

def require(condition, message):
    print(('PASS: ' if condition else 'FAIL: ') + message)
    if not condition:
        issues.append(message)

full_data = ROOT / 'data/real/repair_store_reliability.csv.gz'
require(full_data.exists(), 'full 947-frame GT-labeled dataset exists on cluster')
if full_data.exists():
    d = pd.read_csv(full_data)
    require(len(d) == 13123, '13,123 prediction rows (25 Nov reported experiment)')
    require(int(d.label.eq(0).sum()) == 8473 and int(d.label.eq(1).sum()) == 4650,
            'GT-match label counts agree with terminal result')
    require(set(d.split.unique()) == {'train','test','excluded'}, 'all three temporal partitions exist')

runs_path = ROOT / 'results/full/all_runs.csv'
require(runs_path.exists(), 'results/full/all_runs.csv from full experiment exists')
if runs_path.exists():
    runs = pd.read_csv(runs_path)
    require(len(runs) == 140, '5 seeds × 4 policies × 7 evaluation points = 140 rows')
    require(runs.seed.nunique() == 5, '5 random seeds present')
    require(runs.strategy.nunique() == 4, '4 acquisition strategies present')
    require(set(runs['round'].unique()) == set(range(7)), 'initial + 6 acquisition rounds present')
    require(set(runs.loc[runs['round'] == 6,'total_labels']) == {140}, '140 labeled examples at final budget')

for name in ['learning_curves.png','learning_curves_f1.png','label_counts.png']:
    require((ROOT/'results/full'/name).exists(), 'full figure: '+name)

for name in ['01_active_learning_tracking_reliability.ipynb','00_data_audit.ipynb']:
    p=ROOT/'active_learning'/name
    require(p.exists(), 'notebook exists: '+name)
    if not p.exists():
        continue
    nb=nbformat.read(p,as_version=4)
    texts=[]
    errors=[]
    for cell in nb.cells:
        for item in cell.get('outputs',[]):
            if item.output_type=='stream':texts.append(item.get('text',''))
            if item.output_type=='error':errors.append(item.get('evalue',''))
    stdout='\n'.join(texts)
    require('DATA MODE: FULL' in stdout, 'saved full-data execution outputs in '+name)
    require(not errors, 'no saved notebook execution errors in '+name)

if issues:
    print('\nNOT READY FOR SUBMISSION: '+str(len(issues))+' check(s) failed. Fix before GitHub commit.')
    raise SystemExit(1)
print('\nFULL-DATA ARTIFACT CHECKS PASS. Still inspect notebook figures and source/data permissions before pushing to GitHub.')
