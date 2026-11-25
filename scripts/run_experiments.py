"""Four AL policies, five seeds, six rounds; predict GT-unmatched detections."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
from reliability_al.experiment import Settings,run_experiments,save_experiment,final_table
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset',default='data/real/repair_store_reliability.csv.gz')
p.add_argument('--out',default='results/full')
p.add_argument('--batch-size',type=int,default=20)
p.add_argument('--rounds',type=int,default=6)
p.add_argument('--initial-matched',type=int,default=10)
p.add_argument('--initial-unmatched',type=int,default=10)
a=p.parse_args()
s=Settings(initial_positive=a.initial_matched,initial_negative=a.initial_unmatched,
           batch_size=a.batch_size,rounds=a.rounds)
d=pd.read_csv(a.dataset)
r,q=run_experiments(d,s)
save_experiment(r,q,a.out,s)
print('Final results (unmatched-class AP; five seeds):')
print(final_table(r).to_string())
print('Saved',a.out)
